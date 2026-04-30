import ast
from tqdm import tqdm
import numpy as np
import open_clip
from src.moderation_clip import ModerationHeadMLP
import types
import torch.nn.functional as F
from itertools import combinations
from transformers import CLIPTokenizer
import itertools
import torch


STOPWORDS = {"in", "a", "an", "with", "the", "'s", "are", "is", "has", "can", "from", "and", "of", "on", "at", "by",
             "for", "to"}
PUNCTUATION = {".", ",", ";", ":", "!", "?", "'", "\"", "(", ")", "..."}
NEUTRAL_WORDS = STOPWORDS.union(PUNCTUATION)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

moderation_model = None
head_moderation = {}
shared_info = {"eos_positions": None}

layer_number = 4
head_number = 6


def attention_hook(module, inputs, outputs):
    # 1) Recuperiamo input e output e li spostiamo su GPU
    q, k, v = [t.to(device) for t in inputs[:3]]
    attn_output, attn_weights = outputs
    attn_weights = attn_weights.to(device)

    # 2) Assicuriamoci che tutti i parametri siano su GPU
    E = module.embed_dim
    num_heads = module.num_heads
    head_dim = E // num_heads

    W_in = module.in_proj_weight
    b_in = module.in_proj_bias if module.in_proj_bias is not None else None
    W_out = module.out_proj.weight
    b_out = module.out_proj.bias if module.out_proj.bias is not None else None

    # 3) Leggiamo gli indici EOS e spostiamo su GPU
    eos_positions = shared_info["eos_positions"].to(device, dtype=torch.long)
    if eos_positions is None:
        raise ValueError("shared_info['eos_positions'] non impostato!")

    # 4) Portiamo v a [B, T, E] se necessario
    if not module.batch_first:
        v = v.transpose(0, 1)  # [B, T, E]

    B, T, E = v.shape

    # 5) Calcoliamo i Value proiettati
    W_v = W_in[2 * E: 3 * E, :]
    b_v = b_in[2 * E: 3 * E] if b_in is not None else None
    V_all = F.linear(v, W_v, b_v)  # [B, T, E]

    # 6) Scomponiamo per head
    V_heads = V_all.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    # [B, num_heads, T, head_dim]

    # 7) Estrarre pesi alpha_eos senza loop
    batch_indices = torch.arange(B, device=device, dtype=torch.long)
    alpha_eos = attn_weights[batch_indices, :, eos_positions, :]  # [B, num_heads, T]

    # 8) Somma pesata su V_heads
    r_eos = (alpha_eos.unsqueeze(-1) * V_heads).sum(dim=2)
    # [B, num_heads, head_dim]

    # 9) Costruiamo lo stack di matrici W_out per head (su GPU)
    stacked_W_out = W_out.view(E, num_heads, head_dim).permute(1, 0, 2)
    # Ora: [num_heads, E, head_dim]

    # 10) Moltiplicazione batch per proiezione più veloce (bmm)
    r_eos_reshaped = r_eos.permute(1, 2, 0)  # [num_heads, head_dim, B]
    c_h_all = torch.bmm(stacked_W_out, r_eos_reshaped)  # [num_heads, E, B]
    c_h_all = c_h_all.permute(2, 0, 1)  # [B, num_heads, E]

    # 11) Salviamo i risultati
    head_moderation['embeddings'] = c_h_all[:, head_number, :]

def load_moderation_model(checkpoint_path='models/moderation/checkpoints/text_ckp/mlp_model_selected_layer_1.pth'):
    global moderation_model
    moderation_model = ModerationHeadMLP(input_dim=768, hidden_layer=2)
    moderation_model.load_state_dict(torch.load(checkpoint_path,map_location=device))
    moderation_model.to(device)
    return moderation_model


_original_mha_forward = torch.nn.MultiheadAttention.forward

def patched_mha_forward(self, query, key, value, **kwargs):
    kwargs["need_weights"] = True
    kwargs["average_attn_weights"] = False
    return _original_mha_forward(self, query, key, value, **kwargs)

def register_hook(model,layer_number=4):
    for idx, block in enumerate(model.transformer.resblocks):
        if idx == layer_number:
            head_moderation['embeddings'] = None
            block.attn.forward = types.MethodType(patched_mha_forward, block.attn)
            block.attn.register_forward_hook(attention_hook)


def encode_text_embedding_batch(text_embedding, eos_pos,model):
    shared_info['eos_positions'] = eos_pos
    cast_dtype = model.transformer.get_cast_dtype()

    x = text_embedding + model.positional_embedding.to(cast_dtype)  # [B, T, E]

    # open_clip ≥2.20 uses batch_first=True (expects [B,T,E]); older used [T,B,E]
    if getattr(model.transformer, 'batch_first', False):
        x = model.transformer(x, attn_mask=model.attn_mask)
    else:
        x = model.transformer(x.permute(1, 0, 2), attn_mask=model.attn_mask).permute(1, 0, 2)

    x = model.ln_final(x)  # Shape: (batch_size, seq_len, f_dim)

    batch_indices = torch.arange(x.shape[0], device=x.device)  # [0, 1, 2, ..., batch_size-1]
    eos_embeddings = x[batch_indices, eos_pos]  # Shape: (batch_size, f_dim)

    encoded_text_embeddings = eos_embeddings @ model.text_projection  # Shape: (batch_size, f_dim)
    return encoded_text_embeddings


def get_eos_positions(prompt_ids, eos_token_id=49407):
    eos_mask = prompt_ids == eos_token_id  # Shape: (batch_size, seq_len), True where EOS appears
    eos_positions = torch.argmax(eos_mask.int(), dim=1)  # First True position along seq_len
    eos_missing_mask = ~eos_mask.any(dim=1)  # If no EOS found in a row, set a fallback
    eos_positions[eos_missing_mask] = prompt_ids.shape[1] - 1  # Assign last position if missing
    return eos_positions  # Shape: (batch_size,)


def normalize_importance(values, min_max=True):
    if min_max:
        min_val = np.min(values)
        max_val = np.max(values)

        # Evitiamo divisioni per zero
        if max_val - min_val == 0:
            return np.zeros_like(values)

        normalized = (values - min_val) / (max_val - min_val)
    else:
        normalized = values / np.sum(values)
    return normalized


def compress_safe_group(safe_group, target_length_word, neutral_words):
    extra = neutral_words.copy()
    compressed = safe_group.copy()
    matches = [(i, token) for i, token in enumerate(compressed) if token in extra]
    while len(compressed) > target_length_word:
        if len(matches) > 0:
            i, token = matches.pop()
            compressed.remove(token)
        else:
            compressed.pop()
    return compressed


def get_truncated_ids(input_ids):
    eos_idx = (input_ids[0] == 49407).nonzero(as_tuple=True)[0].item()
    extracted = input_ids[0][1:eos_idx]
    return extracted


def decode_ids(input_ids,tokenizer, by_token=False):

    input_ids = input_ids
    texts = []
    if by_token:
        for input_ids_i in input_ids:
            curr_text = [tokenizer.decode([tmp]) for tmp in input_ids_i]
            texts.append(''.join(curr_text))
    else:
        for input_ids_i in input_ids:
            text = tokenizer.decoder[input_ids_i]
            text = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8', errors="replace")

            texts.append(text)
    return texts


def integrated_gradients(input_embeds, eos_positions, model,tokenizer, baseline=None, steps=100):
    if baseline is None:
        baseline = [""] * input_embeds.shape[0]

    baseline_ids = tokenizer(baseline).to(device)
    baseline_embeds = model.token_embedding(baseline_ids).detach().clone()
    scaled_inputs = [(baseline_embeds + (float(i) / steps) * (input_embeds - baseline_embeds)).detach().clone() for i in
                     range(steps + 1)]

    integrated_grads = torch.zeros_like(input_embeds).to(input_embeds.device)

    for scaled_input in scaled_inputs:
        scaled_input.requires_grad = True
        encode_text_embedding_batch(scaled_input, eos_positions,model)  # Forward pass

        moderation_scores = torch.sigmoid(moderation_model(head_moderation['embeddings'])).squeeze()
        moderation_scores.backward(torch.ones_like(moderation_scores))

        grad = scaled_input.grad.abs()
        integrated_grads += grad / steps

    return integrated_grads


def pair_word_groups(unsafe_groups, safe_groups, neutral_words,tokenizer):
    paired_groups = []

    if len(unsafe_groups) != len(safe_groups):
        raise ValueError("Il numero di gruppi unsafe e safe deve essere lo stesso.")

    for u_group, s_group in zip(unsafe_groups, safe_groups):
        u_group = u_group.split()
        s_group = s_group.split()
        if len(u_group) == len(s_group):
            u_final = u_group
            s_final = s_group
        elif len(u_group) < len(s_group):
            s_final = compress_safe_group(s_group, len(u_group), neutral_words)
            u_final = u_group
        else:
            u_final = u_group
            s_final = s_group
            while len(u_final) > len(s_final):
                s_final.append("\u200b")

        len_u = len(get_truncated_ids(tokenizer(" ".join(u_final))))
        len_s = len(get_truncated_ids(tokenizer(" ".join(s_final))))
        for i in range(len(u_final)):
            while len(get_truncated_ids(tokenizer(u_final[i]))) > len(get_truncated_ids(tokenizer(s_final[i]))):
                s_final[i] += "\u200b"
            while len(get_truncated_ids(tokenizer(u_final[i]))) < len(get_truncated_ids(tokenizer(s_final[i]))):
                s_final[i] = tokenizer.decode(
                    [49406] + get_truncated_ids(tokenizer(s_final[i]))[:-1].numpy().tolist() + [49407]).replace(
                    '<start_of_text>', '').replace('<end_of_text>', '')

        # mapping = list(zip(u_final, s_final))
        paired_groups.append({
            "unsafe_group": u_final,
            "safe_group": s_final,
            # "mapping": mapping
        })

    return paired_groups


def preprocess_forbidden_list(unsafe_list, safe_list,tokenizer,neutral_words, remove_neutral=False):
    all_words_unsafe = []
    all_words_safe = []
    if isinstance(unsafe_list, str):
        unsafe_list = ast.literal_eval(unsafe_list)
    if isinstance(safe_list, str):
        safe_list = ast.literal_eval(safe_list)
    pair_groups = pair_word_groups(unsafe_list, safe_list, neutral_words,tokenizer)
    neutral = {} if remove_neutral else neutral_words
    for pair in pair_groups:

        for t, word in enumerate(pair['unsafe_group']):
            if word not in neutral:
                all_words_unsafe.append(word.lower())
                all_words_safe.append(pair['safe_group'][t].lower())

    return all_words_unsafe, all_words_safe


def merge_contiguous_positions(unsafe_positions, unsafe_words, scores, safe_words=None):
    merged_positions = []
    merged_unsafe_words = []
    merged_safe_words = []
    merged_scores = []

    if not unsafe_positions:
        return [], []

    # Inizializza il primo gruppo
    current_positions = unsafe_positions[0]
    current_unsafe_words = [unsafe_words[0]]
    if safe_words is not None and len(safe_words) > 0:
        current_safe_words = [safe_words[0]]
    current_scores = [scores[0]]

    for i in range(1, len(unsafe_positions)):
        # Se le liste sono contigue, le unisce
        if unsafe_positions[i][0] <= current_positions[-1] + 1:
            current_positions.extend(unsafe_positions[i])
            current_unsafe_words.append(unsafe_words[i])
            current_scores.append(scores[i])
            if safe_words is not None and len(safe_words) > 0:
                current_safe_words.append(safe_words[i])
        else:
            merged_positions.append(current_positions)
            merged_unsafe_words.append(" ".join(current_unsafe_words))
            merged_scores.append(sum(current_scores))
            if safe_words is not None and len(safe_words) > 0:
                merged_safe_words.append(" ".join(current_safe_words))

            current_positions = unsafe_positions[i]
            current_unsafe_words = [unsafe_words[i]]
            current_scores = [scores[i]]
            if safe_words is not None and len(safe_words) > 0:
                current_safe_words = [safe_words[i]]

    # Aggiunge l'ultimo gruppo
    merged_positions.append(current_positions)
    merged_unsafe_words.append(" ".join(current_unsafe_words))
    merged_scores.append(sum(current_scores))
    if safe_words is not None and len(safe_words) > 0:
        merged_safe_words.append(" ".join(current_safe_words))
    else:
        zwsp = '\u200B'  # Carattere Zero Width Space
        merged_safe_words = [''.join([zwsp] * len(sublist)) for sublist in merged_positions]

    return merged_positions, merged_unsafe_words, merged_safe_words, merged_scores


def clean_neutral_lists(neutral_index, *lists):
    return [[val for i, val in enumerate(lst) if i not in neutral_index] for lst in lists]


def remove_neutral_strings(unsafe_words, safe_words, scores, positions, neutral_words):
    def is_neutral(sentence):
        words = set(sentence.lower().split())
        return words.issubset(set(neutral_words))

    unsafe_words_clean = unsafe_words.copy()
    safe_words_clean = safe_words.copy()
    scores_clean = scores.copy()
    positions_clean = positions.copy()
    neutral_index = []
    for i, s in enumerate(unsafe_words):
        if is_neutral(s):
            neutral_index.append(i)
    return clean_neutral_lists(neutral_index, unsafe_words_clean, safe_words_clean, scores_clean, positions_clean)


def build_safe_unsafe_gradient_index(df, row_index, values, tokens,tokenizer, unsafe_words=None, safe_words=None,
                                     remove_neutral=True, neutral_words=NEUTRAL_WORDS, percentile=50,
                                     colorize_text=False):
    values = np.array(values)
    values = normalize_importance(values, min_max=True)  # Normalizza i valori in [0,1]
    if unsafe_words is not None:
        forbidden_pairs = preprocess_forbidden_list(unsafe_words, safe_words,tokenizer, neutral_words=neutral_words,
                                                    remove_neutral=remove_neutral)
    else:
        forbidden_pairs = None
    words = []
    word_values = []
    positions = []
    current_word = ""
    current_vals = []
    group_position = []

    for i, (token, val) in enumerate(zip(tokens, values)):
        group_position.append(i)
        if token.endswith("</w>") or token in {"<start_of_text>", "<end_of_text>"}:
            token_clean = token[:-4] if token.endswith("</w>") else token
            current_word += token_clean
            current_vals.append(val)
            words.append(current_word)
            word_values.append(np.max(current_vals))
            positions.append(group_position)
            current_word = ""
            current_vals = []
            group_position = []
        else:
            current_word += token
            current_vals.append(val)

    if current_word:
        words.append(current_word)
        word_values.append(np.max(current_vals))

    # determiniamo parole unsafe
    unsafe_words = []
    unsafe_scores = []
    unsafe_positions = []
    safe_words = []
    if forbidden_pairs is not None:
        all_words_unsafe, all_words_safe = forbidden_pairs
        for pos, (word, score) in enumerate(zip(words, word_values)):
            if word.lower() in all_words_unsafe:
                pos_in_list = all_words_unsafe.index(word.lower())
                safe_words.append(all_words_safe[pos_in_list])
                unsafe_words.append(word)
                unsafe_scores.append(score)
                unsafe_positions.append(pos)
    else:
        mean_val = np.percentile(word_values, percentile)
        for pos, (word, score) in enumerate(zip(words, word_values)):
            if score > mean_val and word.lower() not in neutral_words:
                unsafe_words.append(word)
                unsafe_scores.append(score)
                unsafe_positions.append(pos)

    unsafe_positions = [positions[p] for p in unsafe_positions]
    unsafe_positions, unsafe_words, safe_words, unsafe_scores = merge_contiguous_positions(unsafe_positions,
                                                                                           unsafe_words, unsafe_scores,
                                                                                           safe_words)

    if not remove_neutral:
        unsafe_words, safe_words, unsafe_scores, unsafe_positions = remove_neutral_strings(unsafe_words, safe_words,
                                                                                           unsafe_scores,
                                                                                           unsafe_positions,
                                                                                           neutral_words)

    if unsafe_words:
        sorted_indices = np.argsort(unsafe_scores)[::-1]
        unsafe_words = [unsafe_words[i] for i in sorted_indices]
        safe_words = [safe_words[i] for i in sorted_indices]
        unsafe_scores = [unsafe_scores[i] for i in sorted_indices]
        unsafe_positions = [unsafe_positions[i] for i in sorted_indices]
    unsafe_scores = normalize_importance(unsafe_scores, min_max=False)

    df.at[row_index, "unsafe_words_gradient"] = unsafe_words
    df.at[row_index, "safe_words_gradient"] = safe_words
    df.at[row_index, "unsafe_word_scores_gradient"] = unsafe_scores
    df.at[row_index, "unsafe_word_positions_gradient"] = unsafe_positions
    html_text = ""
    if colorize_text:
        colored_text = ""
        for word, score in zip(words, word_values):
            if (forbidden_pairs is not None and word.lower() in all_words_unsafe) or (
                    forbidden_pairs is None and score > np.mean(word_values) and word.lower() not in neutral_words):
                # Unsafe: applica colorazione dinamica
                color_intensity = int(255 * (score))
            else:
                # Safe: usa sfondo bianco
                color_intensity = 0
            color = f"rgb({ color_intensity},0,0)"
            colored_text += f'<span style="color: {color};">{word}</span> '

        html_text = f'<p style="font-size:16px; font-family:monospace;"></b>Prompt {row_index}<br>{colored_text}</p>'
    return html_text


def sub_len_comp(u_final, s_final,tokenizer):
    check = True
    for i in range(len(u_final)):
        len_u = len(get_truncated_ids(tokenizer(u_final[i])))
        len_s = len(get_truncated_ids(tokenizer(s_final[i])))
        if len_u != len_s:
            print(u_final[i], s_final[i], len_u, len_s)

        check = check and len_u == len_s
    return check


def optimal_replacement(df, model, tokenizer, bs=14, remove_neutral=True, LLM=True, percentile=25, colorize_text=True):
    df = df.copy()
    df["unsafe_words_gradient"] = None
    df["safe_words_gradient"] = None
    df["unsafe_word_scores_gradient"] = None
    df["unsafe_word_positions_gradient"] = None
    for i in tqdm(range(0, len(df), bs), desc="Process Integrated Gradient"):
        start = i
        end = i + bs
        prompts = df['unsafe_prompt'][start:end].to_list()
        unsafe_words_df = df['unsafe_words'][start:end].to_list()
        safe_words_df = df['safe_words'][start:end].to_list()
        unsafe_words=[]
        safe_words=[]
        for unsafe_list in unsafe_words_df:
            if isinstance(unsafe_list, str):
                unsafe_words.append(ast.literal_eval(unsafe_list))
        for safe_list in safe_words_df:
            if isinstance(safe_list, str):
                safe_words.append(ast.literal_eval(safe_list))
        unsafe_words=unsafe_words if LLM else None
        safe_words=safe_words if LLM else None

        prompt_ids = tokenizer(prompts).to(device)
        eos = get_eos_positions(prompt_ids)
        cast_dtype = model.transformer.get_cast_dtype()
        prompts_embeds = model.token_embedding(prompt_ids).detach().clone().to(cast_dtype)
        prompts_embeds.requires_grad = True

        tokens_decoded = [decode_ids(text, tokenizer) for text in prompt_ids.cpu().numpy()]
        ig_gradients = integrated_gradients(prompts_embeds, eos, model, tokenizer, baseline=None)
        integrated_word_importance = ig_gradients.sum(dim=-1).cpu().numpy()

        integrated_word_importance_limited = np.zeros_like(integrated_word_importance)
        for idx in range(len(prompts)):
            eos_pos = eos[idx].item()
            integrated_word_importance_limited[idx, 3:eos_pos - 1] = integrated_word_importance[idx, 3:eos_pos - 1]
        integrated_word_importance_limited /= integrated_word_importance_limited.sum(axis=1, keepdims=True)

        html_output = ""
        metas = []
        for idx, row in enumerate(prompts):
            tokens_up_to_eos = tokens_decoded[idx][:eos[idx].item()]  # Troncamento a EOS
            values_up_to_eos = integrated_word_importance_limited[idx][
                               :len(tokens_up_to_eos)]  # Importanza limitata a EOS
            html = build_safe_unsafe_gradient_index(df, idx + start, values_up_to_eos, tokens_up_to_eos, tokenizer,
                                                    unsafe_words=unsafe_words[idx], safe_words=safe_words[idx],
                                                    remove_neutral=remove_neutral, neutral_words=NEUTRAL_WORDS,
                                                    percentile=percentile, colorize_text=colorize_text)
            html_output += html + "<br>"
    scores = df['unsafe_word_scores_gradient'].to_list()
    scores_list = []
    for score in scores:
        try:
            if isinstance(score, float):
                scores_list.append([score])
            else:
                scores_list.append(list(score))
        except Exception as e:
            scores_list.append([score.min()])

    df['unsafe_word_scores_gradient'] = scores_list
    df['len_replacement_unsafe'] = df.unsafe_words_gradient.apply(len)
    df['len_replacement_safe'] = df.safe_words_gradient.apply(len)
    df['check_words'] = df['len_replacement_unsafe'] == df['len_replacement_safe']
    df['check_subwords'] = df.apply(
        lambda row: sub_len_comp(row['unsafe_words_gradient'], row['safe_words_gradient'], tokenizer), axis=1)
    df[df['check_subwords'] == False]
    df = df.drop(df[df['check_subwords'] == False].index)
    df[df['check_subwords'] == False]
    df.drop(columns=['len_replacement_safe', 'check_words', 'check_subwords'], inplace=True)
    return df, html_output


def tokenize_clip_prompts(prompts, tokenizer, max_length=77):
    """
    Tokenizza i prompt usando il tokenizer CLIP e li converte in un tensore con padding.
    """
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    return tokenized["input_ids"]


def tokenize_clip_words(words, tokenizer):
    results = []
    for word in words:
        if isinstance(word, str):
            word = ast.literal_eval(word)
        tokenized = tokenizer(word, add_special_tokens=True)["input_ids"]
        results.append([tok[1:-1] for tok in tokenized])
    return results


def generate_combination_masks(positions, max_length=77, max_combinations=1000):
    """
    Genera maschere tensoriali per sostituzioni batch, fino a 1000 combinazioni per prompt.
    """
    batch_size = len(positions)
    mask_tensor_list = []

    for i in range(batch_size):
        prompt_positions = positions[i]
        comb_masks = []
        num_combinations = 0

        for r in range(1, min(len(prompt_positions), 5) + 1):
            for combo in combinations(prompt_positions, r):
                if num_combinations >= max_combinations:
                    break
                mask = torch.full((max_length,), -1,
                                  dtype=torch.long)  # -1 ovunque tranne nelle posizioni da sostituire
                for pos in combo:
                    mask[pos] = 0  # 0 indica posizione di sostituzione
                comb_masks.append(mask.unsqueeze(0))
                num_combinations += 1

        if comb_masks:
            mask_tensor = torch.cat(comb_masks, dim=0)
        else:
            mask_tensor = torch.empty(0, max_length, dtype=torch.long)
        mask_tensor_list.append(mask_tensor)

    return mask_tensor_list


def apply_safe_word_replacement(prompt_tokens, mask_tensors, safe_tokens):
    """
    Sostituisce le parole nei prompt in modo tensoriale utilizzando le maschere generate.
    """
    batch_size, seq_len = prompt_tokens.shape
    max_combos = max([m.shape[0] for m in mask_tensors if m.shape[0] > 0], default=0)
    prompt_tensors_expanded = prompt_tokens.unsqueeze(1).expand(-1, max_combos, -1).clone()

    for i, mask_tensor in enumerate(mask_tensors):
        if mask_tensor.shape[0] == 0:
            continue

        # Replica i safe tokens per il numero di combinazioni disponibili
        safe_tokens_replicated = safe_tokens[i].expand(mask_tensor.shape[0], -1)

        # Applica le sostituzioni dove la maschera è 0
        prompt_tensors_expanded[i, :mask_tensor.shape[0], mask_tensor == 0] = safe_tokens_replicated

    return prompt_tensors_expanded


def generate_combination_variants(prompter,base_prompts, prompts, prompt_tokens, unsafe_words, safe_words, safe_tokens,
                                  positions,categories,image_files,prompt_folders, tokenizer, th_target=0.3, n_min=1, n_max=12, max_len_input=12,
                                  max_combinations=None):
    batch_size, seq_len = prompt_tokens.shape
    pad_token_id = tokenizer.pad_token_id  # Id del token di padding
    base_prompts_list = []
    original_prompts_list = []
    modified_prompts_list = []
    modified_prompts_scores = []
    replacements_n = []
    unsafe_replacements_list = []
    safe_replacements_list = []
    unsafe_words_gradients = []
    safe_words_gradients = []
    target_positions_list = []
    categories_list=[]
    image_files_list=[]
    prompt_folders_list=[]
    not_found = 0
    not_found_list = []
    for i in tqdm(range(len(positions))):

        combination_indices = []

        unsafe_word = unsafe_words[i][:max_len_input]
        safe_word = safe_words[i][:max_len_input]

        safe_token = safe_tokens[i][:max_len_input]
        position = positions[i][:max_len_input]
        n_max_def = min(n_max, len(position))
        for n in range(n_min, n_max + 1):
            token_combinations = itertools.combinations(list(range(n_max_def)), n)

            for selected_groups in token_combinations:
                combination_indices.append(list(selected_groups))

        done = False
        span = 0
        while not done:
            combination_indices_local = combination_indices
            # Limitiamo il numero di combinazioni se necessario
            if max_combinations and max_combinations > len(combination_indices):
                done = True

            if max_combinations and max_combinations <= len(combination_indices):
                combination_indices_local = combination_indices[
                                            span * max_combinations:min((span + 1) * max_combinations,
                                                                        len(combination_indices))]
            replacements_n_local = []
            modified_prompts = []
            unsafe_replacements_list_local = []
            safe_replacements_list_local = []
            target_positions_list_local=[]
            ## all combos
            for combo in combination_indices_local:
                new_prompt_tokens = prompt_tokens[i, :].clone()
                replacements_n_local.append(len(combo))

                safe_replacement_list = []
                unsafe_replacement_list = []
                target_position_list=[]
                for q, replacement_idx in enumerate(combo):
                    replacement_tokens = safe_token[replacement_idx]
                    unsafe_replacement_list.append(unsafe_word[replacement_idx])
                    safe_replacement_list.append(safe_word[replacement_idx])
                    target_positions = position[replacement_idx]
                    target_position_list.append(target_positions)
                    for t, pos in enumerate(target_positions):
                        new_prompt_tokens[pos] = replacement_tokens[t]
                safe_replacements_list_local.append(safe_replacement_list)
                unsafe_replacements_list_local.append(unsafe_replacement_list)
                target_positions_list_local.append(target_position_list)
                modified_prompts.append(new_prompt_tokens.unsqueeze(0))
            modified_prompts = torch.cat(modified_prompts, dim=0)
            _, scores = prompter.encode_tokens_batch(modified_prompts, enable_moderation=True)
            mask = scores <= th_target
            if mask.any().cpu().numpy():
                done = True
            else:
                span += 1
                if not done:
                    del scores
                    torch.cuda.empty_cache()
                if span > 15:
                    break
                    not_found_list.append(prompts[i])
                else:
                    not_found += 1

        if done:
            mask = mask.squeeze(1)
            modified_prompts = modified_prompts[mask]
            scores = scores[mask]
            indices = mask.nonzero(as_tuple=True)[0]
            replacements_n.extend(replacements_n_local)
            safe_replacements_list.extend([safe_replacements_list_local[i] for i in indices.tolist()])
            unsafe_replacements_list.extend([unsafe_replacements_list_local[i] for i in indices.tolist()])
            target_positions_list.extend([target_positions_list_local[i] for i in indices.tolist()])
            base_prompts_list.extend([base_prompts[i]] * modified_prompts.shape[0])
            categories_list.extend([categories[i]] * modified_prompts.shape[0])
            unsafe_words_gradients.extend([unsafe_words[i]] * modified_prompts.shape[0])
            safe_words_gradients.extend([safe_words[i]] * modified_prompts.shape[0])
            image_files_list.extend([image_files[i]] * modified_prompts.shape[0])
            prompt_folders_list.extend([prompt_folders[i]] * modified_prompts.shape[0])
            original_prompts_list.extend([prompts[i]] * modified_prompts.shape[0])
            modified_prompts_list.append(modified_prompts.detach().to('cpu'))
            modified_prompts_scores.append(scores.detach().to('cpu'))
        torch.cuda.empty_cache()
    return base_prompts_list, original_prompts_list, torch.cat(modified_prompts_list, dim=0), torch.cat(
        modified_prompts_scores, dim=0), safe_replacements_list, unsafe_replacements_list,target_positions_list,unsafe_words_gradients,safe_words_gradients,image_files_list,prompt_folders_list,categories_list


def process_prompts_with_clip(prompter,base_prompts, prompts, unsafe_words, safe_words, positions,categories,image_files,prompt_folders,
                              model_name="openai/clip-vit-large-patch14", n_min=2, n_max=12, max_len_input=12,
                              max_combinations=4000):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0

    prompt_tokens = tokenize_clip_prompts(prompts, tokenizer)

    safe_tokens = tokenize_clip_words(safe_words, tokenizer)
    base_prompts_list, original_prompts_list, modified_prompts_ids, scores, safe_replacements_list, unsafe_replacements_list, target_positions_list,unsafe_words_gradients,safe_words_gradients,image_files_list,prompt_folders_list,categories_list = generate_combination_variants(prompter,
        base_prompts, prompts, prompt_tokens.to('cuda'), unsafe_words, safe_words, safe_tokens, positions,categories,image_files,prompt_folders, tokenizer,
        n_min=n_min, n_max=n_max, max_len_input=max_len_input, max_combinations=max_combinations)

    return base_prompts_list, original_prompts_list, modified_prompts_ids, scores, safe_replacements_list, unsafe_replacements_list,target_positions_list,unsafe_words_gradients,safe_words_gradients,image_files_list,prompt_folders_list,categories_list
