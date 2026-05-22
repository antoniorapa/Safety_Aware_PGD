"""
Phase 4 — Classifier Scoring (10 systems, all prompts)
-------------------------------------------------------
Scores ALL adversarial prompts from conditions A, B, C with 10 classifier-based
moderation systems (no LLMs). Fast (~1 hour for 1990 prompts).

Systems scored:
  1. MLP-STD        custom CLIP head (standard trained)
  2. MLP-ADV        custom CLIP head (adversarially fine-tuned)
  3. OpenAI         omni-moderation-latest
  4. DistilRoBERTa  michellejieli/NSFW_text_classifier
  5. DeBERTa        KoalaAI/Text-Moderation
  6. ToxicBERT      unitary/toxic-bert
  7. FB-HateSpeech  facebook/roberta-hate-speech-dynabench-r4-target
  8. ToxicityModel  nicholasKluge/ToxicityModel
  9. RoBERTa-Toxic  s-nlp/roberta_toxicity_classifier
 10. Offensive-Twitter cardiffnlp/twitter-roberta-base-offensive

Reads : data/results_{A,B,C}_*.parquet
Writes: results/4_classifier_scores/scores_{A,B,C}_classifiers.csv
        (skips individual condition if CSV already exists)
"""
import os
import gc
import time
import torch
import torch.nn.functional as _F
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from transformers import (
    pipeline as hf_pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import open_clip

import config
from src.moderation_clip import ModerationHeadMLP as _MHP
from src.optimal_selection import (
    load_moderation_model, register_hook, get_eos_positions,
    encode_text_embedding_batch, head_moderation, shared_info,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ── Load CLIP + MLP models ────────────────────────────────────────────────────
clip_model, _, _ = open_clip.create_model_and_transforms(
    config.CLIP_MODEL_ID, pretrained=config.CLIP_PRETRAINED, device=device
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_ID)
moderation_model = load_moderation_model(config.MODERATION_CKPT)
register_hook(clip_model, layer_number=4)
print('CLIP + MLP-STD loaded.')

mlp_adv = _MHP(input_dim=3072, hidden_layer=2)
mlp_adv.load_state_dict(torch.load(config.MODERATION_CKPT_ADV, map_location=device))
mlp_adv.to(device).eval()
print('MLP-ADV loaded.')

adv_head_cache = {}

def _make_adv_hooks(layer_idx, head_idx):
    captured = {}

    def _pre(module, args):
        captured['q'] = args[0].detach()
        captured['k'] = args[1].detach()
        captured['v'] = args[2].detach()

    def _post(module, inputs, output):
        if 'q' not in captured:
            return
        q = captured.pop('q').to(device)
        k = captured.pop('k').to(device)
        v = captured.pop('v').to(device)
        if not module.batch_first:
            q, k, v = q.transpose(0,1), k.transpose(0,1), v.transpose(0,1)
        E, nH = module.embed_dim, module.num_heads
        hd = E // nH
        W_in, b_in, W_out = module.in_proj_weight, module.in_proj_bias, module.out_proj.weight
        B, T, _ = q.shape
        Q = _F.linear(q, W_in[:E],      b_in[:E]      if b_in is not None else None)
        K = _F.linear(k, W_in[E:2*E],   b_in[E:2*E]   if b_in is not None else None)
        V = _F.linear(v, W_in[2*E:3*E], b_in[2*E:3*E] if b_in is not None else None)
        Q = Q.reshape(B, T, nH, hd).permute(0,2,1,3)
        K = K.reshape(B, T, nH, hd).permute(0,2,1,3)
        V = V.reshape(B, T, nH, hd).permute(0,2,1,3)
        scale  = hd ** -0.5
        scores = torch.matmul(Q, K.transpose(-2,-1)) * scale
        mask   = clip_model.attn_mask[:T,:T].to(q.device, dtype=q.dtype)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn_w = torch.softmax(scores, dim=-1)
        eos_pos = shared_info['eos_positions'].to(q.device, dtype=torch.long)
        bi      = torch.arange(B, device=q.device, dtype=torch.long)
        alpha   = attn_w[bi, :, eos_pos, :]
        r_eos   = (alpha.unsqueeze(-1) * V).sum(dim=2)
        stacked = W_out.view(E, nH, hd).permute(1, 0, 2)
        c_h     = torch.bmm(stacked, r_eos.permute(1, 2, 0)).permute(2, 0, 1)
        adv_head_cache[layer_idx] = c_h[:, head_idx, :].detach()

    return _pre, _post

for _li, _hi in config.ADV_HEAD_LAYERS:
    _block = clip_model.transformer.resblocks[_li]
    _pre, _post = _make_adv_hooks(_li, _hi)
    _block.attn.register_forward_pre_hook(_pre)
    _block.attn.register_forward_hook(_post)
print(f'ADV hooks registered on layers {[l for l,_ in config.ADV_HEAD_LAYERS]}.')

# ── OpenAI client ─────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None

# ── HF classifiers ────────────────────────────────────────────────────────────
def _load_pipe(model_id):
    return hf_pipeline(
        'text-classification', model=model_id,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True, max_length=512, top_k=None, token=config.HF_TOKEN,
    )

distilroberta_pipe  = _load_pipe('michellejieli/NSFW_text_classifier')
deberta_tokenizer   = AutoTokenizer.from_pretrained('KoalaAI/Text-Moderation', token=config.HF_TOKEN)
deberta_model       = AutoModelForSequenceClassification.from_pretrained(
    'KoalaAI/Text-Moderation', token=config.HF_TOKEN).to(device).eval()
deberta_id2label    = deberta_model.config.id2label
deberta_ok_idx      = next(i for i, l in deberta_id2label.items() if l.upper() == 'OK')
toxic_bert_pipe     = _load_pipe('unitary/toxic-bert')
fb_hate_pipe        = _load_pipe('facebook/roberta-hate-speech-dynabench-r4-target')
toxicity_model_pipe = _load_pipe('nicholasKluge/ToxicityModel')
snlp_toxic_pipe     = _load_pipe('s-nlp/roberta_toxicity_classifier')
offensive_pipe      = _load_pipe('cardiffnlp/twitter-roberta-base-offensive')
print('All classifiers loaded.')

# ── Scoring functions ─────────────────────────────────────────────────────────
def score_mlp_std(text):
    ids  = tokenizer([text]).to(device)
    eos  = get_eos_positions(ids)
    cast = clip_model.transformer.get_cast_dtype()
    emb  = clip_model.token_embedding(ids).detach().clone().to(cast)
    with torch.no_grad():
        encode_text_embedding_batch(emb, eos, clip_model)
    score = torch.sigmoid(moderation_model(head_moderation['embeddings'])).squeeze().item()
    head_moderation['embeddings'] = None
    return round(score, 4)

def score_mlp_adv(text):
    ids  = tokenizer([text]).to(device)
    eos  = get_eos_positions(ids)
    cast = clip_model.transformer.get_cast_dtype()
    emb  = clip_model.token_embedding(ids).detach().clone().to(cast)
    with torch.no_grad():
        encode_text_embedding_batch(emb, eos, clip_model)
    adv_emb = torch.cat([adv_head_cache[l] for l, _ in config.ADV_HEAD_LAYERS], dim=-1)
    score = torch.sigmoid(mlp_adv(adv_emb)).squeeze().item()
    adv_head_cache.clear()
    head_moderation['embeddings'] = None
    return round(score, 4)

def score_openai(text):
    if not openai_client:
        return None
    resp = openai_client.moderations.create(model='omni-moderation-latest', input=text)
    return 1.0 if resp.results[0].flagged else 0.0

def _pipe_score(out, harmful_labels):
    scores = {r['label'].lower(): r['score'] for r in out}
    for lbl in harmful_labels:
        if lbl in scores:
            return round(scores[lbl], 4)
    return round(max(0.0, 1.0 - sum(v for k, v in scores.items() if k not in harmful_labels)), 4)

def score_distilroberta(text):  return _pipe_score(distilroberta_pipe(text)[0], ['nsfw'])
def score_deberta(text):
    inputs = deberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = deberta_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()
    return round(1.0 - probs[deberta_ok_idx].item(), 4)
def score_toxic_bert(text):     return _pipe_score(toxic_bert_pipe(text)[0], ['toxic'])
def score_fb_hate(text):        return _pipe_score(fb_hate_pipe(text)[0], ['hate'])
def score_toxicity(text):       return _pipe_score(toxicity_model_pipe(text)[0], ['toxic'])
def score_snlp(text):           return _pipe_score(snlp_toxic_pipe(text)[0], ['toxic'])
def score_offensive(text):      return _pipe_score(offensive_pipe(text)[0], ['off', 'offensive'])

SCORERS = {
    'MLP-STD':           score_mlp_std,
    'MLP-ADV':           score_mlp_adv,
    'OpenAI':            score_openai,
    'DistilRoBERTa':     score_distilroberta,
    'DeBERTa':           score_deberta,
    'ToxicBERT':         score_toxic_bert,
    'FB-HateSpeech':     score_fb_hate,
    'ToxicityModel':     score_toxicity,
    'RoBERTa-Toxic':     score_snlp,
    'Offensive-Twitter': score_offensive,
}

# ── Score helper ──────────────────────────────────────────────────────────────
def score_all_prompts(df_in, label, out_csv):
    if os.path.exists(out_csv):
        print(f'[{label}] Already scored: {out_csv}')
        return pd.read_csv(out_csv)

    records = []
    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc=f'Scoring {label}'):
        text = row['adv_prompt']
        rec = {
            'condition':   label,
            'category':    row.get('category', ''),
            'base_prompt': row['base_prompt'],
            'adv_prompt':  text,
            'clip':        row.get('clip', None),
            'harm':        row.get('harm', None),
        }
        for sys_name, scorer in SCORERS.items():
            wait = 5
            while True:
                try:
                    rec[sys_name] = scorer(text)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if any(k in msg for k in ['429', 'too many requests', 'rate limit']):
                        if wait > 60:
                            rec[sys_name] = None; break
                        time.sleep(wait); wait = min(wait * 2, 65)
                    else:
                        rec[sys_name] = None; break
        records.append(rec)

    out = pd.DataFrame(records)
    out.to_csv(out_csv, index=False)
    print(f'[{label}] Saved {len(out)} rows -> {out_csv}')
    return out

# ── Resolve parquet paths (prefer results/ output, fallback to data/) ─────────
parquet_A = config.PARQUET_A
parquet_B = config.OUT_PARQUET_B if os.path.exists(config.OUT_PARQUET_B) else config.PARQUET_B
parquet_C = config.OUT_PARQUET_C if os.path.exists(config.OUT_PARQUET_C) else config.PARQUET_C

df_adv_A = pd.read_parquet(parquet_A)
df_adv_B = pd.read_parquet(parquet_B)
df_adv_C = pd.read_parquet(parquet_C)
print(f'Rows — A: {len(df_adv_A)}, B: {len(df_adv_B)}, C: {len(df_adv_C)}')

scores_A = score_all_prompts(df_adv_A, 'A: Full Pipeline',
                             os.path.join(config.RES_CLASSIFIERS, 'scores_A_classifiers.csv'))
scores_B = score_all_prompts(df_adv_B, 'B: No SR',
                             os.path.join(config.RES_CLASSIFIERS, 'scores_B_classifiers.csv'))
scores_C = score_all_prompts(df_adv_C, 'C: Random Init',
                             os.path.join(config.RES_CLASSIFIERS, 'scores_C_classifiers.csv'))

print(f'\nPhase 4 complete. Total rows scored: {len(scores_A) + len(scores_B) + len(scores_C)}')
