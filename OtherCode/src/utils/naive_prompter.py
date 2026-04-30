import json
import random
import time
from typing import Mapping, Any
import numpy as np
from statistics import mean
import copy
import open_clip
import torch
from sentence_transformers.util import semantic_search, dot_score, normalize_embeddings
from src.utils.text_moderation import OpenAITextModeration

class NaivePrompter():
    def __init__(self, args, api_key, early_stopping=False, early_stopping_patience=200,epsilon=0.031, query_limit=10000):
        """
        Initialize the OptimizerPipeline with DistilRoberta and OpenAI moderation models.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.openai_moderation = OpenAITextModeration('open_ai',api_key=api_key)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain,device=self.device)
        self.tokenizer = open_clip.get_tokenizer(args.clip_model)
        self.args = args
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.epsilon = epsilon
        self.query_limit = query_limit
        self.api_count=0

    def optimize_prompt(self, target_images=None, target_prompts=None, fixed_prompt=None,forbidden_list=[]):
        """
        Main entry point for optimizing prompts. Retrieves target features and runs the optimization loop.
        """

          # Use the recommended method to get the tokenizer

        all_target_features = self.get_target_feature(
            self.model,
            self.preprocess,
            self.tokenizer,
            target_images,
            target_prompts
        )

        learned_prompt, top_10 = self.optimize_prompt_loop(
            self.model,
            self.tokenizer,
            self.model.token_embedding,
            all_target_features,
            self.args,
            fixed_prompt,
            forbidden_list
        )

        return learned_prompt,  top_10,

    def optimize_prompt_loop(self, model, tokenizer, token_embedding, all_target_features,
                             args, fixed_prompt=None,forbidden_list=[]):
        """
        Runs the optimization loop to find the best prompt embeddings.
        """
        opt_iters, lr, weight_decay, batch_size = args.iter, args.lr, args.weight_decay, args.batch_size
        fixed_prompt_ids = tokenizer.encode(fixed_prompt) if fixed_prompt else None


        # Initialize prompt embeddings
        prompt_embeds, dummy_embeds, dummy_ids = self.initialize_prompt(tokenizer,
                                                                        token_embedding,
                                                                        args,
                                                                        fixed_prompt_ids
                                                                        )

        p_bs, p_len, p_dim = prompt_embeds.shape
        input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

        best_sim, best_negative_sim = -1000 * args.loss_weight, 1000 * args.loss_weight
        best_text, best_negative_text = "", ""
        prompt_list, sim_list, negative_prompt_list, negative_sim_list = [], [], [], []

        # Early stopping variables
        steps_since_improvement = 0
        forbidden_list =forbidden_list

        forbidden_list_tokenized = [torch.tensor(self.tokenizer.encode(seq)) for seq in forbidden_list] if len(forbidden_list)>0 else []
        high_value = 1e9

        for step in range(opt_iters):

            # Randomly sample target features
            if batch_size is None:
                target_features = all_target_features
            else:
                curr_indx = torch.randperm(len(all_target_features))
                target_features = all_target_features[curr_indx][0:batch_size]

            universal_target_features = all_target_features

            # forward projection
            projected_embeds, nn_indices = self.nn_project(prompt_embeds, token_embedding, print_hits=False)

            # get cosine similarity score with other target features
            with torch.no_grad():
                padded_embeds = dummy_embeds.detach().clone()
                padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
                logits_per_image, _ = self.forward_text_embedding(model, padded_embeds, dummy_ids, universal_target_features)
                scores_per_prompt = logits_per_image.mean(dim=0)
                universal_cosim_score = scores_per_prompt.max().item()
                best_indx = scores_per_prompt.argmax().item()

            tmp_embeds = prompt_embeds.detach().clone()
            tmp_embeds.data = projected_embeds.data
            tmp_embeds.requires_grad = True

            # padding
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)

            logits_per_image, _ = self.forward_text_embedding(model, padded_embeds, dummy_ids, target_features)
            cosim_scores = logits_per_image
            loss = 1 - cosim_scores.mean()

            decoded_text = (self.decode_ids(nn_indices, tokenizer)[best_indx])

            if fixed_prompt:
                decoded_text += fixed_prompt

            # Handle forbidden words
            new_forbidden = self.openai_moderation.interpret_text(decoded_text)[0]['critical_words']

            if new_forbidden:
                forbidden_list.append(new_forbidden.strip())
                forbidden_list_tokenized.append(torch.tensor(tokenizer.encode(new_forbidden.strip())))

            prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])

            sensitive_words = self.check_forbidden_list(decoded_text, forbidden_list)
            flagged=False
            if len(forbidden_list)>0 and step==0:
                flagged=True
            if len(sensitive_words)>0:
                flagged=True
                positions=self.forbidden_positions(torch.tensor(tokenizer.encode(decoded_text)),forbidden_list_tokenized)
                for pos in positions:
                    if pos < prompt_embeds.grad.shape[1]:  # Ensure the sequence fits within bounds
                        prompt_embeds.grad[0, pos, :] = high_value

            input_optimizer.step()
            input_optimizer.zero_grad()

            # Logging
            #if step % 100 == 0 or step == opt_iters - 1:
                #print()
                #self.print_loss(step, lr, universal_cosim_score, decoded_text)

            # Update best prompts
            if best_sim < universal_cosim_score and not flagged:
                best_sim = universal_cosim_score
                best_text = decoded_text
                prompt_list.append(decoded_text)
                sim_list.append(universal_cosim_score)
                steps_since_improvement = 0
                #print('\nNEW BEST')
                #self.print_loss(step,lr,best_sim,best_text)
                #print('Forbbidden words:', forbidden_list)


            else:
                steps_since_improvement += 1

            if self.early_stopping and steps_since_improvement >= self.early_stopping_patience:
                print(f"Early stopping at step {step} with best similarities {best_sim}")
                break

        # Sort and return top prompts
        top_10 = sorted(zip(sim_list, prompt_list), reverse=True, key=lambda x: x[0])[:10]
        print(
            f"\nbest cosine sim: {best_sim}\nbest prompt: {best_text}")

        return best_text, top_10

    import torch

    def forbidden_positions(self,tokenized_text, token_sequences):

        positions_set = set()
        text_len = tokenized_text.size(0)

        for sequence in token_sequences:
            seq_len = sequence.size(0)

            # Use striding to create a sliding window view of the tokenized_text
            if seq_len < tokenized_text.size(0):

                windows = tokenized_text.unfold(0, seq_len, 1)

                # Check where windows match the sequence
                matches = (windows == sequence).all(dim=1).nonzero(as_tuple=True)[0]

                # For each match, record positions of each element in the sequence
                for match in matches:
                    match_positions = range(match.item(), match.item() + seq_len)
                    positions_set.update(match_positions)

        return positions_set


    def check_forbidden_list(self, text, forbbidden_list):
        sentence_words = set(text.split())
        found_words = set(forbbidden_list).intersection(sentence_words)
        found_words_list = list(found_words)
        return found_words_list

    def interpolate_prompts(self, prompt1, prompt2, lambda_param, tokenizer, token_embedding, model):
        """
        Interpolates between two prompts using a lambda parameter, projects the combination, and decodes it.
        """
        # Tokenize the prompts
        tokens1 = tokenizer.encode(prompt1, return_tensors='pt').to(self.device)
        tokens2 = tokenizer.encode(prompt2, return_tensors='pt').to(self.device)

        # Evaluate prompt embeddings
        embeds1 = token_embedding(tokens1).detach()
        embeds2 = token_embedding(tokens2).detach()

        # Interpolate embeddings
        interpolated_embeds = (1 - lambda_param) * embeds1 + lambda_param * embeds2

        # Project the interpolated embeddings
        projected_embeds, _ = self.nn_project(interpolated_embeds, token_embedding)

        # Decode the projected combination
        decoded_text = self.decode_ids(projected_embeds.argmax(dim=-1), tokenizer)

        return decoded_text

    # Helper methods

    def print_loss(self, step, lr, cosim_score, decoded_text):
        print(f"step: {step}, lr: {lr}, cosim similarity: {cosim_score:.3f}")
        print(f"text: {decoded_text}")


    @staticmethod
    def compute_cosim_loss(logits_per_image):
        """
        Computes the cosine similarity loss.
        """
        cosim_loss = -torch.log(logits_per_image.mean())
        return cosim_loss

    @staticmethod
    def nn_project(curr_embeds, embedding_layer, print_hits=False):
        """
        Projects the current embeddings into the token embedding space using nearest neighbor search.
        """
        with torch.no_grad():
            bsz, seq_len, emb_dim = curr_embeds.shape
            curr_embeds = curr_embeds.reshape((-1, emb_dim))
            curr_embeds = normalize_embeddings(curr_embeds)
            embedding_matrix = embedding_layer.weight
            embedding_matrix = normalize_embeddings(embedding_matrix)
            hits = semantic_search(curr_embeds, embedding_matrix, query_chunk_size=curr_embeds.shape[0], top_k=1,
                                   score_function=dot_score)
            if print_hits:
                all_hits = [hit[0]["score"] for hit in hits]
                print(f"mean hits: {mean(all_hits)}")
            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device).reshape(
                (bsz, seq_len))
            projected_embeds = embedding_layer(nn_indices)
        return projected_embeds, nn_indices


    def initialize_prompt(self,tokenizer, token_embedding, args, fixed_prompt_ids=None,prompt_ids=None):
        """
        Initializes the prompt embeddings with random tokens.
        """
        prompt_len = args.prompt_len
        if prompt_ids is None:
            prompt_ids = torch.randint(len(tokenizer.encoder), (args.prompt_bs, prompt_len)).to(self.device)
        prompt_embeds = token_embedding(prompt_ids).detach()
        prompt_embeds.requires_grad = True

        template_text = "{}"
        padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))

        dummy_ids = tokenizer.encode(padded_template_text)
        dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
        dummy_ids = ([49406] + dummy_ids)
        if fixed_prompt_ids is not None:
            dummy_ids += fixed_prompt_ids
        dummy_ids += [49407]
        dummy_ids += [0] * (77 - len(dummy_ids))
        dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(self.device)

        tmp_dummy_ids = copy.deepcopy(dummy_ids)
        tmp_dummy_ids[tmp_dummy_ids == -1] = 0
        dummy_embeds = token_embedding(tmp_dummy_ids).detach()
        dummy_embeds.requires_grad = False
        return prompt_embeds, dummy_embeds, dummy_ids

    @staticmethod
    def decode_ids(input_ids, tokenizer, by_token=False):
        """
        Decodes token IDs to text.
        """
        input_ids = input_ids.detach().cpu().numpy()
        texts = []
        if by_token:
            for input_ids_i in input_ids:
                curr_text = [tokenizer.decode([tmp]) for tmp in input_ids_i]
                texts.append('|'.join(curr_text))
        else:
            for input_ids_i in input_ids:
                texts.append(tokenizer.decode(input_ids_i))
        return texts

    @staticmethod
    def set_random_seed(seed=0):
        """
        Sets the random seed for reproducibility.
        """
        torch.manual_seed(seed + 0)
        torch.cuda.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 2)
        np.random.seed(seed + 3)
        random.seed(seed + 5)

    def get_target_feature(self, model, preprocess, tokenizer, target_images=None, target_prompts=None):
        """
        Retrieves target features from images or text prompts.
        """
        with torch.no_grad():
            if target_images is not None:
                curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
                curr_images = torch.cat(curr_images).to(self.device)
                all_target_features = model.encode_image(curr_images)
            else:
                texts = tokenizer(target_prompts).to(self.device)
                all_target_features = model.encode_text(texts)
        return all_target_features

    def measure_similarity(self,orig_images, images, ref_model, ref_clip_preprocess):
        """
        Measures the similarity between original and generated images using a reference model.
        """
        with torch.no_grad():
            ori_batch = torch.cat([ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]).to(self.device)
            gen_batch = torch.cat([ref_clip_preprocess(i).unsqueeze(0) for i in images]).to(self.device)
            ori_feat = ref_model.encode_image(ori_batch) / ref_model.encode_image(ori_batch).norm(dim=1, keepdim=True)
            gen_feat = ref_model.encode_image(gen_batch) / ref_model.encode_image(gen_batch).norm(dim=1, keepdim=True)
        return (ori_feat @ gen_feat.t()).mean().item()

    @staticmethod
    def encode_text_embedding(model, text_embedding, ids, avg_text=False):
        """
        Encodes text embeddings using the model.
        """
        cast_dtype = model.transformer.get_cast_dtype()
        x = text_embedding + model.positional_embedding.to(cast_dtype)
        x = model.transformer(x.permute(1, 0, 2), attn_mask=model.attn_mask).permute(1, 0, 2)
        x = model.ln_final(x)
        if avg_text:
            x = x.mean(dim=1) @ model.text_projection
        else:
            x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ model.text_projection
        return x

    @staticmethod
    def forward_text_embedding(model, embeddings, ids, image_features, avg_text=False, return_feature=False):
        """
        Performs forward propagation of text embeddings through the model.
        """
        text_features = NaivePrompter.encode_text_embedding(model, embeddings, ids, avg_text=avg_text)
        if return_feature:
            return text_features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()

    def get_embeddings(self, prompt):
        token_embedding=self.model.token_embedding
        prompt_ids = self.tokenizer(prompt)[:,1:self.args.prompt_len+1].to(self.device)
        prompt_embeds, _, _ = self.initialize_prompt(
            self.tokenizer,
            token_embedding,
            self.args,
            prompt_ids=prompt_ids
            )
        return  prompt_embeds

    def combine_embeddings(self, lambda_val,prompt_embeds,negative_prompt_embeds):
            return (1 - lambda_val) *prompt_embeds + lambda_val * negative_prompt_embeds

    def decode_embedding(self , embedding):
        projected_embeds, nn_indices = self.nn_project(embedding, self.model.token_embedding, print_hits=False)
        decoded_text = (self.decode_ids(nn_indices, self.tokenizer)[0])
        return decoded_text

    def embedding_to_text(self, embedding):
        # Convert embedding back to text
        return self.decode_embedding(embedding)



def read_json(filename: str, encoding='utf-8') -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename,encoding=encoding) as fp:
        return json.load(fp)



