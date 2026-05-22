"""
Phase 3 — PGD Condition C: Random Initialization
--------------------------------------------------
Starting point: unsafe_prompt + random token embeddings at free positions
Token positions: same as Condition A
random_init: True

Reads : data/df_C_pgd_input.parquet
Writes: results/3_pgd_C/results_C_random_init.parquet
        (skips if output already exists)
"""
import os
import ast
import json
import time
import torch
import pandas as pd
from PIL import Image as PILImage

import config
from src.moderated_prompter import ModeratedPrompter, top_n_per_group
from src.optimal_selection import load_moderation_model, register_hook
import open_clip

OUT_PATH = os.path.join(config.RES_PGD_C, 'results_C_random_init.parquet')

if os.path.exists(OUT_PATH):
    print(f'Output already exists: {OUT_PATH}')
    print('Delete it to re-run. Exiting.')
    raise SystemExit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

with open(config.CATEGORY_CFG, 'r', encoding='utf-8') as f:
    category_block_dict = json.load(f)
CATEGORIES = list(category_block_dict.keys())

clip_model, _, _ = open_clip.create_model_and_transforms(
    config.CLIP_MODEL_ID, pretrained=config.CLIP_PRETRAINED, device=device
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_ID)
register_hook(clip_model, layer_number=4)

prompter = ModeratedPrompter(api_key='', checkpoint_path=config.MODERATION_CKPT, device=device)

df_C = pd.read_parquet(os.path.join(config.DATA_DIR, 'df_C_pgd_input.parquet'))
print(f'Loaded df_C: {len(df_C)} rows')


def run_pgd(df_input, random_init=True):
    final_df = []
    for category in CATEGORIES:
        print(f'\n[Condition C] Category: {category}')
        df_cat = df_input[df_input['category'] == category].copy().reset_index(drop=True)
        df_cat = top_n_per_group(df_cat, n=config.TOP_N)
        all_unsafe_words = list(set().union(*df_cat['unsafe_words_gradient']))

        forbidden_list = list(category_block_dict[category]) + all_unsafe_words
        moderation_weight = torch.tensor(
            [config.GLOBAL_MOD] * len(df_cat), dtype=torch.float32
        ).to(device)

        target_prompts, selected_indices, target_images = [], [], []
        base_prompts, unsafe_prompts, revised_prompts = [], [], []
        unsafe_rep_list, safe_rep_list = [], []
        unsafe_words_list, safe_words_list, rev_scores = [], [], []

        for _, row in df_cat.iterrows():
            target_prompts.append(f"{row['revised_prompt']} Photorealistic")
            base_prompts.append(row['base_prompt'])
            unsafe_prompts.append(row['unsafe_prompt'])
            revised_prompts.append(row['revised_prompt'])
            unsafe_rep_list.append(row['unsafe_words_gradient'])
            safe_rep_list.append(row['safe_words_gradient'])
            unsafe_words_list.append(row['unsafe_words_gradient'])
            safe_words_list.append(row['safe_words_gradient'])
            rev_scores.append(row['harm'])

            img_path = os.path.join(
                config.REFERENCE_IMG_DIR,
                row['category'], row['prompt_folder'], f"{row['image_file']}.png"
            )
            target_images.append(PILImage.open(img_path))

            fp = row['flatten_positions']
            selected_indices.append(ast.literal_eval(fp) if isinstance(fp, str) else fp)

        all_results = prompter.text_inversion_base_batched(
            target_prompts, selected_indices,
            target_images=target_images,
            target_prompt=[None] * len(target_prompts),
            n_iterations=config.N_ITERATIONS,
            forbidden_list=forbidden_list,
            random_init=random_init,
            focus=None,
            moderation_focus=None,
            moderation_weight=moderation_weight,
            tau_prime=config.TAU_PRIME,
        )

        for i, prompt in enumerate(target_prompts):
            res = pd.DataFrame(all_results[prompt]).drop_duplicates(subset=['all_list'])
            if res['all_moderation_scores'].min() < 0.5:
                res = res[res['all_moderation_scores'] <= 0.5].sort_values(
                    by='all_clip_scores', ascending=False)
                local_flagged = False
            else:
                res = res.sort_values(by='all_clip_scores', ascending=False)
                local_flagged = True

            first_n = res.head(config.N_S).values
            sub_df = pd.DataFrame({
                'base_prompt':           [base_prompts[i]]    * len(first_n),
                'unsafe_prompt':         [unsafe_prompts[i]]  * len(first_n),
                'revised_prompt':        [revised_prompts[i]] * len(first_n),
                'revised_prompts_score': [rev_scores[i]]      * len(first_n),
                'safe_replacements':     [safe_rep_list[i]]   * len(first_n),
                'unsafe_replacements':   [unsafe_rep_list[i]] * len(first_n),
                'unsafe_words':          [unsafe_words_list[i]]* len(first_n),
                'safe_words':            [safe_words_list[i]] * len(first_n),
                'adv_prompt':            [t[0] for t in first_n],
                'loss':                  [t[1] for t in first_n],
                'clip':                  [t[2] for t in first_n],
                'harm':                  [t[3] for t in first_n],
                'local_flagged':         [local_flagged]       * len(first_n),
                'category':              [category]            * len(first_n),
            })
            final_df.append(sub_df)

        torch.cuda.empty_cache()
        time.sleep(60)

    return pd.concat(final_df, ignore_index=True)


df_out = run_pgd(df_C, random_init=True)
df_out.to_parquet(OUT_PATH, index=False)
print(f'\nCondition C saved: {len(df_out)} rows -> {OUT_PATH}')
print('Phase 3 complete.')
