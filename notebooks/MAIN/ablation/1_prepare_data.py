"""
Phase 1 — Data Preparation
--------------------------
- HuggingFace login
- Copy source files into data/
- Load adversarial_test.json, parse list columns
- Compute harm scores for unsafe_prompts (needed for Conditions B and C)
- Build and save df_B_input.parquet and df_C_input.parquet to data/
  (df_A is just the original adversarial_test; Condition A results already exist)

Output files (data/):
  adversarial_test.json
  dall_e_3_results_plain.parquet
  results_A_full_pipeline.parquet
  results_B_no_sr.parquet          (pre-computed PGD results, if available)
  results_C_random_init.parquet    (pre-computed PGD results, if available)
  df_B_pgd_input.parquet           (PGD input for Condition B)
  df_C_pgd_input.parquet           (PGD input for Condition C)
"""
import ast
import shutil
import torch
import pandas as pd
from huggingface_hub import login as _hf_login
import open_clip

import config
from src.optimal_selection import (
    load_moderation_model, register_hook, get_eos_positions,
    encode_text_embedding_batch, head_moderation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# ── HuggingFace login ─────────────────────────────────────────────────────────
_hf_login(token=config.HF_TOKEN, add_to_git_credential=False)
print('HuggingFace login OK.')

# ── Copy source files to data/ ────────────────────────────────────────────────
copies = [
    (config.SRC_ADV_TEST_JSON,  config.ADV_TEST_JSON),
    (config.SRC_DALL_E_PARQUET, config.DALL_E_PARQUET),
    (config.SRC_PARQUET_A,      config.PARQUET_A),
    (config.SRC_PARQUET_B,      config.PARQUET_B),
    (config.SRC_PARQUET_C,      config.PARQUET_C),
]
import os
for src, dst in copies:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f'Copied: {os.path.basename(dst)}')
    elif os.path.exists(dst):
        print(f'Already present: {os.path.basename(dst)}')
    else:
        print(f'WARNING: source not found: {src}')

# ── Load adversarial_test.json ────────────────────────────────────────────────
df = pd.read_json(config.ADV_TEST_JSON, orient='records')

for col in ['flatten_positions', 'positions', 'unsafe_replacements',
            'safe_replacements', 'unsafe_words_gradient', 'safe_words_gradient']:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(f'Loaded {len(df)} rows | {df["base_prompt"].nunique()} base_prompts'
      f' | categories: {sorted(df["category"].unique())}')

# ── Load CLIP + moderation model ──────────────────────────────────────────────
clip_model, _, _ = open_clip.create_model_and_transforms(
    config.CLIP_MODEL_ID, pretrained=config.CLIP_PRETRAINED, device=device
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_ID)
moderation_model = load_moderation_model(config.MODERATION_CKPT)
register_hook(clip_model, layer_number=4)
print('CLIP + moderation model loaded.')

# ── Compute harm scores for unsafe_prompts ────────────────────────────────────
def compute_harm_batch(prompts, bs=32):
    scores = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        ids = tokenizer(batch).to(device)
        eos = get_eos_positions(ids)
        cast = clip_model.transformer.get_cast_dtype()
        emb  = clip_model.token_embedding(ids).detach().clone().to(cast)
        with torch.no_grad():
            encode_text_embedding_batch(emb, eos, clip_model)
        s = torch.sigmoid(moderation_model(head_moderation['embeddings'])).squeeze(-1)
        scores.extend(s.detach().cpu().tolist())
        head_moderation['embeddings'] = None
        torch.cuda.empty_cache()
    return scores

print('Computing harm scores for unsafe_prompts...')
harm_unsafe = compute_harm_batch(df['unsafe_prompt'].tolist())

# ── Build PGD input DataFrames ────────────────────────────────────────────────
# Condition B: start PGD from unsafe_prompt (no safe replacements)
df_B = df.copy()
df_B['revised_prompt'] = df_B['unsafe_prompt']
df_B['harm']           = harm_unsafe

# Condition C: same starting point as B, random_init handled at PGD level
df_C = df_B.copy()

print(f'Harm score (revised_prompt) mean : {df["harm"].mean():.4f}')
print(f'Harm score (unsafe_prompt)  mean : {df_B["harm"].mean():.4f}')

# ── Save ──────────────────────────────────────────────────────────────────────
df_B.to_parquet(os.path.join(config.DATA_DIR, 'df_B_pgd_input.parquet'), index=False)
df_C.to_parquet(os.path.join(config.DATA_DIR, 'df_C_pgd_input.parquet'), index=False)
print('Saved df_B_pgd_input.parquet and df_C_pgd_input.parquet to data/')
print('Phase 1 complete.')
