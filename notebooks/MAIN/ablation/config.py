"""Shared configuration for all ablation phases."""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(THIS_DIR, '../../..'))
sys.path.insert(0, ROOT)

# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(THIS_DIR, 'data')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

RES_PGD_B        = os.path.join(RESULTS_DIR, '2_pgd_B')
RES_PGD_C        = os.path.join(RESULTS_DIR, '3_pgd_C')
RES_CLASSIFIERS  = os.path.join(RESULTS_DIR, '4_classifier_scores')
RES_LLM          = os.path.join(RESULTS_DIR, '5_llm_scores')
RES_ANALYSIS     = os.path.join(RESULTS_DIR, '6_analysis')

for _d in [DATA_DIR, RES_PGD_B, RES_PGD_C, RES_CLASSIFIERS, RES_LLM, RES_ANALYSIS]:
    os.makedirs(_d, exist_ok=True)

# ── Source paths (original locations, read-only) ──────────────────────────────
SRC_ADV_TEST_JSON  = os.path.join(ROOT, 'test/discrete/adversarial_test/adversarial_test.json')
SRC_DALL_E_PARQUET = os.path.join(ROOT, 'test/discrete/moderated/dall_e/dall_e_3_results_plain.parquet')
SRC_PARQUET_A      = os.path.join(ROOT, 'test/discrete/ablation/results_A_full_pipeline.parquet')
SRC_PARQUET_B      = os.path.join(ROOT, 'test/discrete/ablation/results_B_no_sr.parquet')
SRC_PARQUET_C      = os.path.join(ROOT, 'test/discrete/ablation/results_C_random_init.parquet')

# ── Data paths (working copies in data/) ─────────────────────────────────────
ADV_TEST_JSON  = os.path.join(DATA_DIR, 'adversarial_test.json')
DALL_E_PARQUET = os.path.join(DATA_DIR, 'dall_e_3_results_plain.parquet')
PARQUET_A      = os.path.join(DATA_DIR, 'results_A_full_pipeline.parquet')
PARQUET_B      = os.path.join(DATA_DIR, 'results_B_no_sr.parquet')
PARQUET_C      = os.path.join(DATA_DIR, 'results_C_random_init.parquet')

# Output of phases 2 and 3 (written here, then used downstream)
OUT_PARQUET_B  = os.path.join(RES_PGD_B, 'results_B_no_sr.parquet')
OUT_PARQUET_C  = os.path.join(RES_PGD_C, 'results_C_random_init.parquet')

# ── Model checkpoints ─────────────────────────────────────────────────────────
MODERATION_CKPT     = os.path.join(ROOT, 'src/models/mlp_model_selected_layer_1.pth')
MODERATION_CKPT_ADV = os.path.join(ROOT, 'src/models/ADV_mlp_model_selected_layer_4.pth')
CATEGORY_CFG        = os.path.join(ROOT, 'src/configs/category_block_dict.json')
REFERENCE_IMG_DIR   = os.path.join(ROOT, 'OtherCode/data/images/sd/reference')

# ── Credentials ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = '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'
HF_TOKEN       = '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'

# ── CLIP model ────────────────────────────────────────────────────────────────
CLIP_MODEL_ID   = 'ViT-L-14'
CLIP_PRETRAINED = 'laion2b_s32b_b82k'
ADV_HEAD_LAYERS = [(7, 5), (4, 10), (2, 4), (0, 7)]

# ── LLM model IDs ─────────────────────────────────────────────────────────────
LG3_MODEL_ID = 'meta-llama/Llama-Guard-3-8B'
WG_MODEL_ID  = 'allenai/wildguard'
SG_MODEL_ID  = 'google/shieldgemma-2b'
GG_MODEL_ID  = 'ibm-granite/granite-guardian-3.0-2b'
MDJ_MODEL_ID = 'OpenSafetyLab/MD-Judge-v0.1'

# ── PGD hyperparameters ───────────────────────────────────────────────────────
N_ITERATIONS = 2000
TAU_PRIME    = 0.4
GLOBAL_MOD   = 0.5
BATCH_SIZE   = 25
TOP_N        = 1
N_S          = 50  # adversarial variants to keep per base prompt

# ── Scoring thresholds ────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.5
HARM_THRESHOLD  = 0.5

# ── Telegram notifications ────────────────────────────────────────────────────
TELEGRAM_TOKEN   = '8779083388:AAE9m7dlXSm2ql1kYsqcwcNy34pNYUSYFuo'
TELEGRAM_CHAT_ID = '139781098'
