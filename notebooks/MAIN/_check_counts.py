import pandas as pd, os, sys
sys.path.insert(0, os.path.abspath('../../'))
ROOT = os.path.abspath('../../')
RESULTS_DIR = os.path.join(ROOT, 'test/discrete/ablation')
for name, f in [('A','results_A_full_pipeline.parquet'),('B','results_B_no_sr.parquet'),('C','results_C_random_init.parquet')]:
    df = pd.read_parquet(os.path.join(RESULTS_DIR, f))
    bp = df['base_prompt'].nunique()
    ap = df['adv_prompt'].nunique()
    cats = sorted(df['category'].unique()) if 'category' in df.columns else 'N/A'
    print(f'Condition {name}: {len(df)} rows | {bp} base_prompts | {ap} unique adv_prompts')
    print(f'  categories: {cats}')
    print(f'  adv_prompts per base_prompt: {df.groupby("base_prompt")["adv_prompt"].nunique().describe().to_dict()}')
    print()
