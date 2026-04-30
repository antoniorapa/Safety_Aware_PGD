---
license: cc-by-4.0
task_categories:
  - image-classification
  - text-classification
language:
  - en
tags:
  - safety
  - text-to-image
  - adversarial
  - jailbreak
  - benchmark
  - nsfw
pretty_name: JAVIS-images
size_categories:
  - 10K<n<100K
viewer: true
dataset_info:
  - config_name: javis_nsfw
    features:
      - name: id
        dtype: int64
      - name: prompt
        dtype: string
      - name: label
        dtype: int64
      - name: category_i2p
        dtype: string
      - name: category_air
        dtype: string
      - name: subcategory_air
        dtype: string
      - name: source
        dtype: string
      - name: type
        dtype: string
      - name: image_filename
        dtype: string
      - name: image
        dtype: image
      - name: pair_id
        dtype: int64
      - name: pair_quality
        dtype: int64
    splits:
      - name: train
        num_examples: 92894
  - config_name: javis_adv
    features:
      - name: category
        dtype: string
      - name: t2i_model
        dtype: string
      - name: adv_prompt
        dtype: string
      - name: base_prompt
        dtype: string
      - name: unsafe_prompt
        dtype: string
      - name: safe_prompt
        dtype: string
      - name: human_rating
        dtype: float64
      - name: classifier_score
        dtype: float64
      - name: clip_score
        dtype: float64
      - name: label
        dtype: int64
      - name: annotator_1
        dtype: int64
      - name: annotator_2
        dtype: int64
      - name: annotator_3
        dtype: int64
      - name: image_id
        dtype: string
      - name: image
        dtype: image
    splits:
      - name: train
        num_examples: 5775
configs:
  - config_name: javis_nsfw
    data_files:
      - split: train
        path: javis_nsfw/data/*.parquet
  - config_name: javis_adv
    data_files:
      - split: train
        path: javis_adv/data/*.parquet
extra_gated_prompt: |
  This dataset contains harmful, sexual, violent, and otherwise sensitive imagery generated for the explicit purpose of research on text-to-image safety evaluation. Access is restricted to verified academic and industrial researchers working on:
  (i) safety evaluation of generative models,
  (ii) content moderation systems,
  (iii) adversarial robustness studies.
  By requesting access you agree to use the dataset for research only and to not redistribute the images.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use the dataset for research purposes only: checkbox
---

# JAVIS-images — Full prompts + images

> The full version of the JAVIS benchmark, with **prompt + metadata + images embedded** as `datasets.Image` features. Sharded parquet for fast streaming and HF viewer compatibility.

This repository hosts the **private (request-access)** image-bearing parquet shards.
For the lightweight **prompts-only** version (no images, no access gating), see
[`JAVIS-DATASET/JAVIS`](https://huggingface.co/datasets/JAVIS-DATASET/JAVIS).

## Configurations

| Config         | Rows    | Shards | Description                                                                  |
|----------------|---------|--------|------------------------------------------------------------------------------|
| `javis_nsfw`   | 99,149  | 47     | NSFW prompt corpus with images (LAION + NewRealityXL synthetic)              |
| `javis_adv`    | 5,775   | 3      | Adversarial PGD prompts evaluated on DALL·E 3 and Imagen 3, human-annotated  |

## Quick start

```python
from datasets import load_dataset

# Full NSFW with images
nsfw = load_dataset("JAVIS-DATASET/JAVIS-images", "javis_nsfw", split="train")
row = nsfw[0]
print(row["prompt"])
print(row["category_i2p"], row["pair_id"], row["pair_quality"])
row["image"].show()  # PIL.Image

# Adversarial set with images
adv = load_dataset("JAVIS-DATASET/JAVIS-images", "javis_adv", split="train")
print(adv[0]["adv_prompt"])
adv[0]["image"].show()
```

## Schema

### `javis_nsfw` (12 columns + image)

| Column            | Type    | Description                                                  |
|-------------------|---------|--------------------------------------------------------------|
| `id`              | int64   | Global unique identifier (matches public parquet)            |
| `prompt`          | string  | Text prompt                                                  |
| `label`           | int64   | 0 = safe, 1 = unsafe                                         |
| `category_i2p`    | string  | Coarse I2P category (7 classes)                              |
| `category_air`    | string  | Fine-grained AIR 2024 category (13 classes)                  |
| `subcategory_air` | string  | AIR sub-category                                             |
| `source`          | string  | Origin of prompt (LAION400M, GPT4o_MINI, DOLPHIN, …)         |
| `type`            | string  | `REAL` (LAION) or `SYNTHETIC` (LLM-generated)                |
| `image_filename`  | string  | `<id>.jpg`                                                   |
| `pair_id`         | int64?  | Identifier linking a safe row with its unsafe counterpart    |
| `pair_quality`    | int64?  | 1 = highest quality (native), 5 = lowest (no constraint)     |
| `image`           | Image   | Embedded JPEG bytes (decoded as `PIL.Image`)                 |

### `javis_adv` (14 columns + image)

| Column             | Type     | Description                                             |
|--------------------|----------|---------------------------------------------------------|
| `category`         | string   | I2P category (7 classes)                                |
| `t2i_model`        | string   | `DALL-E 3` or `Imagen 3`                                |
| `adv_prompt`       | string   | PGD-optimised adversarial prompt                        |
| `base_prompt`      | string   | Original concept prompt                                 |
| `unsafe_prompt`    | string   | Expanded unsafe variant                                 |
| `safe_prompt`      | string   | Structurally identical safe counterpart                 |
| `human_rating`     | float64  | Mean Likert score (1–5) from 3 annotators               |
| `classifier_score` | float64  | CLIP-based safety classifier output                     |
| `clip_score`       | float64  | Image–prompt CLIP similarity                            |
| `label`            | int64    | 0 = safe, 1 = unsafe (from `human_rating ≥ 3`)          |
| `annotator_1/2/3`  | int64    | Individual Likert scores                                |
| `image_id`         | string   | Image filename (hash)                                   |
| `image`            | Image    | Embedded image bytes (`PIL.Image`)                      |

## Pair quality levels (`javis_nsfw` only)

The pair-id mapping is built in 5 progressive rounds:

| Quality | Pairs   | Method                                                        |
|---------|---------|---------------------------------------------------------------|
| 1       | 21,929  | Native: same `matching_list` (exact safe↔unsafe substitution) |
| 2       | 7,676   | SYNT, same I2P category + AIR sub-category                    |
| 3       | 9,930   | REAL, same I2P category                                       |
| 4       | 5,529   | Cross-type (REAL+SYNT) within same I2P category               |
| 5       | 3,653   | Cross-category (no constraint, lowest quality)                |
| **Total** | **48,717** | **97,434 / 99,149 rows (98.3%)** — 1,715 safe rows are orphans |

To keep only the highest-quality pairs:

```python
strict = nsfw.filter(lambda x: x["pair_quality"] in (1, 2))  # 29,605 pairs
```

## License

Released under **CC BY 4.0** for research purposes only. The dataset contains
references to harmful and offensive content for the explicit purpose of safety
evaluation. Do not redistribute.
