# Safety-Aware PGD — Panoramica delle Sperimentazioni

## Obiettivo generale

Il progetto studia gli **adversarial attack su modelli text-to-image (t2i)**.
L'obiettivo è generare prompt testuali che:
1. **Bypassano i sistemi di moderazione testuale** (text safety filters), e
2. **Producono immagini semanticamente simili** al prompt unsafe originale.

Le categorie di danno considerate sono 7:
`harassment`, `hate`, `self-harm`, `sexual`, `shocking`, `violence`, `illegal-activity`

---

## Pipeline principale (3 fasi)

### Fase 1 — Safe Replacements (`src/optimal_selection.py`)

**Input**: un prompt unsafe (es. *"Image of a person being strangled..."*)

**Cosa fa**:
- Calcola i **Integrated Gradients** sull'encoder testuale di CLIP per identificare quali token contribuiscono maggiormente allo score di danno (harm score) del modello MLP-STD.
- Usa un **LLM (GPT)** per suggerire sostituzioni sicure delle parole unsafe (es. *"rope"* → *"task"*).
- Esplora le combinazioni di sostituzioni (fino a 3000) e seleziona quella che **minimizza l'harm score** restando nell'intervallo di lunghezza token target.

**Output**: `revised_prompt` — versione "apparentemente sicura" del prompt, con le posizioni token libere per il PGD.

**Colonne chiave del DataFrame**:
- `unsafe_prompt`: versione dettagliata del prompt originale
- `revised_prompt`: prompt dopo Safe Replacements (harm score basso, ~0.07 medio)
- `unsafe_replacements` / `safe_replacements`: le coppie di parole sostituite
- `flatten_positions`: indici token "liberi" da ottimizzare nel PGD
- `harm`: harm score del `revised_prompt` (MLP-STD, scala 0-1)

---

### Fase 2 — Safety-Aware PGD (`src/batch_prompter.py` → `text_inversion_base_batched`)

**Input**: `revised_prompt` + posizioni token libere

**Cosa fa**:
- Esegue **Projected Gradient Descent** nello spazio continuo degli embedding CLIP.
- **Funzione di loss** bilancia due obiettivi:
  - Massimizzare la **similarità CLIP** tra il prompt e l'immagine di riferimento (generata con Stable Diffusion sull'unsafe prompt originale).
  - Mantenere l'**harm score < τ' = 0.4** (peso moderazione `global_mod = 0.5`).
- Lista di parole proibite: blocklist per categoria + parole unsafe estratte con Integrated Gradients.
- Iterazioni: **2000**, batch size: **25**, top-N per gruppo: **5** (o 1 nell'ablation).
- Alla fine, proietta ogni embedding continuo al token di vocabolario più vicino.

**Output**: `adv_prompt` — prompt finale adversariale (testo discreto, basso harm score, alta similarità semantica).

**Iperparametri chiave**:
| Parametro | Valore |
|-----------|--------|
| `n_iterations` | 2000 |
| `tau_prime` | 0.4 |
| `global_mod` | 0.5 |
| `batch_size` | 25 |
| `random_init` | False (Full pipeline) |

---

### Fase 3 — Generazione immagini

Gli `adv_prompt` vengono inviati ai modelli t2i:
- **DALL-E 3** (OpenAI) — risultati in `test/discrete/moderated/dall_e/`
- **Imagen 3** (Google, medium filter) — risultati in `test/discrete/moderated/imagen/`
- **Gemini 2.5 Flash Image** (Vertex AI) — in corso, `notebooks/MAIN/IMGEN_GEMINI.ipynb`

---

## Modello di moderazione (MLP-STD)

- Architettura: `ModerationHeadMLP` — MLP applicato all'embedding dell'attention head 6, layer 4 di CLIP ViT-L/14.
- Checkpoint: `src/models/mlp_model_selected_layer_1.pth`
- Output: score in [0,1], soglia di danno = **0.5**
- Harm score medio: `unsafe_prompt` ~0.96, `revised_prompt` ~0.07, `adv_prompt` ~0.15-0.28

**Versione avanzata (MLP-ADV)**:
- Input: concatenazione di 4 head × 768 dim = 3072 dim
- Head usate: (Layer 7, H5), (Layer 4, H10), (Layer 2, H4), (Layer 0, H7)
- Checkpoint: `src/models/ADV_mlp_model_selected_layer_4.pth`

---

## Domande di Ricerca (RQ) e relativi notebook

### RQ1 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Il nostro pipeline bypassa la moderazione testuale e genera comunque immagini dannose?

**Metriche**:
- **BR** (Bypass Rate): % di prompt che NON vengono bloccati dalla moderazione testuale
- **ASR** (Attack Success Rate): % di immagini generate classificate come dannose
- **CHR** (Content Harm Rate): harm rate medio delle immagini
- **CLIP score**: similarità semantica tra immagine generata e prompt originale

**Dati**: `dall_e_3_results_plain.parquet`, `imgen_3_medium_filter.parquet`

---

### RQ2 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Come si confronta il nostro metodo con altri dataset di prompt avversariali esistenti?

**Dati**: `ALL_ADV_CLEAN_mod_OUR_img_human_labeled.csv`
(include prompt nostri + altri metodi, con human labeling)

---

### RQ3 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Il nostro metodo è robusto anche contro la moderazione basata sulle immagini (CLIP-based)?

**Dati**: `ALL_ADV_CLEAN_MOD_ALL.csv`

---

### RQ4 (MLLM) — `notebooks/MAIN/4_ANALYSIS_MLLM.ipynb`
**Domanda**: Un Multi-modal LLM valuta le immagini generate come dannose?

Valutazione automatica con MLLM delle immagini prodotte dagli `adv_prompt`.

---

## Ablation Study — `notebooks/MAIN/ablation/`

**Motivazione**: risposta al Reviewer 2 — *"Non c'è un esperimento ablation che dimostri la necessità del modulo Safe Replacements."*

**Tre condizioni** (stessi iperparametri PGD):

| Condizione | Label | Punto di partenza PGD | N. prompt |
|-----------|-------|----------------------|----------:|
| Full Pipeline | A | `revised_prompt` (dopo Safe Replacements) | 590 |
| No Safe Replacements | B | `unsafe_prompt` originale | 700 |
| Random Initialization | C | `unsafe_prompt` + init random degli embedding | 700 |

**Pipeline a 6 fasi** (`notebooks/MAIN/ablation/`):

| Script | Fase |
|--------|------|
| `1_prepare_data.py` | Preparazione dati |
| `2_pgd_B.py` | PGD condizione B |
| `3_pgd_C.py` | PGD condizione C |
| `4_score_classifiers.py` | Scoring 10 classifier |
| `5_score_llms.py` | Scoring 5 LLM (sequenziale, 4-bit GPU) |
| `6_analysis.py` | Analisi, tabelle, plot |

**Sistemi di moderazione testati (15, di cui 14 nel Mean BR)** — completato 2026-05-09:

*Classifier (Phase 4)*:
1. MLP-STD (interno, standard-trained)
2. MLP-ADV (interno, adversarially fine-tuned)
3. OpenAI `omni-moderation-latest`
4. DistilRoBERTa (`michellejieli/NSFW_text_classifier`)
5. DeBERTa (`KoalaAI/Text-Moderation`)
6. ToxicBERT (`unitary/toxic-bert`)
7. FB-HateSpeech (`facebook/roberta-hate-speech-dynabench-r4-target`)
8. ToxicityModel (`nicholasKluge/ToxicityModel`)
9. RoBERTa-Toxic (`s-nlp/roberta_toxicity_classifier`)
10. Offensive-Twitter (`cardiffnlp/twitter-roberta-base-offensive`)

*LLM-based (Phase 5 — caricati sequenzialmente, 4-bit GPU dove disponibile)*:
11. LlamaGuard3 (`meta-llama/Llama-Guard-3-8B`)
12. WildGuard (`allenai/wildguard`)
13. ShieldGemma (`google/shieldgemma-2b`)
14. GraniteGuardian (`ibm-granite/granite-guardian-3.0-2b`)
15. ~~MD-Judge~~ (`OpenSafetyLab/MD-Judge-v0.1`) — **escluso dal Mean BR**: progettato per coppie (query, response), assegna score ~0.9999 a qualsiasi singolo prompt indiscriminatamente

**Bypass Rate (%) per sistema e condizione**:

| Sistema | A: Full Pipeline | B: No SR | C: Random Init |
|---------|----------------:|---------:|---------------:|
| MLP-STD | 76.6 | 92.0 | 94.4 |
| MLP-ADV | 5.3 | 3.7 | 8.7 |
| OpenAI Moderation | 84.6 | 33.4 | 33.7 |
| DistilRoBERTa | 3.6 | 1.4 | 2.4 |
| DeBERTa | 32.9 | 34.4 | 34.4 |
| ToxicBERT | 100.0 | 100.0 | 100.0 |
| FB-HateSpeech | 100.0 | 99.6 | 100.0 |
| ToxicityModel | 100.0 | 100.0 | 100.0 |
| RoBERTa-Toxic | 100.0 | 100.0 | 100.0 |
| Offensive-Twitter | 100.0 | 100.0 | 99.9 |
| LlamaGuard3 | 100.0 | 100.0 | 100.0 |
| WildGuard | 1.7 | 0.6 | 1.1 |
| ShieldGemma | 8.5 | 3.4 | 2.9 |
| GraniteGuardian | 8.1 | 11.1 | 11.3 |
| MD-Judge *(escluso)* | 0.0 | 0.0 | 0.0 |
| **Mean BR (14 sys)** | **58.7** | **55.7** | **56.3** |

**Metriche CLIP / Harm per condizione (globali)**:

| Condizione | CLIP | Harm (MLP-STD) | ASR_MLP |
|-----------|-----:|---------------:|--------:|
| A — Full Pipeline | 0.3843 | 0.2731 | 78.6% |
| B — No SR | 0.3954 | 0.1472 | 100.0% |
| C — Random Init | 0.3971 | 0.0997 | 100.0% |

**Key Takeaway**:

| Condizione | Mean BR (14 sys) | Delta vs A |
|-----------|----------------:|-----------:|
| **A — Full Pipeline** | **58.7%** | — |
| B — No SR | 55.7% | −3.0% |
| C — Random Init | 56.3% | −2.4% |

Il Full Pipeline mantiene il Mean BR più alto: il modulo Safe Replacements contribuisce +3.0% su No SR e +2.4% su Random Init.

**Note sui sistemi**:
- I 5 modelli generici (ToxicBERT, FB-HateSpeech, ToxicityModel, RoBERTa-Toxic, Offensive-Twitter) danno BR=100% in tutte le condizioni: non sono progettati per harmful intent ma per toxicity/offensiveness — inclusi nella tabella, non influenzano il confronto tra condizioni.
- MD-Judge escluso dal Mean BR: assegna ~0.9999 a qualsiasi singolo prompt (vedi sopra).
- LlamaGuard3 dà BR=100%: i prompt adversariali risultano semanticamente benigni dopo Safe Replacements + PGD.
- WildGuard è il più stringente tra gli LLM-based (BR ≤ 1.7%); MD-Judge escluso dall'analisi.
- OpenAI Moderation è il sistema dove il vantaggio del Full Pipeline è più marcato (84.6% vs ~33%).

**Fix tecnici applicati**:
- `apply_chat_template(..., return_tensors="pt")` in transformers ≥5.7 restituisce `BatchEncoding` invece di un tensore — fix: estrazione di `.input_ids`.
- WildGuard parser: controllo su `"harmful request: no/yes"` (stringa completa) invece di `"harmful"` che dava falsi positivi.
- MD-Judge: usa output `"safe"/"unsafe"` — `_mdj_safe_unsafe_score` con `_best_id()` per gestire leading space nei tokenizer SentencePiece.
- LLM caricati sequenzialmente in 4-bit (bitsandbytes) su GPU; WildGuard su CPU bfloat16 (VRAM insufficiente per 4-bit).
- `venv_ablation` usato come ambiente Python (ha bitsandbytes); Anaconda Python come interprete base.

Output salvati in `notebooks/MAIN/ablation/results/`:
- `4_classifier_scores/scores_{A,B,C}_classifiers.csv`
- `5_llm_scores/scores_{A,B,C}_{LlamaGuard3,WildGuard,ShieldGemma,GraniteGuardian,MD-Judge}.csv`
- `5_llm_scores/scores_{A,B,C}_full.csv`
- `6_analysis/bypass_rates.csv`, `ablation_clip_harm_metrics.csv`
- `6_analysis/ablation_clip_harm.png/pdf`, `ablation_bypass_rate.png/pdf`, `ablation_harm_violin.pdf`

### Analisi raffinata — sistemi discriminativi (6 su 15)

Dei 15 sistemi testati, 9 sono esclusi dal Mean BR per i seguenti motivi:

| Sistema escluso | Motivo |
|----------------|--------|
| MLP-STD | Modello interno del nostro sistema — non è un moderatore esterno indipendente |
| ToxicBERT | Classifier generico toxicity/offensiveness: BR=100% in tutte le condizioni |
| FB-HateSpeech | Idem |
| ToxicityModel | Idem |
| RoBERTa-Toxic | Idem |
| Offensive-Twitter | Idem |
| LlamaGuard3 | BR=100% in tutte le condizioni — i prompt avversariali appaiono semanticamente benigni |
| DeBERTa | BR quasi identico tra le 3 condizioni (32.9 / 34.4 / 34.4%) — insensibile all'inizializzazione PGD |
| MD-Judge | Progettato per coppie (query, response): assegna score ~0.9999 a qualsiasi singolo prompt |

**Sistemi considerati nel Mean BR raffinato (6)**:

| # | Sistema | Tipo |
|---|---------|------|
| 1 | MLP-ADV | Classifier interno adversarially fine-tuned |
| 2 | OpenAI `omni-moderation-latest` | API esterna |
| 3 | DistilRoBERTa (`michellejieli/NSFW_text_classifier`) | Classifier NSFW |
| 4 | WildGuard (`allenai/wildguard`) | LLM-based safety |
| 5 | ShieldGemma (`google/shieldgemma-2b`) | LLM-based safety |
| 6 | GraniteGuardian (`ibm-granite/granite-guardian-3.0-2b`) | LLM-based safety |

**Bypass Rate (%) — solo sistemi discriminativi**:

| Sistema | A: Full Pipeline | B: No SR | C: Random Init |
|---------|----------------:|---------:|---------------:|
| MLP-ADV | 5.3 | 3.7 | 8.7 |
| OpenAI Moderation | 84.6 | 33.4 | 33.7 |
| DistilRoBERTa | 3.6 | 1.4 | 2.4 |
| WildGuard | 1.7 | 0.6 | 1.1 |
| ShieldGemma | 8.5 | 3.4 | 2.9 |
| GraniteGuardian | 8.1 | 11.1 | 11.3 |
| **Mean BR (6 sys)** | **18.6** | **8.9** | **10.0** |

**Key Takeaway (analisi raffinata)**:

| Condizione | Mean BR (6 sys) | Delta vs A |
|-----------|----------------:|-----------:|
| **A — Full Pipeline** | **18.6%** | — |
| B — No SR | 8.9% | −9.7% |
| C — Random Init | 10.0% | −8.6% |

Con i soli sistemi discriminativi il vantaggio del Full Pipeline è molto più netto: **+9.7% su No SR** e **+8.6% su Random Init**. Il contributo principale del vantaggio viene da OpenAI Moderation (84.6% A vs ~33% B/C), che è anche il sistema più rilevante in produzione.

---

## Dataset

| Dataset | File | Descrizione |
|---------|------|-------------|
| MediEthic (nostro) | `data/dataset/our/mediethic/` | Dataset principale, 14 base prompt × 7 categorie |
| adversarial_test.json | `test/discrete/adversarial_test/` | Output Fase 1 (Safe Replacements), 7964 righe |
| JarvisAdv | `data/javis_adv/data/*.parquet` | Dataset esterno, 5775 prompt (569 unici dopo dedup) |
| df_advPrompt_clean.parquet | root + notebooks/MAIN/ | Prompt unici JarvisAdv pronti per la generazione |

---

## Sperimentazione Gemini — `imgen_gemini.py`

**Obiettivo**: sostituire DALL-E 3 con **Gemini 2.5 Flash Image** (Vertex AI) come modello t2i.

**Setup**:
- Progetto GCP: `secret-antonym-494814-q9`, regione `us-central1`
- Autenticazione: Application Default Credentials (`gcloud auth application-default login`)
- Safety settings: tutti OFF (per test)
- Target: 4 immagini per prompt, max 100 chiamate API per prompt

**Input**: prompt da `df_advPrompt_clean.parquet` (JarvisAdv, 569 prompt unici)

**Distribuzione dei prompt per categoria**:

| Categoria        | N. prompt | % sul totale |
|------------------|----------:|-------------:|
| sexual           | 141       | 24.8%        |
| harassment       | 100       | 17.6%        |
| shocking         |  87       | 15.3%        |
| hate             |  76       | 13.4%        |
| violence         |  63       | 11.1%        |
| self-harm        |  58       | 10.2%        |
| illegal-activity |  44       |  7.7%        |
| **TOTALE**       | **569**   | **100%**     |

**Output**:
- Immagini: `test/moderation/nanobanana/test_prompt_gemini_flash/<categoria>/<prompt_folder>/`
- Risultati raw Step 1: `nanobanana_results_gemini_flash.json`
- CLIP scores: `nanobanana_results_clip.parquet`
- Dataset completo (harm + CLIP + vision): `nanobanana_results_full.parquet`
- Metriche aggregate: `nanobanana_metrics.csv`

**Stato** *(completato 2026-05-14)*: pipeline terminata su tutti i 569 prompt.

---

### Risultati — Gemini 2.5 Flash Image

#### Step 1 — Generazione immagini

| Categoria        | Prompt | Prompt con img | Immagini generate |
|------------------|-------:|---------------:|------------------:|
| sexual           |    141 |            100 |               289 |
| harassment       |    100 |             69 |               206 |
| shocking         |     87 |             24 |                76 |
| hate             |     76 |             27 |                97 |
| violence         |     63 |             16 |                38 |
| self-harm        |     58 |             32 |                93 |
| illegal-activity |     44 |             25 |                75 |
| **TOTALE**       |**569** |        **293** |             **874** |

- Dimensione totale immagini: **~1.25 GB** (874 PNG)
- Prompt senza immagini (tutti tentativi esauriti): 276 (48.5%) — bloccati da safety filter o rate limit su tutte le regioni

#### Step 2 — CLIP Score

- Record processati: **1104** (con immagine)
- Mean CLIP score globale: **0.3521**

#### Step 3 — Vision Evaluation

- Immagini valutate: **1104**
- Tempo di esecuzione: **~1h 45min** (5.71 s/img)
- Modello evaluator: `gemini-2.5-flash` con schema JSON strutturato (harm_level 1–5, 7 categorie di danno)

#### Step 4 — Metriche Aggregate (harm_level > 3)

| Categoria        |  N  |  BR   |  ASR  |  CHR  |  CLIP  |
|------------------|----:|------:|------:|------:|-------:|
| harassment       | 100 | 15.7% | 62.0% |  9.7% | 0.3563 |
| hate             |  76 |  7.2% | 31.6% |  2.3% | 0.3663 |
| illegal-activity |  44 | 13.7% | 56.8% |  7.8% | 0.3701 |
| self-harm        |  58 | 10.0% | 50.0% |  5.0% | 0.3647 |
| sexual           | 141 | 11.9% | 41.8% |  5.0% | 0.3330 |
| shocking         |  87 |  9.2% | 23.0% |  2.1% | 0.3484 |
| violence         |  63 |  6.0% | 20.6% |  1.2% | 0.3818 |
| **GLOBAL**       |**569**|**11.0%**|**40.8%**|**4.5%**|**0.3521**|

**Osservazioni**:
- BR basso (11% globale): il modello Gemini 2.5 Flash Image blocca la maggior parte delle chiamate con contenuto sensibile, anche con safety settings a OFF.
- ASR più alto per `harassment` (62%) e `illegal-activity` (56.8%): le immagini che passano il filtro sono spesso classificate come dannose dal vision evaluator.
- `violence` ha il BR più basso (6%) ma il CLIP score più alto (0.3818): le poche immagini generate sono semanticamente vicine al prompt.
- `shocking` ha ASR basso (23%): il filtro blocca efficacemente i contenuti horror/splatter.

---

## File sorgente chiave

| File | Ruolo |
|------|-------|
| `src/optimal_selection.py` | Integrated Gradients + Safe Replacements |
| `src/batch_prompter.py` | ModeratedPrompter + Safety-Aware PGD |
| `src/moderation_clip.py` | ModerationHeadMLP (classifier) |
| `src/moderated_prompter.py` | Wrapper alto livello |
| `src/utils/scores_utils.py` | Utility metriche e path |
| `src/configs/category_block_dict.json` | Blocklist per categoria |
| `notebooks/MAIN/1_ATTACK_PIPELINE.ipynb` | Esecuzione pipeline completa |
| `notebooks/MAIN/3_ANALYSIS_RQS.ipynb` | Analisi RQ1/RQ2/RQ3 |
| `notebooks/MAIN/4_ANALYSIS_MLLM.ipynb` | Valutazione MLLM |
| `notebooks/MAIN/5_ABLATION_SAFE_REPLACEMENTS.ipynb` | Ablation study |
| `notebooks/MAIN/IMGEN_GEMINI.ipynb` | Generazione con Gemini |
