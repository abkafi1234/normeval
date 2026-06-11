# NormEval

[![PyPI version](https://badge.fury.io/py/normeval.svg)](https://badge.fury.io/py/normeval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-red.svg)](https://2026.emnlp.org)

A modular, language-agnostic evaluation framework for text normalization pipelines. Given an original corpus and its normalized counterpart, **NormEval** returns five intrinsic metrics and three downstream performance signals from a single call — or from individual function calls if you only need part of the picture.

---

## Why NormEval

Most normalization evaluation relies on a single metric: either vocabulary compression statistics or performance on one downstream task. Neither is enough on its own, and English-only evaluation hides language-dependent effects entirely.

NormEval treats normalization evaluation as a **multi-objective problem**. When benchmarked across five languages comparing Snowball stemming and Stanza lemmatization, the framework surfaces findings that no single metric would expose — including a 0.249 BLEU-point gap between two normalizers on Russian machine translation, invisible to both vocabulary statistics and classification F1.

---

## Installation

```bash
pip install normeval
```

**Optional dependencies** (required only for specific modules):

```bash
# For IRS and AES (semantic fidelity metrics)
pip install sentence-transformers

# For Classification DSP
pip install transformers torch

# For Generative DSP
pip install transformers torch evaluate rouge_score sacrebleu
```

---

## Quick Start

### Minimal usage — intrinsic metrics only, no labels required

```python
from normeval import NormalizationEvaluator

evaluator = NormalizationEvaluator(
    texts_original   = original_corpus,    # list of strings
    texts_normalized = normalized_corpus,  # list of strings, same length
)

cr   = evaluator.calculate_cr()
anld = evaluator.calculate_anld()
kl   = evaluator.calculate_kl_divergence()

print(f"CR: {cr:.3f}  |  ANLD: {anld:.3f}  |  KL: {kl:.3f}")
```

### Full pipeline — intrinsic + all DSP modules

```python
from normeval import NormalizationEvaluator
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

evaluator = NormalizationEvaluator(
    texts_original   = original_corpus,
    texts_normalized = normalized_corpus,
    labels           = labels,
    classifiers      = [LogisticRegression()],
    embedding_model  = SentenceTransformer(
        "paraphrase-multilingual-mpnet-base-v2"
    ),
)

results = evaluator.evaluate_all(
    run_classification  = True,
    classification_args = {
        "model_name": "xlm-roberta-base",
        "task_kind":  "multiclass",
    },
    run_generative  = True,
    generative_args = {
        "model_name": "google/mt5-small",
        "task":       "translation",
    },
)
# results keys:
# CR, IRS, AES, ANLD, KL_Divergence,
# Traditional_DSP, Classification_DSP, Generative_DSP
```

---

## Modular Design

Every evaluation module is independently callable. You do not need to run the full stack to use any single metric.

```python
evaluator = NormalizationEvaluator(
    texts_original   = original_corpus,
    texts_normalized = normalized_corpus,
)

# --- Intrinsic only (no labels, no models) ---
cr   = evaluator.calculate_cr()
anld = evaluator.calculate_anld()
kl   = evaluator.calculate_kl_divergence()

# --- Add embedding model for IRS and AES ---
from sentence_transformers import SentenceTransformer
evaluator.embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)
irs = evaluator.calculate_irs()
aes = evaluator.calculate_aes(cr, irs)

# --- Traditional DSP only (no neural models needed) ---
from sklearn.svm import LinearSVC
evaluator.labels      = labels
evaluator.classifiers = [LinearSVC()]
trad = evaluator.calculate_traditional_dsp()

# --- Classification DSP only ---
cls = evaluator.calculate_classification_dsp(
    model_name = "xlm-roberta-base",
    task_kind  = "multiclass",
    n_splits   = 5,
    epochs     = 3,
    batch_size = 16,
)

# --- Generative DSP only ---
gen = evaluator.calculate_generative_dsp(
    model_name = "google/mt5-small",
    task       = "summarization",
    n_splits   = 3,
    epochs     = 2,
    batch_size = 8,
)
```

---

## Metrics

### Intrinsic Suite

These five metrics run on `(D, D')` alone — no labels required.

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **CR** — Compression Ratio | How much the vocabulary shrank: `|V(D)| / |V(D')|` | `[0, ∞)` · higher = more compression |
| **IRS** — Information Retention Score | Semantic similarity between original and normalized documents via paired sentence embeddings | `[0, 1]` · higher = more meaning preserved |
| **AES** — Adjusted Efficiency Score | Harmonic mean of compression gain (VRG) and semantic fidelity (IRS). Penalizes normalizations that sacrifice either. | `[0, 1]` · higher = better balance |
| **ANLD** — Alignment-based Normalized Levenshtein Distance | Per-word orthographic transformation intensity, averaged over the vocabulary mapping induced by sequence alignment | `[0, ∞)` · higher = more aggressive per-word change |
| **KL Divergence** | Distributional shift from original to normalized token frequencies. Low values mean the statistical signature of the corpus was preserved. | `[0, ∞)` · lower = less shift |

> **AES** and **ANLD** are novel metrics introduced in this work. AES is the first metric to jointly measure compression efficiency and semantic preservation via a harmonic formulation. ANLD captures orthographic transformation intensity at the type level using sequence alignment before computing edit distance.

### Downstream Performance (DSP) Suite

All three modules follow a paired evaluation protocol: the same model is trained on `D` and `D'` independently, and performance is reported as a signed delta `Δ = score(D') − score(D)`.

#### Traditional DSP

Stratified k-fold cross-validation with TF-IDF representations and scikit-learn classifiers. Reports macro-F1 delta with Wilcoxon signed-rank p-value per classifier.

```python
results = evaluator.calculate_traditional_dsp(
    n_splits       = 5,          # cross-validation folds
    random_state   = 42,
    average_method = "macro",    # F1 averaging: macro, micro, weighted
    vectorizer     = None,       # custom vectorizer or None for TF-IDF
)
# returns: {clf_name: {"DSP (Delta)": float, "p-value": float}}
```

#### Classification DSP

Fine-tunes a transformer classifier independently on each corpus partition. Reports accuracy and macro-F1 delta with Wilcoxon p-values.

```python
results = evaluator.calculate_classification_dsp(
    model_name     = "bert-base-uncased",
    task_kind      = "binary",       # binary | multiclass
    n_splits       = 3,
    epochs         = 3,
    batch_size     = 16,
    average_method = "macro",
)
# returns: {
#   "Classification DSP - Acc (Delta)": float,
#   "Accuracy Wilcoxon p-value":         float,
#   "Classification DSP - F1 (Delta)":  float,
#   "F1 Wilcoxon p-value":              float,
# }
```

#### Generative DSP

Fine-tunes a seq2seq model and evaluates with BLEU (translation) or ROUGE-1 (summarization). Reports mean delta across folds.

> **Note on significance testing**: Corpus-level BLEU scores violate the symmetry assumptions of the Wilcoxon signed-rank test [(Graham et al., 2014)](https://aclanthology.org/W14-3339/). The methodologically correct alternative — paired bootstrap resampling [(Koehn, 2004)](https://aclanthology.org/W04-3250/) — is computationally prohibitive at scale. NormEval reports mean BLEU/ROUGE delta as the default; bootstrap resampling is planned as an optional parameter in a future release.

```python
results = evaluator.calculate_generative_dsp(
    model_name = "google/t5-efficient-mini",
    task       = "summarization",   # summarization | translation
    n_splits   = 3,
    epochs     = 2,
    batch_size = 8,
)
# returns: {
#   "Original Text Performance (ROUGE-1)":   float,
#   "Normalized Text Performance (ROUGE-1)": float,
#   "Generative DSP (Delta)":                float,
# }
```

---

## Constructor Reference

```python
NormalizationEvaluator(
    texts_original,    # list[str]  — original corpus
    texts_normalized,  # list[str]  — normalized corpus, same length
    labels     = None, # list       — class labels for DSP modules
    classifiers= None, # list       — sklearn classifiers for Traditional DSP
    embedding_model = None,  # SentenceTransformer — for IRS and AES
    tokenizer  = None, # callable   — custom tokenizer: str -> list[str]
    delimiter  = None, # str        — delimiter for simple splitting
)
```

If neither `tokenizer` nor `delimiter` is provided, whitespace splitting is used. This default works for most Latin-script languages; supply a language-specific tokenizer for morphologically complex or non-whitespace-delimited languages.

---

## Multilingual Support

NormEval imposes no language constraint. The `tokenizer` parameter accepts any callable that maps a string to a list of tokens.

```python
# Example: MeCab tokenizer for Japanese
import MeCab
mecab = MeCab.Tagger("-Owakati")

def japanese_tokenizer(text):
    return mecab.parse(text).strip().split()

evaluator = NormalizationEvaluator(
    texts_original   = japanese_originals,
    texts_normalized = japanese_normalized,
    tokenizer        = japanese_tokenizer,
)
```

```python
# Example: Stanza tokenizer for Arabic
import stanza
stanza.download("ar")
nlp = stanza.Pipeline("ar", processors="tokenize")

def arabic_tokenizer(text):
    return [w.text for s in nlp(text).sentences for w in s.words]

evaluator = NormalizationEvaluator(
    texts_original   = arabic_originals,
    texts_normalized = arabic_normalized,
    tokenizer        = arabic_tokenizer,
)
```

---

## Benchmark Results

Benchmarked on XNLI (classification) and Opus-100 (translation) across five languages comparing Snowball stemming and Stanza lemmatization.

### Intrinsic Metrics (XNLI, 2,000 samples per language)

| Lang | Method | CR ↑ | IRS ↑ | AES ↑ | ANLD | KL ↓ |
|------|--------|------|-------|-------|------|------|
| en | Snowball | 1.290 | 0.897 | 0.360 | 0.149 | **10.90** |
| en | Stanza   | **1.692** | **0.948** | **0.571** | 0.232 | 11.82 |
| fr | Snowball | 1.289 | 0.882 | 0.358 | 0.176 | **13.67** |
| fr | Stanza   | **1.807** | **0.948** | **0.607** | 0.344 | 16.37 |
| de | Snowball | 1.211 | 0.878 | 0.291 | 0.201 | **16.42** |
| de | Stanza   | **1.589** | **0.938** | **0.532** | 0.288 | 16.53 |
| es | Snowball | 1.417 | 0.851 | 0.437 | 0.217 | **15.99** |
| es | Stanza   | **1.792** | **0.924** | **0.598** | 0.274 | 16.01 |
| ru | Snowball | 1.424 | 0.923 | 0.450 | 0.176 | **16.03** |
| ru | Stanza   | **1.985** | **0.924** | **0.646** | 0.406 | 18.99 |

### Downstream Performance Δ

| Lang | Method | LR Δ | SVC Δ | XLM-R F1 Δ | BLEU Δ |
|------|--------|------|-------|------------|--------|
| en | Snowball | −0.0085 | +0.0020 | −0.0004 | −0.0131 |
| en | Stanza | **−0.0038** | +0.0007 | −0.0007 | **−0.0119** |
| fr | Snowball | **+0.0040** | **+0.0046** | **−0.0007** | +0.0057 |
| fr | Stanza | −0.0197 | −0.0167† | −0.0325 | **+0.0605** |
| de | Snowball | −0.0072 | −0.0196 | −0.0007 | **−0.0205** |
| de | Stanza | **+0.0016** | **−0.0168** | −0.0007 | −0.0587 |
| es | Snowball | −0.0464† | −0.0326† | 0.0000 | **−0.0148** |
| es | Stanza | **−0.0047** | **−0.0028** | 0.0000 | −0.0259 |
| ru | Snowball | −0.0280† | −0.0250 | +0.0312 | −0.0808 |
| ru | Stanza | **−0.0148** | **−0.0199** | −0.0007 | **+0.1684** |

†p ≤ 0.0625 (Wilcoxon signed-rank, 5-fold). Bold = best Δ per language. XLM-R p-values omitted (3-fold, statistically underpowered). Generative p-values omitted (see note above).

These results cannot be recovered from any single evaluation paradigm. Compression ratio alone, classification F1 alone, or English-only evaluation each give a partial and potentially misleading answer.

---

## Citation

If you use NormEval in your research, please cite:
<!--
```bibtex
@inproceedings{normeval2026,
  title     = {{NormEval}: A Modular Multilingual Evaluation
               Framework for Text Normalization Pipelines},
  author    = {[anonymized]},
  booktitle = {Proceedings of EMNLP 2026: System Demonstrations},
  year      = {2026},
  address   = {Budapest, Hungary},
}
```
-->
---

## Contributing

Contributions are welcome. To add support for a new DSP module or evaluation metric, open an issue describing the metric's mathematical formulation and the task type it targets.

For bug reports, include the Python version, package version (`pip show normeval`), and a minimal reproducible example.

---

## License

MIT License. Copyright © 2026. See [LICENSE](LICENSE) for details.
