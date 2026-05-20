# normeval: Scientific Evaluation Suite for Text Normalization

`normeval` is a robust, mathematically grounded Python framework for evaluating text normalization algorithms. It goes beyond simple string matching by combining semantic preservation, vocabulary compression, character-level fidelity, and statistically rigorous downstream impact analysis.

---

## 🔬 Motivation: Why do we need this?

Evaluating text normalization (such as stemming, lemmatization, spell-correction, or noise reduction) is traditionally a fragmented process. Researchers often rely solely on downstream accuracy, which is highly susceptible to dataset bias and random initialization.

A **"good"** normalization algorithm must balance two competing objectives:

1.  **Compression**: It must collapse noisy variations of words into a single representation to reduce the feature space.
2.  **Preservation**: It must not destroy the underlying semantic meaning or map unrelated words together.

`normeval` provides a holistic, unified suite to measure this balance, ensuring your normalization techniques are rigorously tested before being deployed in NLP pipelines.

---

## 📦 Installation

Install directly from PyPI:

```bash
pip install normeval
```
### 🚀 Quick StartPythonfrom normeval import NormalizationEvaluator

```python
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# 1. Your parallel datasets
texts_original = [
    "The cats are running!!",
    "A running cat..."
]

texts_normalized = [
    "the cat run",
    "a run cat"
]

labels = [1, 0]

# 2. Define your models
classifiers = [LogisticRegression()]
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Initialize Evaluator
evaluator = NormalizationEvaluator(
    texts_original,
    texts_normalized,
    labels=labels,
    classifiers=classifiers,
    embedding_model=embedder
)

# 4. Run the full suite
results = evaluator.evaluate_all(lang="Global")

print(results)
```
# 📊 Metrics (Function Deep-Dive)

`normeval` evaluates text normalization across **macroscopic**, **semantic**, and **downstream** dimensions.

---

## 1. Compression Ratio (CR)

- **Function:** `calculate_cr()`
- **What it is:** Measures the macroscopic reduction in vocabulary size.
- **The Math:**  
  `CR = |V_original| / |V_normalized|`
- **Interpretation:**
  - CR = 1.0 → No compression occurred (bijective mapping).
  - CR > 1.0 → Multiple noisy variants successfully collapsed into fewer normalized forms.

---

## 2. Information Retention Score (IRS)

- **Function:** `calculate_irs(batch_size=32)`
- **What it is:** Measures how much semantic meaning survived the normalization process.
- **Mechanism:** Converts both original and normalized texts into dense vector embeddings and computes paired cosine similarity.
- **Interpretation:**
  - Range: [-1, 1]
  - A score near 1.0 indicates semantic meaning was preserved successfully.

---

## 3. Algorithm Effectiveness Score (AES)

- **Function:** `calculate_aes(cr, irs)`
- **What it is:** A harmonic mean balancing Compression (CR) and Preservation (IRS).
- **The Math:**
  `VRG = 1-1/CR`
  `AES = (2 × IRS × VRG) / (IRS + VRG)`
- **Interpretation:** Punishes algorithms that are:
  - Too aggressive (High CR, low IRS)
  - Too passive (High IRS, low CR)

---

## 4. Average Normalized Levenshtein Distance (ANLD)

- **Function:** `calculate_anld()`
- **What it is:** A micro-level safety metric measuring character-level fidelity.
- **The Math:**  
  `ANLD = (1 / |V|) Σ [ LD(w, σ(w)) / |w| ]`
- Where:
  - `LD` = Levenshtein Distance  
  - `σ(w)` = normalized form of word `w`

---

## 5. Model Performance Delta (MPD) & Statistical Significance

- **Function:** `calculate_mpd(n_splits=5, average_method='weighted')`
- **What it is:** Measures the impact on classification performance using Stratified N-Fold CV.
- **Statistical Rigor:** Applies the **Wilcoxon Signed-Rank Test** across CV folds (p < 0.05).
- **Leakage Prevention:** `TfidfVectorizer` is fit inside each CV fold to strictly prevent train/test leakage.

## 🏗️ Project Goals
Reproducible evaluation pipelines.

Statistically grounded benchmarking.

Semantic preservation analysis.

Language-agnostic evaluation.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, open an issue first to discuss proposed improvements.

## 📄 License
This project is licensed under the MIT License.

⭐ Citation
If you use normeval in your research, please cite: underway 
