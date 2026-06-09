import gc
import re
import string
import unicodedata
import numpy as np
import difflib
import Levenshtein
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import paired_cosine_distances

from lowresnltk import UniversalClassifier, UniversalGenerator

# ==============================================================================
#  UNIFIED NORMEVAL SYSTEM: INTRINSIC & DOWNSTREAM PERFORMANCE (DSP) SUITE
# ==============================================================================

class NormalizationEvaluator:
    def __init__(self, texts_original, texts_normalized, labels=None, classifiers=None,
                 embedding_model=None, tokenizer=None, delimiter=None):
        """
        Main evaluation hub orchestrating intrinsic alignment transformations
        and downstream neural/statistical task impacts.
        """
        self.texts_original   = list(texts_original)
        self.texts_normalized = list(texts_normalized)
        self.labels           = labels
        self.classifiers      = classifiers if classifiers is not None else []
        self.embedding_model  = embedding_model

        if len(self.texts_original) != len(self.texts_normalized):
            raise ValueError("Original and Normalized datasets must have the same number of documents.")

        if tokenizer is not None:
            self.tokenizer_func = tokenizer
        elif delimiter is not None:
            self.tokenizer_func = lambda x: [t for t in x.split(delimiter) if t]
        else:
            self.tokenizer_func = lambda x: x.split()

    # --------------------------------------------------------------------------
    #  INTERNAL GPU MEMORY MANAGEMENT
    # --------------------------------------------------------------------------

    @staticmethod
    def _release_gpu_memory(*models):
        """
        Explicitly move model weights to CPU and flush the PyTorch CUDA allocator.

        Previous versions used hasattr(model, 'model') which silently failed when
        UniversalClassifier / UniversalGenerator stored the underlying nn.Module
        under a different attribute name (e.g. self.clf, self.transformer, etc.).

        This version inspects ALL instance attributes via vars() and moves every
        nn.Module it finds to CPU — regardless of attribute naming convention.
        That guarantees CUDA pages are freed before the next model is instantiated.

        Must be called:
          - After clf_orig.predict()  → before clf_norm is created
          - After clf_norm.predict()  → before next fold's clf_orig is created
          - After gen_orig.predict()  → before gen_norm is created
          - After gen_norm.predict()  → before next fold's gen_orig is created
        """
        try:
            import torch
            _cuda = torch.cuda.is_available()
        except ImportError:
            _cuda = False

        for model in models:
            if model is None:
                continue
            try:
                if _cuda:
                    # Walk all instance attributes; move every nn.Module to CPU.
                    # This is attribute-name-agnostic — works regardless of whether
                    # the internal model is stored as .model, .clf, .encoder, etc.
                    import torch
                    for attr_val in vars(model).values():
                        if isinstance(attr_val, torch.nn.Module):
                            try:
                                attr_val.to("cpu")
                            except Exception:
                                pass
            except Exception:
                pass
            finally:
                del model

        gc.collect()

        if _cuda:
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    #  INTRINSIC MODULES
    # --------------------------------------------------------------------------

    def calculate_cr(self):
        """Calculates Vocabulary Compression Ratio (CR)."""
        vec_original  = CountVectorizer(tokenizer=self.tokenizer_func, token_pattern=None, lowercase=False)
        vec_normalized = CountVectorizer(tokenizer=self.tokenizer_func, token_pattern=None, lowercase=False)
        vec_original.fit(self.texts_original)
        vec_normalized.fit(self.texts_normalized)
        vocab_size_original   = len(vec_original.vocabulary_)
        vocab_size_normalized = len(vec_normalized.vocabulary_)
        return vocab_size_original / vocab_size_normalized if vocab_size_normalized > 0 else 0.0

    def calculate_irs(self, batch_size=32):
        """Calculates Information Retention Score (IRS) via dense semantic vectors."""
        if self.embedding_model is None:
            return 1.0

        def encode_in_batches(texts):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings.extend(self.embedding_model.encode(batch, show_progress_bar=False))
            return np.array(embeddings)

        emb_orig = encode_in_batches(self.texts_original)
        emb_norm = encode_in_batches(self.texts_normalized)
        distances = paired_cosine_distances(emb_orig, emb_norm)
        return float(np.mean(1 - distances))

    def calculate_aes(self, cr, irs):
        """Calculates Adjusted Efficiency Score (AES) balancing CR and IRS."""
        vrg = 0.0 if cr < 1.0 else 1.0 - (1.0 / float(cr))
        irs_clamped = max(0.0, min(1.0, float(irs)))
        denominator = irs_clamped + vrg
        return (2.0 * irs_clamped * vrg) / denominator if denominator > 0.0 else 0.0

    def calculate_anld(self):
        """Calculates Alignment-based Normalized Levenshtein Distance (ANLD)."""
        vocab_mapping = {}
        for orig_text, norm_text in zip(self.texts_original, self.texts_normalized):
            orig_tokens = self.tokenizer_func(orig_text)
            norm_tokens = self.tokenizer_func(norm_text)
            sm = difflib.SequenceMatcher(None, orig_tokens, norm_tokens)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag in ('equal', 'replace'):
                    orig_sub = orig_tokens[i1:i2]
                    norm_sub = norm_tokens[j1:j2]
                    max_len  = max(len(orig_sub), len(norm_sub))
                    orig_sub += [''] * (max_len - len(orig_sub))
                    norm_sub += [''] * (max_len - len(norm_sub))
                    for o_w, n_w in zip(orig_sub, norm_sub):
                        if o_w and o_w not in vocab_mapping:
                            vocab_mapping[o_w] = n_w
                elif tag == 'delete':
                    for o_w in orig_tokens[i1:i2]:
                        if o_w and o_w not in vocab_mapping:
                            vocab_mapping[o_w] = ''

        if not vocab_mapping:
            return 0.0

        total_normalized_ld, valid_words = 0.0, 0
        for w, sigma_w in vocab_mapping.items():
            if len(w) == 0:
                continue
            total_normalized_ld += Levenshtein.distance(w, sigma_w) / len(w)
            valid_words += 1
        return total_normalized_ld / valid_words if valid_words > 0 else 0.0

    # --------------------------------------------------------------------------
    #  DOWNSTREAM PERFORMANCE (DSP) EXTRINSIC MODULES
    # --------------------------------------------------------------------------

    def calculate_traditional_dsp(self, n_splits=5, random_state=42, average_method='macro', vectorizer=None):
        """
        Calculates Traditional Downstream Performance (DSP) change using classic
        scikit-learn classifiers paired with sparse matrices or static embeddings.
        """
        if self.labels is None or not self.classifiers:
            return None

        y = np.array(self.labels)
        dsp_results = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        is_precomputed     = False
        classic_transformer = None

        if vectorizer is not None:
            if isinstance(vectorizer, np.ndarray):
                X_orig_all = vectorizer
                is_precomputed = True
            elif callable(vectorizer) and not hasattr(vectorizer, "fit_transform"):
                X_orig_all = np.array(vectorizer(self.texts_original))
                X_norm_all = np.array(vectorizer(self.texts_normalized))
                is_precomputed = True
            elif hasattr(vectorizer, "fit_transform"):
                classic_transformer = vectorizer
        elif self.embedding_model is not None:
            X_orig_all = np.array(self.embedding_model.encode(self.texts_original, show_progress_bar=False))
            X_norm_all = np.array(self.embedding_model.encode(self.texts_normalized, show_progress_bar=False))
            is_precomputed = True
        else:
            classic_transformer = TfidfVectorizer(
                tokenizer=self.tokenizer_func, token_pattern=None, lowercase=False
            )

        for clf in self.classifiers:
            clf_name = type(clf).__name__
            fold_f1_orig, fold_f1_norm = [], []
            texts_orig_arr = np.array(self.texts_original)
            texts_norm_arr = np.array(self.texts_normalized)

            for train_index, test_index in skf.split(self.texts_original, y):
                y_train, y_test = y[train_index], y[test_index]

                if is_precomputed:
                    X_orig_train, X_orig_test = X_orig_all[train_index], X_orig_all[test_index]
                    X_norm_train, X_norm_test = X_norm_all[train_index], X_norm_all[test_index]
                else:
                    vec_orig = clone(classic_transformer)
                    X_orig_train = vec_orig.fit_transform(texts_orig_arr[train_index])
                    X_orig_test  = vec_orig.transform(texts_orig_arr[test_index])
                    vec_norm = clone(classic_transformer)
                    X_norm_train = vec_norm.fit_transform(texts_norm_arr[train_index])
                    X_norm_test  = vec_norm.transform(texts_norm_arr[test_index])

                m_orig = clone(clf).fit(X_orig_train, y_train)
                fold_f1_orig.append(f1_score(y_test, m_orig.predict(X_orig_test), average=average_method))
                m_norm = clone(clf).fit(X_norm_train, y_train)
                fold_f1_norm.append(f1_score(y_test, m_norm.predict(X_norm_test), average=average_method))

            mean_orig, mean_norm = np.mean(fold_f1_orig), np.mean(fold_f1_norm)
            diffs = np.array(fold_f1_norm) - np.array(fold_f1_orig)
            stat, p_val = (0, 1.0) if np.all(diffs == 0) else wilcoxon(fold_f1_orig, fold_f1_norm)
            dsp_results[clf_name] = {"DSP (Delta)": mean_norm - mean_orig, "p-value": p_val}

        return dsp_results

    def calculate_classification_dsp(self, model_name="bert-base-uncased", task_kind="binary",
                                     n_splits=3, epochs=3, batch_size=16, average_method='macro'):
        """Runs a parameterized, cross-validated deep learning sequence classification evaluation."""
        if self.labels is None or UniversalClassifier is None:
            return None

        y = np.array(self.labels)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        orig_acc_scores, norm_acc_scores = [], []
        orig_f1_scores,  norm_f1_scores  = [], []

        print(f"\n[DSP Classification] Running {n_splits}-fold evaluation on {model_name}...")
        print(f" -> Config: {epochs} Epochs | Batch Size: {batch_size}")

        for train_idx, test_idx in skf.split(self.texts_original, y):
            X_orig_train = np.array(self.texts_original)[train_idx]
            X_orig_test  = np.array(self.texts_original)[test_idx]
            X_norm_train = np.array(self.texts_normalized)[train_idx]
            X_norm_test  = np.array(self.texts_normalized)[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ── Original Space Pass ──────────────────────────────────────────
            clf_orig = UniversalClassifier(model_name=model_name, kind=task_kind)
            clf_orig.fit(list(X_orig_train), list(y_train), epochs=epochs, batch_size=batch_size)
            preds_orig = clf_orig.predict(list(X_orig_test))
            orig_acc_scores.append(np.mean(np.array(preds_orig) == np.array(y_test)))
            orig_f1_scores.append(f1_score(y_test, preds_orig, average=average_method))

            # Release before loading clf_norm — prevents dual-model GPU occupation
            self._release_gpu_memory(clf_orig)
            del clf_orig

            # ── Normalized Space Pass ────────────────────────────────────────
            clf_norm = UniversalClassifier(model_name=model_name, kind=task_kind)
            clf_norm.fit(list(X_norm_train), list(y_train), epochs=epochs, batch_size=batch_size)
            preds_norm = clf_norm.predict(list(X_norm_test))
            norm_acc_scores.append(np.mean(np.array(preds_norm) == np.array(y_test)))
            norm_f1_scores.append(f1_score(y_test, preds_norm, average=average_method))

            # Release before next fold's clf_orig
            self._release_gpu_memory(clf_norm)
            del clf_norm

        # ── Accuracy stats ───────────────────────────────────────────────────
        mean_orig_acc = np.mean(orig_acc_scores)
        mean_norm_acc = np.mean(norm_acc_scores)
        acc_diffs = np.array(norm_acc_scores) - np.array(orig_acc_scores)
        _, acc_p_val = (0, 1.0) if np.all(acc_diffs == 0) else wilcoxon(orig_acc_scores, norm_acc_scores)

        # ── F1 stats ─────────────────────────────────────────────────────────
        mean_orig_f1 = np.mean(orig_f1_scores)
        mean_norm_f1 = np.mean(norm_f1_scores)
        f1_diffs = np.array(norm_f1_scores) - np.array(orig_f1_scores)
        _, f1_p_val = (0, 1.0) if np.all(f1_diffs == 0) else wilcoxon(orig_f1_scores, norm_f1_scores)

        return {
            "Original Text Performance (Acc)":   round(mean_orig_acc, 4),
            "Normalized Text Performance (Acc)":  round(mean_norm_acc, 4),
            "Classification DSP - Acc (Delta)":   round(mean_norm_acc - mean_orig_acc, 4),
            "Accuracy Wilcoxon p-value":           round(acc_p_val, 4),
            "Original Text Performance (F1)":     round(mean_orig_f1, 4),
            "Normalized Text Performance (F1)":   round(mean_norm_f1, 4),
            "Classification DSP - F1 (Delta)":    round(mean_norm_f1 - mean_orig_f1, 4),
            "F1 Wilcoxon p-value":                round(f1_p_val, 4),
        }

    def calculate_generative_dsp(self, model_name="google/t5-efficient-mini", task="summarization",
                                  n_splits=3, epochs=2, batch_size=8):
        """Runs a parameterized, cross-validated deep learning seq2seq generation performance trace."""
        if self.labels is None or UniversalGenerator is None:
            return None

        import evaluate
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        orig_scores, norm_scores = [], []

        metric       = evaluate.load("rouge" if task == "summarization" else "sacrebleu")
        metric_key   = "rouge1" if task == "summarization" else "score"
        metric_label = "ROUGE-1" if task == "summarization" else "BLEU"

        print(f"\n[DSP Generative] Running {n_splits}-fold evaluation on {model_name}...")
        print(f" -> Config: {epochs} Epochs | Batch Size: {batch_size} | Metric: {metric_label}")

        for train_idx, test_idx in kf.split(self.texts_original):
            X_orig_train = np.array(self.texts_original)[train_idx]
            X_orig_test  = np.array(self.texts_original)[test_idx]
            X_norm_train = np.array(self.texts_normalized)[train_idx]
            X_norm_test  = np.array(self.texts_normalized)[test_idx]
            y_train, y_test = np.array(self.labels)[train_idx], np.array(self.labels)[test_idx]

            # ── Original Generator Pass ──────────────────────────────────────
            gen_orig = UniversalGenerator(model_name=model_name)
            gen_orig.fit(list(X_orig_train), list(y_train), epochs=epochs, batch_size=batch_size)

            # Convert to fp16 for inference only.
            # Training in fp32 ensures gradient stability; fp16 at inference time
            # halves both the model footprint and the beam-search KV cache, which
            # is the primary cause of OOM on ≤12 GB GPUs.
            self._convert_to_fp16(gen_orig)

            preds_orig = gen_orig.predict(list(X_orig_test))

            if task == "summarization":
                res_orig = metric.compute(predictions=preds_orig, references=list(y_test))
            else:
                res_orig = metric.compute(predictions=preds_orig, references=[[r] for r in y_test])
            orig_scores.append(res_orig[metric_key])

            # Release gen_orig before loading gen_norm
            self._release_gpu_memory(gen_orig)
            del gen_orig

            # ── Normalized Generator Pass ────────────────────────────────────
            gen_norm = UniversalGenerator(model_name=model_name)
            gen_norm.fit(list(X_norm_train), list(y_train), epochs=epochs, batch_size=batch_size)
            self._convert_to_fp16(gen_norm)
            preds_norm = gen_norm.predict(list(X_norm_test))

            if task == "summarization":
                res_norm = metric.compute(predictions=preds_norm, references=list(y_test))
            else:
                res_norm = metric.compute(predictions=preds_norm, references=[[r] for r in y_test])
            norm_scores.append(res_norm[metric_key])

            # Release gen_norm before next fold
            self._release_gpu_memory(gen_norm)
            del gen_norm

        mean_orig = np.mean(orig_scores)
        mean_norm = np.mean(norm_scores)

        return {
            f"Original Text Performance ({metric_label})":  round(mean_orig, 4),
            f"Normalized Text Performance ({metric_label})": round(mean_norm, 4),
            f"Generative DSP (Delta)":                       round(mean_norm - mean_orig, 4),
        }

    @staticmethod
    def _convert_to_fp16(model_wrapper):
        """
        Convert the underlying nn.Module inside a wrapper object to fp16.

        Called after .fit() and before .predict() in calculate_generative_dsp().
        Training always happens in fp32 (gradient stability); only inference
        is run in fp16 to halve activation and KV-cache memory.

        Uses the same attribute-agnostic vars() walk as _release_gpu_memory()
        so it works regardless of how UniversalGenerator names its inner model.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            for attr_val in vars(model_wrapper).values():
                if isinstance(attr_val, torch.nn.Module):
                    attr_val.half()
                    break
            torch.cuda.empty_cache()
        except Exception:
            pass  # graceful fallback to fp32 if conversion fails

    def calculate_kl_divergence(self, epsilon=1e-12):
        """
        Calculates KL Divergence between token frequency distributions of
        original (P) and normalized (Q) text. Closer to 0 = better preservation
        of the underlying vocabulary probability profile.
        """
        from scipy.stats import entropy

        vec = CountVectorizer(tokenizer=self.tokenizer_func, token_pattern=None, lowercase=False)
        vec.fit(self.texts_original + self.texts_normalized)

        # .sum(axis=0) returns int64; cast to float64 before adding epsilon
        # (in-place += on int64 array raises UFuncTypeError with float64 rhs)
        counts_orig = np.array(vec.transform(self.texts_original).sum(axis=0), dtype=np.float64).flatten() + epsilon
        counts_norm = np.array(vec.transform(self.texts_normalized).sum(axis=0), dtype=np.float64).flatten() + epsilon

        prob_p = counts_orig / np.sum(counts_orig)
        prob_q = counts_norm / np.sum(counts_norm)

        return float(entropy(prob_p, prob_q))

    # --------------------------------------------------------------------------
    #  ORCHESTRATION REPORT GENERATOR
    # --------------------------------------------------------------------------

    def evaluate_all(self, random_state=42, run_classification=False, classification_args=None,
                     run_generative=False, generative_args=None):
        """
        Executes unified structural evaluations. Traditional and Deep-Learning
        DSP modules run conditionally based on parameter state adjustments.
        """
        cr     = self.calculate_cr()
        irs    = self.calculate_irs()
        aes    = self.calculate_aes(cr, irs)
        anld   = self.calculate_anld()
        kl_div = self.calculate_kl_divergence()

        output = {"CR": cr, "IRS": irs, "AES": aes, "ANLD": anld, "KL_Divergence": kl_div}

        # 1. Traditional DSP Branch
        traditional_dsp = self.calculate_traditional_dsp(random_state=random_state)
        if traditional_dsp is not None:
            output["Traditional_DSP"] = traditional_dsp

        # 2. Neural Classification DSP Branch
        if run_classification and UniversalClassifier is not None:
            c_args = classification_args if classification_args is not None else {}
            output["Classification_DSP"] = self.calculate_classification_dsp(**c_args)

        # 3. Neural Generative DSP Branch
        if run_generative and UniversalGenerator is not None:
            g_args = generative_args if generative_args is not None else {}
            output["Generative_DSP"] = self.calculate_generative_dsp(**g_args)

        return output
