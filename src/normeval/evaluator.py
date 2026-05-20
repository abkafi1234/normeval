import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Levenshtein
from collections import Counter
from scipy.stats import wilcoxon
from scipy.ndimage import gaussian_filter1d
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.model_selection import StratifiedKFold

class NormalizationEvaluator:
    def __init__(self, texts_original, texts_normalized, labels=None, classifiers=None, embedding_model=None):
        """
        Initializes the evaluator with parallel datasets and required models.
        """
        # Ensure texts are lists of strings (handles pandas Series automatically)
        self.texts_original = list(texts_original)
        self.texts_normalized = list(texts_normalized)
        
        self.labels = labels
        self.classifiers = classifiers if classifiers is not None else []
        self.embedding_model = embedding_model
        
        # Basic validation
        if len(self.texts_original) != len(self.texts_normalized):
            raise ValueError("Original and Normalized datasets must have the same number of documents.")
        
    def calculate_cr(self):
            """
            Calculates the Compression Ratio (CR).
            Formula: Unique Words Before Transformation / Unique Words After Transformation
            """
            # Initialize CountVectorizers
            vec_original = CountVectorizer()
            vec_normalized = CountVectorizer()
            
            # Fit the vectorizers to extract the vocabularies
            vec_original.fit(self.texts_original)
            vec_normalized.fit(self.texts_normalized)
            
            # Get the size of the vocabularies
            vocab_size_original = len(vec_original.vocabulary_)
            vocab_size_normalized = len(vec_normalized.vocabulary_)
            
            # Edge case: Avoid division by zero if normalized text is completely empty
            if vocab_size_normalized == 0:
                return 0.0
                
            # Calculate CR
            cr = vocab_size_original / vocab_size_normalized
            
            print(f"Original Vocab Size: {vocab_size_original}")
            print(f"Normalized Vocab Size: {vocab_size_normalized}")
            print(f"Calculated CR: {cr:.4f}")
            
            return cr
    def calculate_mpd(self, n_splits=5, random_state=42, average_method='weighted'):
        """
        Calculates Model Performance Delta (MPD) using N-Fold Cross-Validation 
        and determines statistical significance via Wilcoxon Signed-Rank Test.
        """
        if self.labels is None:
            raise ValueError("Ground truth labels must be provided to calculate MPD.")
        if not self.classifiers:
            print("No classifiers provided. Skipping MPD calculation.")
            return None

        # Convert to numpy arrays for advanced indexing during CV
        y = np.array(self.labels)
        texts_orig = np.array(self.texts_original)
        texts_norm = np.array(self.texts_normalized)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mpd_results = {}

        for clf in self.classifiers:
            clf_name = type(clf).__name__
            fold_f1_orig = []
            fold_f1_norm = []

            print(f"\n--- Running {n_splits}-Fold CV for {clf_name} ---")

            for i, (train_index, test_index) in enumerate(skf.split(texts_orig, y)):
                # 1. Split Data
                X_orig_train, X_orig_test = texts_orig[train_index], texts_orig[test_index]
                X_norm_train, X_norm_test = texts_norm[train_index], texts_norm[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # 2. Vectorize (Fit on Train only to prevent leakage)
                vec_orig = TfidfVectorizer()
                X_orig_train_vec = vec_orig.fit_transform(X_orig_train)
                X_orig_test_vec = vec_orig.transform(X_orig_test)

                vec_norm = TfidfVectorizer()
                X_norm_train_vec = vec_norm.fit_transform(X_norm_train)
                X_norm_test_vec = vec_norm.transform(X_norm_test)

                # 3. Evaluate Original
                model_orig = clone(clf)
                model_orig.fit(X_orig_train_vec, y_train)
                f1_o = f1_score(y_test, model_orig.predict(X_orig_test_vec), average=average_method)
                fold_f1_orig.append(f1_o)

                # 4. Evaluate Normalized
                model_norm = clone(clf)
                model_norm.fit(X_norm_train_vec, y_train)
                f1_n = f1_score(y_test, model_norm.predict(X_norm_test_vec), average=average_method)
                fold_f1_norm.append(f1_n)
                
                print(f"Fold {i+1}: Orig F1={f1_o:.4f}, Norm F1={f1_n:.4f}")

            # --- Statistical Analysis ---
            mean_orig = np.mean(fold_f1_orig)
            mean_norm = np.mean(fold_f1_norm)
            std_norm = np.std(fold_f1_norm)
            
            # Wilcoxon Signed-Rank Test (Comparing the pairs of fold results)
            # Note: If all differences are zero, wilcoxon raises a ValueError.
            diffs = np.array(fold_f1_norm) - np.array(fold_f1_orig)
            if np.all(diffs == 0):
                stat, p_val = 0, 1.0
            else:
                stat, p_val = wilcoxon(fold_f1_orig, fold_f1_norm)

            mpd_results[clf_name] = {
                "Mean F1 Original": mean_orig,
                "Mean F1 Normalized": mean_norm,
                "F1 Std Dev (Norm)": std_norm,
                "MPD (Delta)": mean_norm - mean_orig,
                "Wilcoxon Stat": stat,
                "p-value": p_val,
                "Significant (p<0.05)": p_val < 0.05
            }

            print(f"Result: Delta={mpd_results[clf_name]['MPD (Delta)']:+.4f}, p={p_val:.4f}")

        return mpd_results
        
    def calculate_irs(self, batch_size=32):
        """
        Calculates the Information Retention Score (IRS).
        RSE Update: Added manual batching to prevent GPU/CPU OOM errors on large datasets.
        """
        if self.embedding_model is None:
            print("No embedding model provided. Skipping IRS calculation.")
            return None
            
        print(f"Generating embeddings using batch size {batch_size}...")
        
        # Helper function for manual batching (agnostic to embedding model's built-in batching)
        def encode_in_batches(texts):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Assuming the model has a standard .encode() method
                batch_emb = self.embedding_model.encode(batch)
                embeddings.extend(batch_emb)
            return np.array(embeddings)
            
        emb_orig = encode_in_batches(self.texts_original)
        emb_norm = encode_in_batches(self.texts_normalized)
        
        # Calculate paired cosine distances
        distances = paired_cosine_distances(emb_orig, emb_norm)
        
        # Cosine Similarity = 1 - Cosine Distance
        similarities = 1 - distances
        irs = np.mean(similarities)
        
        print(f"Information Retention Score (IRS): {irs:.4f}")
        return irs
        
    def calculate_aes(self, cr, irs):
            """
            Calculates the corrected Algorithm Effectiveness Score (AES).
            
            Formula:
                VRG = 1.0 - (1.0 / CR) if CR >= 1.0 else 0.0
                AES = (2 * IRS * VRG) / (IRS + VRG)
                
            Domain:
                Bounds both components strictly within [0, 1] to prevent 
                scale dominance during harmonic fusion.
            """
            # 1. Structural Null Defense
            if cr is None or irs is None:
                print("Warning: Calculation halted. CR or IRS contains null values.")
                return 0.0
            
            # 2. Boundary Verification & Transformation
            # If an algorithm anomalously expands vocabulary (CR < 1.0), Gain is clamped to 0.0
            if cr < 1.0:
                vrg = 0.0
            else:
                vrg = 1.0 - (1.0 / float(cr))
                
            # Ensure IRS adheres strictly to semantic probability boundaries [0, 1]
            irs_clamped = max(0.0, min(1.0, float(irs)))
            
            # 3. Dynamic Denominator Defense
            denominator = irs_clamped + vrg
            if denominator == 0.0:
                print("AES Calculation: Absolute baseline failure (IRS + VRG = 0). Returning 0.0")
                return 0.0
                
            # 4. Harmonic Mean Execution
            aes = (2.0 * irs_clamped * vrg) / denominator
            
            print(f"AES (Corrected): {aes:.4f} | Derived VRG: {vrg:.4f} | Checked IRS: {irs_clamped:.4f}")
            return aes
        
    def calculate_anld(self):
        """
        Calculates the Average Normalized Levenshtein Distance (ANLD).
        Formula: (1 / |V|) * sum( LD(w, sigma(w)) / |w| )
        """
        vocab_mapping = {}
        
        # Build a dictionary mapping each original word to its normalized form
        for orig_text, norm_text in zip(self.texts_original, self.texts_normalized):
            orig_tokens = orig_text.split()
            norm_tokens = norm_text.split()
            
            # We iterate up to the minimum length in case the user's algorithm 
            # unpredictably dropped or merged tokens, preventing index errors.
            min_len = min(len(orig_tokens), len(norm_tokens))
            for i in range(min_len):
                w = orig_tokens[i]
                sigma_w = norm_tokens[i]
                
                # Only add unique original words to the vocabulary V
                if w not in vocab_mapping:
                    vocab_mapping[w] = sigma_w
                    
        if not vocab_mapping:
            print("No valid vocabulary mapping could be created for ANLD.")
            return 0.0
            
        total_normalized_ld = 0.0
        valid_words = 0
        
        # Calculate the normalized distance for each word in the vocabulary
        for w, sigma_w in vocab_mapping.items():
            if len(w) == 0:
                continue # Prevent division by zero
                
            ld = Levenshtein.distance(w, sigma_w)
            total_normalized_ld += ld / len(w)
            valid_words += 1
            
        anld = total_normalized_ld / valid_words if valid_words > 0 else 0.0
        
        print(f"Average Normalized Levenshtein Distance (ANLD): {anld:.4f}")
        return anld
    def plot_distribution_curve(self, save_path="distribution_smooth.png", smooth_sigma=3.0):
        """
        Visualizes the shift using Unique Word Indexing and Gaussian Smoothing 
        for a highly fluid, continuous visual curve.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import Counter
        from scipy.ndimage import gaussian_filter1d
        
        # 1. Get raw word frequencies
        orig_words = [w.lower() for t in self.texts_original for w in t.split()]
        norm_words = [w.lower() for t in self.texts_normalized for w in t.split()]
        
        orig_counts = Counter(orig_words)
        norm_counts = Counter(norm_words)
        
        # 2. Assign Unique IDs to Original Words (Sorted by frequency)
        sorted_orig = [w for w, c in orig_counts.most_common()]
        word_to_id = {w: i for i, w in enumerate(sorted_orig)}
        num_orig_words = len(sorted_orig)
        
        # 3. Assign NEW IDs to OOV Words created by the normalizer
        oov_words = [w for w, c in norm_counts.most_common() if w not in word_to_id]
        for w in oov_words:
            word_to_id[w] = len(word_to_id)
            
        total_vocab_size = len(word_to_id)
        
        # 4. Map frequencies to the IDs
        x_vals = np.arange(total_vocab_size)
        y_orig = np.zeros(total_vocab_size)
        y_norm = np.zeros(total_vocab_size)
        
        for w, count in orig_counts.items():
            y_orig[word_to_id[w]] = count
            
        for w, count in norm_counts.items():
            y_norm[word_to_id[w]] = count
            
        # Log-transform to handle extreme frequency differences safely
        y_orig_log = np.log1p(y_orig)
        y_norm_log = np.log1p(y_norm)
        
        # 5. SMOOTHING: Apply Gaussian Filter to melt the discrete points into a curve
        y_orig_smooth = gaussian_filter1d(y_orig_log, sigma=smooth_sigma)
        y_norm_smooth = gaussian_filter1d(y_norm_log, sigma=smooth_sigma)
        
        # 6. Plotting
        plt.figure(figsize=(12, 6))
        
        plt.plot(x_vals, y_orig_smooth, label='Original Text', color='#1f77b4', lw=2.5)
        plt.fill_between(x_vals, y_orig_smooth, color='#1f77b4', alpha=0.3)
        
        plt.plot(x_vals, y_norm_smooth, label='Normalized Text', color='#d62728', lw=2.5)
        plt.fill_between(x_vals, y_norm_smooth, color='#d62728', alpha=0.3)
        
        plt.axvline(x=num_orig_words - 0.5, color='black', linestyle='--', lw=2, label='OOV Boundary')
        
        plt.title(f"Distribution Shift (Gaussian Smoothed, $\sigma$={smooth_sigma})", fontsize=15)
        # plt.xlabel("Unique Word ID (Left = Original Vocab, Right = New OOV Stems)", fontsize=12)
        plt.xlabel("Word", fontsize=12)
        plt.ylabel("Smoothed Log Word Frequency", fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Smoothed distribution plot saved to: {save_path}")
        return save_path
    
    def evaluate_all(self, test_size=0.2, random_state=42, verbose=False, lang="Global"):
        """
        Full scientific evaluation suite.
        """
        print(f"\n========== Evaluating Language: {lang} ==========")
        print("--- 1. Macroscopic Evaluation ---")
        cr = self.calculate_cr()
        print("\n--- 2. Semantic Preservation ---")
        irs = self.calculate_irs()
        print("\n--- 3. Overall Effectiveness ---")
        aes = self.calculate_aes(cr, irs)
        print("\n--- 4. Micro-level Fidelity (Safety Gate) ---")
        anld = self.calculate_anld()
        print("\n--- 5. Downstream Impact ---")
        mpd = self.calculate_mpd(random_state=random_state)
        print("\n--- 7. Distribution Shift ---")
        self.plot_distribution_curve(save_path=f"dist_shift_seed{random_state}.png")
        
        return {
            "CR": cr, "IRS": irs, "AES": aes, "ANLD": anld, "MPD": mpd
        }
