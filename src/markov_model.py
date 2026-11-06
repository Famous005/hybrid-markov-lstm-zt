"""
markov_model.py
---------------
Markov Chain component - models state transitions per class.

Thesis Sections 3.6.1 and 4.3.3:
  - State space built from policy violation feature patterns
  - Transition matrix P[i][j] = C[i][j] / sum_k(C[i][k])
  - Prediction via most probable next state class
  - Latency target: ~0.23ms (Table 4.15)
"""

import numpy as np
import pandas as pd
import pickle, time, os
from collections import defaultdict


SEED = 42
np.random.seed(SEED)


def encode_state(row):
    """
    Encode a row into a discrete state using multiple features.
    Real dataset features are pre-scaled so we bin them.
    """
    def safe_float(val):
        try:
            return float(val)
        except:
            return 0.0

    # Authentication failure — 5 bins
    auth  = safe_float(row.get("Authentication_Failure_Count", 0))
    a_bin = min(int(auth * 10), 4)

    # Destination criticality — 4 bins
    crit  = safe_float(row.get("Destination_Criticality", 0))
    c_bin = min(int(crit * 4), 3)

    # Is off hours — binary
    off   = safe_float(row.get("Is_Off_Hours", 0))
    o_bin = 1 if off > 0 else 0

    # Source high value target — binary
    hvt   = safe_float(row.get("Source_High_Value_Target", 0))
    h_bin = 1 if hvt > 0 else 0

    # EventID_3.0 (network connection) — binary
    e3    = safe_float(row.get("EventID_3.0", 0))
    e_bin = 1 if e3 > 0.5 else 0

    state = (a_bin * 10000 +
             c_bin * 1000 +
             o_bin * 100 +
             h_bin * 10 +
             e_bin)
    return state


class MarkovChainModel:

    def __init__(self):
        self.transition_matrix = {}
        self.state_classes     = {}
        self.class_priors      = np.array([1/3, 1/3, 1/3])
        self.all_states        = set()
        self.is_fitted         = False

    def fit(self, X_train, y_train):
        print("=" * 60)
        print("MARKOV CHAIN MODEL — TRAINING")
        print("=" * 60)

        df = X_train.copy()
        df["Label"] = y_train.values

        # Build class priors from training labels
        counts = df["Label"].value_counts().sort_index()
        total  = counts.sum()
        self.class_priors = np.array([
            counts.get(0, 0) / total,
            counts.get(1, 0) / total,
            counts.get(2, 0) / total,
        ])

        # Build state sequences
        states      = []
        class_votes = {}

        for _, row in df.iterrows():
            state = encode_state(row)
            lbl   = int(row["Label"])
            states.append((state, lbl))
            self.all_states.add(state)
            if state not in class_votes:
                class_votes[state] = {0: 0, 1: 0, 2: 0}
            class_votes[state][lbl] += 1

        # State → majority class
        self.state_classes = {
            s: max(v, key=v.get)
            for s, v in class_votes.items()
        }

        # State → class probability distribution
        self.state_class_proba = {}
        for s, v in class_votes.items():
            total_v = sum(v.values())
            self.state_class_proba[s] = np.array([
                v[0]/total_v, v[1]/total_v, v[2]/total_v
            ])

        # Transition counts
        tc = {}
        for i in range(len(states) - 1):
            s_from = states[i][0]
            s_to   = states[i+1][0]
            if s_from not in tc:
                tc[s_from] = {}
            tc[s_from][s_to] = tc[s_from].get(s_to, 0) + 1

        # Normalise → probability matrix
        self.transition_matrix = {}
        for s_from, trans in tc.items():
            total_t = sum(trans.values())
            self.transition_matrix[s_from] = {
                s_to: cnt/total_t
                for s_to, cnt in trans.items()
            }

        self.is_fitted = True

        print(f"\n  States observed   : {len(self.all_states)}")
        print(f"  Theoretical max   : 672")
        print(f"  Transitions       : "
              f"{sum(len(v) for v in self.transition_matrix.values())}")
        print(f"  Class priors      : "
              f"Normal={self.class_priors[0]:.3f} "
              f"Susp={self.class_priors[1]:.3f} "
              f"Malic={self.class_priors[2]:.3f}")

        self._print_sample()
        return self

    def _print_sample(self):
        print("\n  Sample transitions:")
        print(f"  {'From':>8} → {'To':>8} | Prob")
        print(f"  {'-'*35}")
        count = 0
        for s_from, trans in self.transition_matrix.items():
            if count >= 5:
                break
            for s_to, prob in sorted(trans.items(),
                                     key=lambda x: -x[1])[:2]:
                c_from = self.state_classes.get(s_from, 0)
                c_to   = self.state_classes.get(s_to, 0)
                names  = {0:"Normal", 1:"Suspicious", 2:"Malicious"}
                print(f"  State {s_from:>5} ({names[c_from][:4]}) → "
                      f"State {s_to:>5} ({names[c_to][:4]}) | {prob:.4f}")
            count += 1

    def predict_proba(self, X):
        """
        For each row:
        1. Encode current state
        2. Look up state's class probability distribution
        3. If state unseen → use class priors
        4. Blend with transition matrix for next-state signal
        """
        probs = []
        for _, row in X.iterrows():
            state = encode_state(row)

            if state in self.state_class_proba:
                # Use empirical class distribution for this state
                class_prob = self.state_class_proba[state].copy()

                # Blend with transition signal if available
                if state in self.transition_matrix:
                    trans_prob = np.zeros(3)
                    for s_to, p in self.transition_matrix[state].items():
                        c_to = self.state_classes.get(s_to, 0)
                        trans_prob[c_to] += p
                    # 70% state distribution, 30% transition signal
                    class_prob = 0.7 * class_prob + 0.3 * trans_prob
            else:
                # Unseen state → use class priors
                class_prob = self.class_priors.copy()

            total = class_prob.sum()
            if total > 0:
                class_prob /= total
            probs.append(class_prob)

        return np.array(probs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def explain_prediction(self, row, top_n=3):
        state      = encode_state(row)
        class_prob = self.predict_proba(pd.DataFrame([row]))[0]
        predicted  = np.argmax(class_prob)
        names      = {0:"Normal", 1:"Suspicious", 2:"Malicious"}
        lines = [
            f"  State          : {state}",
            f"  Prediction     : {names[predicted]} (class {predicted})",
            f"  Probabilities  : Normal={class_prob[0]:.4f} "
            f"Susp={class_prob[1]:.4f} Malic={class_prob[2]:.4f}",
        ]
        if state in self.transition_matrix:
            lines.append("  Top transitions:")
            for s_to, p in sorted(
                self.transition_matrix[state].items(),
                key=lambda x: -x[1]
            )[:top_n]:
                c = self.state_classes.get(s_to, 0)
                lines.append(f"    → State {s_to} "
                             f"({names[c]}) prob={p:.4f}")
        return "\n".join(lines)

    def measure_latency(self, X, n_runs=1000):
        sample = X.iloc[[0]]
        times  = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(sample)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        print(f"\n  Latency (avg {n_runs} runs): {avg:.4f} ms")
        return avg

    def save(self, path="models/markov_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"\n  ✓ Markov saved to: {path}")

    @staticmethod
    def load(path="models/markov_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


def train_and_evaluate():
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 confusion_matrix, roc_auc_score)

    print("\nLoading preprocessed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/processed/X_val.csv")
    y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    model = MarkovChainModel()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    from sklearn.metrics import accuracy_score
    print(f"\n  Validation accuracy: "
          f"{accuracy_score(y_val, val_preds):.4f}")

    print("\n" + "=" * 60)
    print("MARKOV — TEST SET EVALUATION")
    print("=" * 60)

    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    acc      = accuracy_score(y_test, test_preds)
    prec     = precision_score(y_test, test_preds,
                               average="weighted", zero_division=0)
    rec      = recall_score(y_test, test_preds,
                            average="weighted", zero_division=0)
    f1       = f1_score(y_test, test_preds,
                        average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, test_preds,
                        average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_test, test_proba,
                            multi_class="ovr", average="weighted")
    except:
        auc = 0.0

    cm = confusion_matrix(y_test, test_preds, labels=[0,1,2])
    fpr_list = []
    for cls in range(3):
        tn = cm.sum()-cm[cls,:].sum()-cm[:,cls].sum()+cm[cls,cls]
        fp = cm[:,cls].sum()-cm[cls,cls]
        fpr_list.append(fp/(fp+tn) if (fp+tn)>0 else 0)
    mean_fpr = np.mean(fpr_list)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  Macro F1  : {f1_macro:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Mean FPR  : {mean_fpr:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':15} Pred:Normal  Pred:Susp  Pred:Malic")
    names = ["Normal","Suspicious","Malicious"]
    for i, row in enumerate(cm):
        print(f"  Actual:{names[i]:10} "
              f"{row[0]:11d}  {row[1]:9d}  {row[2]:9d}")

    latency = model.measure_latency(X_test)

    print("\n  Explanation example:")
    print(model.explain_prediction(X_test.iloc[0]))

    model.save()
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame([{
        "model": "Markov Chain",
        "accuracy":   round(acc,4),
        "precision":  round(prec,4),
        "recall":     round(rec,4),
        "f1_score":   round(f1,4),
        "macro_f1":   round(f1_macro,4),
        "auc_roc":    round(auc,4),
        "mean_fpr":   round(mean_fpr,4),
        "latency_ms": round(latency,4),
    }]).to_csv("results/tables/markov_results.csv", index=False)
    print(f"\n  ✓ Results saved")
    return model


if __name__ == "__main__":
    train_and_evaluate()