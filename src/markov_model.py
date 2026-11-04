"""
markov_model.py
---------------
Implements the Markov Chain component of the hybrid framework.
  - First-order Markov Chain (memoryless assumption, Equation 3.1)
  - State space: computer × privilege × auth_outcome × access_pattern
  - Theoretical max: 672 states (28 computers × 4 privilege × 3 auth × 2 access)
  - Transition matrix built from training sequences (Equation 3.2)
  - Prediction: highest probability next state → class label
  - Latency target: ~0.23ms per prediction (Table 4.15)

Zero Trust concept:
  The Markov Chain models attacker state transitions inside a Zero Trust
  network. Each state represents where an attacker is and what privilege
  they hold. The transition matrix encodes how likely they are to move
  to the next state — directly interpretable by a security analyst.
"""

import numpy as np
import pandas as pd
import pickle
import time
import os
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# STATE SPACE DEFINITION
# ════════════════════════════════════════════════════════════

# Thesis Section 3.6.1: state combines 4 attributes
# 28 computers × 4 privilege levels × 3 auth outcomes × 2 access patterns
# = 672 theoretical states

PRIVILEGE_LEVELS  = [0, 1, 2, 3]       # standard→domain_admin
AUTH_OUTCOMES     = [0, 1, 2]          # normal, suspicious, malicious
ACCESS_PATTERNS   = [0, 1]             # normal, lateral

def state_to_label(state):
    """
    Map a state integer back to a class label (0, 1, or 2).
    Uses the auth_fail component of the encoded state.
    State encoding: priv*1000 + auth_fail*100 + denied*20 + off_hours*10 + consec*2 + high_val
    auth_fail bucket: 0=normal, 1-2=suspicious, 3-4=malicious
    """
    auth_fail = (state % 1000) // 100
    if auth_fail == 0:
        return 0   # Normal
    elif auth_fail <= 2:
        return 1   # Suspicious
    else:
        return 2   # Malicious

def encode_state(row):
    """
    Encode a single event row into a discrete state integer.
    Uses more features to create a richer state space that can
    distinguish Normal, Suspicious, and Malicious events.
    """
    priv   = int(row.get("privilege_level", 0)) % 4

    # Use policy violation features as state components
    auth_fail = min(int(row.get("Authentication_Failure_Count", 0)), 4)
    denied    = min(int(row.get("Denied_Access_Count", 0)), 4)
    off_hours = int(row.get("Is_Off_Hours", 0)) % 2
    consec    = min(int(row.get("Consecutive_Failures", 0)), 3)
    high_val  = int(row.get("Source_High_Value_Target", 0)) % 2

    # Encode as single integer
    state = (priv * 1000 +
             auth_fail * 100 +
             denied * 20 +
             off_hours * 10 +
             consec * 2 +
             high_val)
    return state


# ════════════════════════════════════════════════════════════
# MARKOV CHAIN CLASS
# ════════════════════════════════════════════════════════════

class MarkovChainModel:
    """
    First-order Markov Chain for lateral movement prediction.

    Attributes:
        transition_counts  : raw count matrix C[i][j]
        transition_matrix  : normalised probability matrix P[i][j]
        n_states           : number of distinct states observed
        state_labels       : maps state index to class label
    """

    def __init__(self):
        self.transition_counts  = {}
        self.transition_matrix  = {}
        self.n_states           = 0
        self.all_states         = set()
        self.is_fitted          = False

    # ── BUILD TRANSITION MATRIX ──────────────────────────────

    def fit(self, X_train, y_train):
        """
        Build the transition probability matrix from training sequences.

        Implements Equation 3.2 (thesis):
          P[i][j] = C[i][j] / sum_k(C[i][k])

        Where C[i][j] is the count of transitions from state i to state j
        observed in the training data.

        Parameters:
            X_train : DataFrame of preprocessed training features
            y_train : Series of class labels (0, 1, 2)
        """
        print("=" * 60)
        print("MARKOV CHAIN MODEL — TRAINING")
        print("=" * 60)

        # Combine features only - Label is NOT used in state encoding
        # to prevent data leakage. Label is only used for the
        # state-to-class mapping after prediction.
        train_df           = X_train.copy()
        train_df["Label"]  = y_train.values
        self.state_classes = {}   # maps state → most common class

        # Sort by any available temporal feature
        time_col = None
        for col in ["SystemTime_hour", "SystemTime_minute",
                    "Authentication_Failure_Count"]:
            if col in train_df.columns:
                time_col = col
                break

        if time_col:
            train_df = train_df.sort_values(time_col)

        # Build transition counts
        # Also record which class is most common for each state
        states      = []
        class_votes = {}   # state → {0:count, 1:count, 2:count}

        for _, row in train_df.iterrows():
            state = encode_state(row)
            states.append(state)
            self.all_states.add(state)
            lbl = int(row["Label"])
            if state not in class_votes:
                class_votes[state] = {0: 0, 1: 0, 2: 0}
            class_votes[state][lbl] += 1

        # Store majority class per state for prediction
        self.state_classes = {
            s: max(votes, key=votes.get)
            for s, votes in class_votes.items()
        }

        # Count transitions: state[t] → state[t+1]
                # Count transitions: state[t] → state[t+1]
        for i in range(len(states) - 1):
            s_from = states[i]
            s_to   = states[i + 1]
            if s_from not in self.transition_counts:
                self.transition_counts[s_from] = {}
            if s_to not in self.transition_counts[s_from]:
                self.transition_counts[s_from][s_to] = 0
            self.transition_counts[s_from][s_to] += 1

        # Normalise rows → probability matrix (Equation 3.2)
        for s_from, transitions in self.transition_counts.items():
            total = sum(transitions.values())
            self.transition_matrix[s_from] = {
                s_to: count / total
                for s_to, count in transitions.items()
            }

        self.n_states  = len(self.all_states)
        self.is_fitted = True

        print(f"\n  States observed in training : {self.n_states}")
        print(f"  Theoretical max states      : 672")
        print(f"  Transitions recorded        : "
              f"{sum(len(v) for v in self.transition_matrix.values())}")
        self._print_transition_example()
        return self

    def _print_transition_example(self):
        """
        Print a worked example of the transition matrix
        (matches Table 3.3 in thesis).
        """
        print("\n  Sample transition probabilities (Table 3.3 style):")
        print(f"  {'From State':>12} → {'To State':>10} | Probability")
        print(f"  {'-'*45}")

        count = 0
        for s_from, transitions in self.transition_matrix.items():
            if count >= 5:
                break
            for s_to, prob in sorted(transitions.items(),
                                     key=lambda x: -x[1])[:2]:
                label_from = state_to_label(s_from)
                label_to   = state_to_label(s_to)
                print(f"  State {s_from:>5} (cls {label_from}) → "
                      f"State {s_to:>5} (cls {label_to}) | {prob:.4f}")
            count += 1

    # ── PREDICT ──────────────────────────────────────────────

    def predict_proba(self, X):
        """
        Return probability estimates for each class [0, 1, 2].
        Uses transition matrix + state-class mapping trained on labels.
        """
        probs = []
        for _, row in X.iterrows():
            state      = encode_state(row)
            class_prob = np.zeros(3)

            if state in self.transition_matrix:
                for s_to, prob in self.transition_matrix[state].items():
                    # Map destination state to its majority class
                    label = self.state_classes.get(s_to, 0)
                    class_prob[label] += prob
            else:
                # Unseen state: use prior distribution
                class_prob = np.array([0.964, 0.025, 0.011])

            # Normalise
            total = class_prob.sum()
            if total > 0:
                class_prob /= total

            probs.append(class_prob)

        return np.array(probs)

    def predict(self, X):
        """Return predicted class labels (argmax of probabilities)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── EXPLAINABILITY ───────────────────────────────────────

    def explain_prediction(self, row, top_n=3):
        """
        Explain why the model made a prediction for a single event.
        This is the interpretability advantage of the Markov Chain
        over the LSTM (thesis Section 4.7.1.1).

        Returns a human-readable string showing which transitions fired.
        """
        state      = encode_state(row)
        class_prob = np.zeros(3)
        transitions_fired = []

        if state in self.transition_matrix:
            for s_to, prob in sorted(
                self.transition_matrix[state].items(),
                key=lambda x: -x[1]
            )[:top_n]:
                label = state_to_label(s_to)
                class_prob[label] += prob
                transitions_fired.append((state, s_to, prob, label))

        predicted_class = np.argmax(class_prob)
        label_names     = {0: "Normal", 1: "Suspicious", 2: "Malicious"}

        explanation = [
            f"  Current state  : {state}",
            f"  Prediction     : {label_names[predicted_class]} "
            f"(class {predicted_class})",
            f"  Top transitions:"
        ]
        for s_from, s_to, prob, lbl in transitions_fired:
            explanation.append(
                f"    State {s_from} → State {s_to} "
                f"(prob={prob:.4f}, class={label_names[lbl]})"
            )
        return "\n".join(explanation)

    # ── LATENCY MEASUREMENT ──────────────────────────────────

    def measure_latency(self, X, n_runs=1000):
        """
        Measure average prediction latency per sample in milliseconds.
        Target: ~0.23ms (thesis Table 4.15).
        Averaged over n_runs predictions on a single sample.
        """
        sample = X.iloc[[0]]
        times  = []

        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(sample)
            end   = time.perf_counter()
            times.append((end - start) * 1000)   # convert to ms

        avg_ms = np.mean(times)
        print(f"\n  Latency (avg over {n_runs} runs): {avg_ms:.4f} ms")
        return avg_ms

    # ── SAVE / LOAD ──────────────────────────────────────────

    def save(self, path="models/markov_model.pkl"):
        """Save the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"\n  ✓ Markov model saved to: {path}")

    @staticmethod
    def load(path="models/markov_model.pkl"):
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ════════════════════════════════════════════════════════════
# TRAINING + EVALUATION RUNNER
# ════════════════════════════════════════════════════════════

def train_and_evaluate():
    """
    Train the Markov Chain on preprocessed data and evaluate on test set.
    Prints all metrics that will appear in Table 4.9 of the thesis.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score,
                                 confusion_matrix, roc_auc_score)

    # ── Load preprocessed data ──
    print("\nLoading preprocessed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/processed/X_val.csv")
    y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    # ── Train ──
    model = MarkovChainModel()
    model.fit(X_train, y_train)

    # ── Validation accuracy ──
    val_preds = model.predict(X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    print(f"\n  Validation accuracy: {val_acc:.4f}")

    # ── Test evaluation ──
    print("\n" + "=" * 60)
    print("MARKOV CHAIN — TEST SET EVALUATION")
    print("=" * 60)

    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds,
                           average="weighted", zero_division=0)
    rec  = recall_score(y_test, test_preds,
                        average="weighted", zero_division=0)
    f1   = f1_score(y_test, test_preds,
                    average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, test_preds,
                        average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_test, test_proba,
                            multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_test, test_preds)

    # False positive rate (mean across classes)
    fpr_list = []
    for cls in range(3):
        tn = cm.sum() - cm[cls, :].sum() - cm[:, cls].sum() + cm[cls, cls]
        fp = cm[:, cls].sum() - cm[cls, cls]
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    mean_fpr = np.mean(fpr_list)

    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1-score        : {f1:.4f}")
    print(f"  Macro F1        : {f1_macro:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Mean FPR        : {mean_fpr:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':15} Pred:Normal  Pred:Susp  Pred:Malic")
    labels = ["Normal", "Suspicious", "Malicious"]
    for i, row in enumerate(cm):
        print(f"  Actual:{labels[i]:10} {row[0]:11d}  {row[1]:9d}  {row[2]:9d}")

    # ── Latency ──
    latency = model.measure_latency(X_test)

    # ── Explainability example ──
    print("\n  Sample prediction explanation (interpretability demo):")
    print(model.explain_prediction(X_test.iloc[0]))

    # ── Save model and results ──
    model.save()

    os.makedirs("results/tables", exist_ok=True)
    results = {
        "model":      "Markov Chain",
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "macro_f1":   round(f1_macro, 4),
        "auc_roc":    round(auc,  4),
        "mean_fpr":   round(mean_fpr, 4),
        "latency_ms": round(latency,  4),
    }
    pd.DataFrame([results]).to_csv(
        "results/tables/markov_results.csv", index=False
    )
    print(f"\n  ✓ Results saved to: results/tables/markov_results.csv")

    return model, results


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, results = train_and_evaluate()