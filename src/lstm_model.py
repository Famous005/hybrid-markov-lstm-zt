"""
lstm_model.py
-------------
LSTM-based threat classifier for lateral movement prediction.
Architecture matches thesis Table 4.6:
  LSTM layer 1 : 128 units, tanh, dropout 0.2
  LSTM layer 2 :  64 units, tanh, dropout 0.2  
  Dense layer 1:  32 units, ReLU, dropout 0.1
  Dense layer 2:   3 units, Softmax

Each event is treated as a sequence of 1 timestep with all 28 features,
allowing the LSTM gates to learn temporal patterns within the feature
vector itself rather than across events (which fails with 96% class
imbalance in sequential ordering).

Zero Trust concept:
  The LSTM captures feature interactions across the policy violation
  artifact dimensions — authentication failures, denied access counts,
  temporal patterns — that indicate lateral movement progression.
"""

import numpy as np
import pandas as pd
import pickle, time, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SEQUENCE_LENGTH = 1       # each row is 1 timestep — avoids ordering problem
BATCH_SIZE      = 32
EPOCHS          = 100
LEARNING_RATE   = 0.001
L2_REG          = 0.001
N_CLASSES       = 3


# ════════════════════════════════════════════════════════════
# PREPARE INPUT
# ════════════════════════════════════════════════════════════

def prepare_input(X):
    """
    Reshape (n_samples, n_features) → (n_samples, 1, n_features)
    so each row becomes a sequence of length 1 for the LSTM.
    This preserves the LSTM architecture while ensuring every
    sample gets its own independent prediction.
    """
    return X.values.astype(np.float32).reshape(-1, 1, X.shape[1])


# ════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — Table 4.6 (thesis)
# ════════════════════════════════════════════════════════════

def build_lstm_model(n_features):
    """
    Stacked LSTM architecture from thesis Table 4.6.
    Input shape: (1, n_features) — one timestep per sample.
    """
    model = Sequential([
        LSTM(128, activation="tanh",
             return_sequences=True,
             kernel_regularizer=l2(L2_REG),
             input_shape=(1, n_features),
             name="lstm_1"),
        Dropout(0.2, name="dropout_1"),

        LSTM(64, activation="tanh",
             return_sequences=False,
             kernel_regularizer=l2(L2_REG),
             name="lstm_2"),
        Dropout(0.2, name="dropout_2"),

        Dense(32, activation="relu",
              kernel_regularizer=l2(L2_REG),
              name="dense_1"),
        Dropout(0.1, name="dropout_3"),

        Dense(N_CLASSES, activation="softmax", name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ════════════════════════════════════════════════════════════
# LSTM MODEL CLASS
# ════════════════════════════════════════════════════════════

class LSTMModel:

    def __init__(self):
        self.model      = None
        self.n_features = None
        self.history    = None
        self.is_fitted  = False

    def fit(self, X_train, y_train, X_val, y_val):
        print("=" * 60)
        print("LSTM MODEL — TRAINING")
        print("=" * 60)

        self.n_features = X_train.shape[1]

        # Add small gaussian noise to training to prevent perfect memorisation
        X_train_noisy = X_train.copy()
        noise = np.random.RandomState(SEED).normal(0, 0.01, X_train.shape)
        X_train_noisy = pd.DataFrame(
            X_train.values + noise, columns=X_train.columns
        )

        # Reshape to (n, 1, features)
        X_tr = prepare_input(X_train_noisy)
        X_vl = prepare_input(X_val)
        y_tr = y_train.values.astype(np.int32)
        y_vl = y_val.values.astype(np.int32)

        # Subsample Normal class to 3x minority — reduces dominance
        # while keeping natural proportions
        idx_normal = np.where(y_tr == 0)[0]
        idx_susp   = np.where(y_tr == 1)[0]
        idx_mal    = np.where(y_tr == 2)[0]
        target_n   = len(idx_mal) * 3
        target_n = min(target_n, len(idx_normal))
        idx_normal_sub = np.random.choice(idx_normal, size=target_n,
                                          replace=False)
        idx_keep = np.concatenate([idx_normal_sub, idx_susp, idx_mal])
        idx_keep = np.random.permutation(idx_keep)
        X_tr     = X_tr[idx_keep]
        y_tr     = y_tr[idx_keep]
        print(f"  Subsampled train: {X_tr.shape} | "
              f"classes: {np.bincount(y_tr)}")

        print(f"\n  Train shape : {X_tr.shape} | "
              f"classes: {np.bincount(y_tr)}")
        print(f"  Val shape   : {X_vl.shape} | "
              f"classes: {np.bincount(y_vl)}")

        y_tr_cat = to_categorical(y_tr, N_CLASSES)
        y_vl_cat = to_categorical(y_vl, N_CLASSES)

        self.model = build_lstm_model(self.n_features)
        self.model.summary()

        # Manually tuned class weights for real dataset
        # Suspicious gets moderate boost, not extreme
        class_weight = {0: 1.0, 1: 8.0, 2: 0.5}

        print(f"\n  Class weights: { {k: round(v,2) for k,v in class_weight.items()} }")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True,
                          mode="min", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-6, verbose=1),
        ]

        print(f"\n  Training... (up to {EPOCHS} epochs)")
        self.history = self.model.fit(
            X_tr, y_tr_cat,
            validation_data=(X_vl, y_vl_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        self.is_fitted = True
        best_val = max(self.history.history["val_accuracy"])
        best_ep  = np.argmax(
            self.history.history["val_accuracy"]) + 1
        print(f"\n  Best epoch   : {best_ep}")
        print(f"  Best val acc : {best_val:.4f}")
        return self

    def predict_proba(self, X):
        """Return softmax probabilities for all 3 classes."""
        X_arr = prepare_input(X)
        return self.model.predict(X_arr, verbose=0, batch_size=256)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def measure_latency(self, X, n_runs=1000):
        sample = X.iloc[[0]]
        _ = self.predict(sample)   # warm up
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(sample)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        print(f"\n  Latency (avg {n_runs} runs): {avg:.4f} ms")
        return avg

    def save(self, path="models/lstm_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        with open("models/lstm_meta.pkl", "wb") as f:
            pickle.dump({"n_features": self.n_features,
                         "is_fitted":  self.is_fitted}, f)
        print(f"\n  ✓ LSTM saved to: {path}")

    @staticmethod
    def load(path="models/lstm_model.keras"):
        from tensorflow.keras.models import load_model
        w = LSTMModel()
        w.model = load_model(path)
        with open("models/lstm_meta.pkl", "rb") as f:
            m = pickle.load(f)
        w.n_features = m["n_features"]
        w.is_fitted  = True
        return w


# ════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════

def train_and_evaluate():
    print("\nLoading preprocessed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_val   = pd.read_csv("data/processed/X_val.csv")
    y_val   = pd.read_csv("data/processed/y_val.csv").squeeze()
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

    model = LSTMModel()
    model.fit(X_train, y_train, X_val, y_val)

    print("\n" + "=" * 60)
    print("LSTM — TEST SET EVALUATION")
    print("=" * 60)

    test_preds = model.predict(X_test)
    test_proba = model.predict_proba(X_test)

    y_al     = y_test.values
    preds_al = test_preds
    proba_al = test_proba

    acc      = accuracy_score(y_al, preds_al)
    prec     = precision_score(y_al, preds_al,
                               average="weighted", zero_division=0)
    rec      = recall_score(y_al, preds_al,
                            average="weighted", zero_division=0)
    f1       = f1_score(y_al, preds_al,
                        average="weighted", zero_division=0)
    f1_macro = f1_score(y_al, preds_al,
                        average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_al, proba_al,
                            multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_al, preds_al, labels=[0, 1, 2])

    fpr_list = []
    for cls in range(3):
        tn = cm.sum() - cm[cls,:].sum() - cm[:,cls].sum() + cm[cls,cls]
        fp = cm[:,cls].sum() - cm[cls,cls]
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
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
    for i, row in enumerate(cm):
        lbl = ["Normal", "Suspicious", "Malicious"][i]
        print(f"  Actual:{lbl:10} "
              f"{row[0]:11d}  {row[1]:9d}  {row[2]:9d}")

    latency = model.measure_latency(X_test)
    model.save()

    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame([{
        "model":      "LSTM",
        "accuracy":   round(acc, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
        "f1_score":   round(f1, 4),
        "macro_f1":   round(f1_macro, 4),
        "auc_roc":    round(auc, 4),
        "mean_fpr":   round(mean_fpr, 4),
        "latency_ms": round(latency, 4),
    }]).to_csv("results/tables/lstm_results.csv", index=False)
    print(f"\n  ✓ Results saved to: results/tables/lstm_results.csv")

    return model, {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_score": f1, "macro_f1": f1_macro, "auc_roc": auc,
        "mean_fpr": mean_fpr, "latency_ms": latency
    }


if __name__ == "__main__":
    train_and_evaluate()