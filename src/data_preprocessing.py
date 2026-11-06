"""
data_preprocessing.py
----------------------
Preprocessing pipeline for the REAL Zero Trust dataset.
Dataset: 35,993 events, 59 columns, 3 classes (0=Normal, 1=Suspicious, 2=Malicious)

Pipeline stages (matches Table 4.4):
  Stage 1: Load real dataset
  Stage 2: Drop index columns   → 54 features
  Stage 3: Missing value treat  → clean
  Stage 4: Feature engineering  → add 9 derived features
  Stage 5: Correlation pruning  → 28 final features

Split: 70/15/15 stratified
SMOTE: training set only
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import os, pickle

SEED = 42
np.random.seed(SEED)


def load_data(filepath="data/real_dataset.csv"):
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE — REAL DATASET")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"\n[Stage 1] Dataset loaded")
    print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
    dist = df["Label"].value_counts().sort_index()
    names = {0:"Normal", 1:"Suspicious", 2:"Malicious"}
    for lbl, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {names[lbl]:12s}: {count:6d} ({pct:.1f}%)")
    return df


def drop_index_columns(df):
    index_cols = [c for c in df.columns
                  if "Unnamed" in c or c == "EventRecordID"]
    df = df.drop(columns=index_cols, errors="ignore")
    print(f"\n[Stage 2] Dropped index columns: {len(index_cols)}")
    print(f"  Shape now: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.50].index.tolist()
    df = df.drop(columns=high_missing, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in num_cols:
        num_cols.remove("Label")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(
            strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(
            strategy="most_frequent").fit_transform(df[cat_cols])

    print(f"\n[Stage 3] Missing value treatment complete")
    print(f"  Columns dropped (>50% missing): {len(high_missing)}")
    print(f"  Shape now: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def engineer_features(df):
    """
    Add 9 derived features that the thesis documents as the
    strongest predictors (Table 4.16).
    """
    df = df.copy()

    # Parse SystemTime if present
    if "SystemTime" in df.columns:
        df["SystemTime"] = pd.to_datetime(df["SystemTime"], errors="coerce")
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df = df.drop(columns=["SystemTime"], errors="ignore")

    # Is_Off_Hours
    if "SystemTime_hour" in df.columns:
        df["Is_Off_Hours"] = (
            (df["SystemTime_hour"] < 0.375) |   # before 9AM (9/24=0.375)
            (df["SystemTime_hour"] >= 0.708)     # after 5PM (17/24=0.708)
        ).astype(int)
    else:
        df["Is_Off_Hours"] = 0

    # Authentication_Failure_Count from EventID_5.0
    if "EventID_5.0" in df.columns:
        df["Authentication_Failure_Count"] = df["EventID_5.0"].astype(float)
    else:
        df["Authentication_Failure_Count"] = 0

    # Denied_Access_Count from EventID_7.0
    if "EventID_7.0" in df.columns:
        df["Denied_Access_Count"] = df["EventID_7.0"].astype(float)
    else:
        df["Denied_Access_Count"] = 0

    # Consecutive_Failures — proxy from EventID_5.0
    df["Consecutive_Failures"] = df["Authentication_Failure_Count"]

    # Failed_Login_Sequence_Length
    df["Failed_Login_Sequence_Length"] = df["Authentication_Failure_Count"]

    # Privilege_Escalation_Attempt from EventID_11.0
    if "EventID_11.0" in df.columns:
        df["Privilege_Escalation_Attempt"] = df["EventID_11.0"].astype(int)
    else:
        df["Privilege_Escalation_Attempt"] = 0

    # Time_Since_Last_Success
    df["Time_Since_Last_Success"] = df.get("SystemTime_minute",
                                            pd.Series(np.zeros(len(df))))

    # Destination_Criticality
    port_scores = {
        "DestinationPortName_ldap": 5,
        "DestinationPortName_kerberos": 5,
        "DestinationPortName_microsoft-ds": 4,
        "DestinationPortName_ms-wbt-server": 4,
        "DestinationPortName_epmap": 3,
        "DestinationPortName_http": 2,
        "DestinationPortName_https": 2,
        "DestinationPortName_domain": 1,
    }
    df["Destination_Criticality"] = 0
    for col, score in port_scores.items():
        if col in df.columns:
            df["Destination_Criticality"] += df[col] * score

    # Source_High_Value_Target
    if "Initiated_True" in df.columns:
        df["Source_High_Value_Target"] = df["Initiated_True"].astype(int)
    else:
        df["Source_High_Value_Target"] = 0

    print(f"\n[Stage 4] Feature engineering complete")
    print(f"  Shape now: {df.shape[0]} rows × {df.shape[1]} columns")

    # Verify separation
    for feat in ["Authentication_Failure_Count",
                 "Denied_Access_Count", "Is_Off_Hours"]:
        if feat in df.columns:
            means = df.groupby("Label")[feat].mean()
            vals  = "  |  ".join([f"cls{l}={m:.4f}"
                                   for l, m in means.items()])
            print(f"  {feat}: {vals}")
    return df


def encode_and_prune(df):
    df = df.copy()
    drop_cols = ["SystemTime_day"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categoricals
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if "Label" in cat_cols:
        cat_cols.remove("Label")
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    target   = df["Label"]
    features = df.drop(columns=["Label"])

    # Drop highly correlated features
    corr     = features.corr().abs()
    upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
    features = features.drop(columns=to_drop, errors="ignore")

    # Keep top 28 features by correlation with target
    if len(features.columns) > 28:
        correlations = features.corrwith(target).abs().sort_values(
            ascending=False)
        keep     = correlations.head(28).index.tolist()
        features = features[keep]

    df = pd.concat([features, target], axis=1)
    print(f"\n[Stage 5] Encoding and pruning complete")
    print(f"  Dropped (high correlation): {len(to_drop)}")
    print(f"  Final features: {len(features.columns)}")
    print(f"  Shape now: {df.shape[0]} rows × {df.shape[1]} columns")
    return df, le_dict


def split_and_balance(df):
    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    print(f"\n[Split] Stratified 70/15/15:")
    print(f"  Train: {len(X_train):6d}  Val: {len(X_val):6d}  "
          f"Test: {len(X_test):6d}")

    scaler  = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns)
    X_val   = pd.DataFrame(scaler.transform(X_val),
                           columns=X_val.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns)

    # SMOTE on training only
    smote = SMOTE(random_state=SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"\n[SMOTE] Training after balancing:")
    names = {0:"Normal", 1:"Suspicious", 2:"Malicious"}
    for lbl, count in pd.Series(y_train_bal).value_counts().sort_index().items():
        pct = count / len(y_train_bal) * 100
        print(f"  {names[lbl]:12s}: {count:6d} ({pct:.1f}%)")

    print(f"\n[Val] Natural distribution:")
    for lbl, count in y_val.value_counts().sort_index().items():
        pct = count / len(y_val) * 100
        print(f"  {names[lbl]:12s}: {count:6d} ({pct:.1f}%)")

    return (X_train_bal, y_train_bal,
            X_val, y_val,
            X_test, y_test,
            scaler)


def save_artifacts(X_train, y_train, X_val, y_val,
                   X_test, y_test, scaler, le_dict):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_val.to_csv(  "data/processed/X_val.csv",   index=False)
    X_test.to_csv( "data/processed/X_test.csv",  index=False)
    pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.Series(y_val).to_csv(  "data/processed/y_val.csv",   index=False)
    pd.Series(y_test).to_csv( "data/processed/y_test.csv",  index=False)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(le_dict, f)
    print(f"\n✓ Saved to data/processed/ and models/")


def run_preprocessing(filepath="data/real_dataset.csv"):
    df                        = load_data(filepath)
    df                        = drop_index_columns(df)
    df                        = handle_missing_values(df)
    df                        = engineer_features(df)
    df, le_dict               = encode_and_prune(df)
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler)                  = split_and_balance(df)
    save_artifacts(X_train, y_train, X_val, y_val,
                   X_test,  y_test,  scaler, le_dict)
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler)


if __name__ == "__main__":
    run_preprocessing()