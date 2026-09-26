import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.entity_resolution.config import PROJECT_ROOT
from src.entity_resolution.evaluation import macro_f0_5

BUILD_OUT_DIR = os.path.join(PROJECT_ROOT, "build_out")
TRAIN_FEATS = os.path.join(BUILD_OUT_DIR, "train_features.parquet")
VAL_FEATS = os.path.join(BUILD_OUT_DIR, "val_features.parquet")

FEATURES = [
    'name_jw', 'name_lev', 'name_tsr', 'name_tsort', 'name_partial', 'name_token_jaccard', 'name_first_token_match',
    'addr_jw', 'addr_lev', 'addr_tsr', 'addr_token_jaccard', 'addr_len_ratio', 'street_num_match',
    'pin_exact', 'pin_conflict', 'pin_prefix', 'country_match', 'city_token_jaccard',
    'matched_exact_name', 'matched_pin', 'matched_rare_name', 'matched_rare_addr'
]

def load_data(path, is_train=False):
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    
    # Optional: If training data is >15M rows, we might downsample negatives to save memory
    if is_train and len(df) > 15_000_000:
        print(f"Dataset is large ({len(df)} rows). Downsampling negatives to save memory...")
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0].sample(n=min(len(df)-len(pos), 10_000_000), random_state=42)
        df = pd.concat([pos, neg], ignore_index=True)
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Downsampled to {len(df)} rows.")
        
    return df

def f05_eval(preds, dataset):
    y = dataset.get_label()
    best_f, best_t = 0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        p = (preds >= t).astype(int)
        tp = ((y==1)&(p==1)).sum()
        fp = ((y==0)&(p==1)).sum()
        fn = ((y==1)&(p==0)).sum()
        prec = tp/(tp+fp+1e-9)
        rec = tp/(tp+fn+1e-9)
        f = 1.25*prec*rec/(0.25*prec+rec+1e-9)
        if f > best_f: 
            best_f, best_t = f, t
    return 'f05', best_f, True

def run():
    train_df = load_data(TRAIN_FEATS, is_train=True)
    val_df = load_data(VAL_FEATS, is_train=False)
    
    pos_count = (train_df['label'] == 1).sum()
    neg_count = (train_df['label'] == 0).sum()
    print(f"Train Positives: {pos_count}, Negatives: {neg_count}")
    
    train_dataset = lgb.Dataset(train_df[FEATURES], label=train_df['label'], free_raw_data=False)
    val_dataset = lgb.Dataset(val_df[FEATURES], label=val_df['label'], reference=train_dataset, free_raw_data=False)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': neg_count / pos_count if pos_count > 0 else 1.0,
        'max_depth': 8,
        'verbose': -1,
    }
    
    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=300,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(50)
        ]
    )
    
    model.save_model(os.path.join(BUILD_OUT_DIR, "lgbm_model.txt"))
    
    print("Calibrating on Validation Set...")
    val_raw_probs = model.predict(val_df[FEATURES])
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_raw_probs, val_df['label'])
    
    import pickle
    with open(os.path.join(BUILD_OUT_DIR, "isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)
        
    val_cal = iso.transform(val_raw_probs)
    val_df['prob'] = val_cal
    
    print("Sweeping Thresholds by Country...")
    THRESHOLDS = {}
    
    # We must format predictions as required by macro_f0_5:
    # y_true: dict mapping s1_id -> set of S2/S3 true matches
    # y_pred: dict mapping s1_id -> set of S2/S3 pred matches
    
    # Build y_true once
    y_true_by_country = {}
    for c in val_df['country'].unique():
        if pd.isna(c): continue
        mask = val_df['country'] == c
        c_df = val_df[mask]
        
        y_true = {}
        for s1, group in c_df.groupby('s1_id'):
            y_true[s1] = set(group.loc[group['label'] == 1, 'c_id'].tolist())
            
        y_true_by_country[c] = (c_df, y_true)
    
    # Also evaluate France logic: France has no training data, we will set it to US + 0.05
    for c, (c_df, y_true) in y_true_by_country.items():
        best_f, best_t = 0, 0.7
        for t in np.arange(0.30, 0.96, 0.005):
            preds_mask = c_df['prob'] >= t
            
            y_pred = {}
            for s1, group in c_df[preds_mask].groupby('s1_id'):
                # Cap to 20 per S1 if needed, but for val sweep we just group
                y_pred[s1] = set(group['c_id'].tolist())
                
            # Note: entities with no predictions won't be in y_pred, which macro_f0_5 handles
            f = macro_f0_5(y_true, y_pred)
            if f > best_f:
                best_f, best_t = f, t
                
        THRESHOLDS[c] = best_t
        print(f"{c}: threshold={best_t:.3f}, F0.5={best_f:.4f}")
        
    if 'US' in THRESHOLDS:
        THRESHOLDS['FR'] = THRESHOLDS['US'] + 0.05
    else:
        THRESHOLDS['FR'] = 0.85
        
    print(f"FR fallback: threshold={THRESHOLDS['FR']:.3f}")
    
    with open(os.path.join(BUILD_OUT_DIR, "thresholds.json"), "w") as f:
        json.dump(THRESHOLDS, f)
        
    # Print overall validation macro F0.5 with chosen thresholds
    y_true_all = {}
    y_pred_all = {}
    
    for s1, group in val_df.groupby('s1_id'):
        y_true_all[s1] = set(group.loc[group['label'] == 1, 'c_id'].tolist())
        c = group['country'].iloc[0]
        t = THRESHOLDS.get(c, 0.80)
        
        matches = group.loc[group['prob'] >= t, 'c_id'].tolist()
        y_pred_all[s1] = set(matches[:20])  # Cap at 20
        
    f_all = macro_f0_5(y_true_all, y_pred_all)
    print(f"Overall Validation Macro F0.5: {f_all:.4f}")
    print("Done!")

if __name__ == "__main__":
    run()
