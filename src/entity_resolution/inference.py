import os
import sys
import gc
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.entity_resolution.blocking import Blocker
from src.entity_resolution.config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3, PROJECT_ROOT
)
from src.entity_resolution.features import extract_features
from src.entity_resolution.model import FEATURES

BUILD_OUT_DIR = os.path.join(PROJECT_ROOT, "build_out")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MATCHING_OUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def load_lookup(path):
    print(f"Loading lookup for {path}...")
    df = pd.read_csv(path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'])
    df.fillna('', inplace=True)
    df.set_index('entity_id', inplace=True)
    return df

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading models and thresholds...")
    lgbm_model = lgb.Booster(model_file=os.path.join(BUILD_OUT_DIR, "lgbm_model.txt"))
    with open(os.path.join(BUILD_OUT_DIR, "isotonic.pkl"), "rb") as f:
        iso = pickle.load(f)
    with open(os.path.join(BUILD_OUT_DIR, "thresholds.json"), "r") as f:
        THRESHOLDS = json.load(f)
        
    print("Loading test lookups...")
    s1_lookup = load_lookup(TEST_SOURCE1)
    s2_lookup = load_lookup(TEST_SOURCE2)
    s3_lookup = load_lookup(TEST_SOURCE3)
    target_lookup = pd.concat([s2_lookup, s3_lookup])
    del s2_lookup, s3_lookup
    gc.collect()
    
    print("Initializing Blocker on Test Set...")
    # Blocker index needs to be rebuilt for test S2/S3
    blocker = Blocker(s2_path=TEST_SOURCE2, s3_path=TEST_SOURCE3, rare_freq_threshold=50)
    
    print("Running Inference...")
    with open(MATCHING_OUT, 'w', newline='') as f_match, open(CANDIDATE_OUT, 'w', newline='') as f_cand:
        f_match.write('source1_entity_id\tmatched_entity_ids\n')
        f_cand.write('source1_entity_id\tcandidate_entity_ids\n')
        
        chunk_iter = pd.read_csv(TEST_SOURCE1, sep='\t', dtype=str, chunksize=50_000)
        
        for i, chunk in enumerate(chunk_iter):
            print(f"Processing Test S1 chunk {i+1}...")
            
            pairs = []
            for row in chunk.itertuples(index=False):
                s1_id = row.entity_id
                c = row.country
                
                cands = blocker.get_candidates(pd.Series({
                    'entity_id': s1_id,
                    'business_name': row.business_name,
                    'business_address': row.business_address,
                    'country': c
                }))
                
                # Write to candidate_pairs.tsv immediately
                cand_list_str = ','.join(cands)
                f_cand.write(f'{s1_id}\t{cand_list_str}\n')
                
                for cid in cands:
                    pairs.append({'s1_id': s1_id, 'c_id': cid, 'country': c})
                    
            if not pairs:
                # all singletons
                for row in chunk.itertuples(index=False):
                    f_match.write(f'{row.entity_id}\t\n')
                continue
                
            pairs_df = pd.DataFrame(pairs)
            
            # Extract features using multi-threading
            import concurrent.futures
            num_threads = 8
            splits = np.array_split(pairs_df, num_threads)
            
            def process_split(sub_df):
                return extract_features(sub_df, s1_lookup, target_lookup, blocker)
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(process_split, splits))
                
            X_list = [r[0] for r in results]
            X = np.vstack(X_list)
            
            # Predict
            raw_probs = lgbm_model.predict(X)
            cal_probs = iso.transform(raw_probs)
            pairs_df['prob'] = cal_probs
            
            # Threshold and write
            for s1_id, group in pairs_df.groupby('s1_id'):
                c = group['country'].iloc[0]
                t = THRESHOLDS.get(c, 0.85)  # fallback to conservative
                
                # Sort by prob descending and take top 20 above threshold
                group = group[group['prob'] >= t].sort_values('prob', ascending=False).head(20)
                matches = group['c_id'].tolist()
                f_match.write(f'{s1_id}\t{",".join(matches)}\n')
                
            # For S1s with 0 candidates, we didn't group by them in pairs_df
            # We need to make sure every S1 from chunk gets a row in matching_results
            s1_processed = set(pairs_df['s1_id'].unique())
            for row in chunk.itertuples(index=False):
                if row.entity_id not in s1_processed:
                    f_match.write(f'{row.entity_id}\t\n')
                    
            del pairs_df, X, raw_probs, cal_probs
            gc.collect()
            
    print("Inference Complete.")

if __name__ == "__main__":
    run_inference()
