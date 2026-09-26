import os
import sys
import gc
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.entity_resolution.blocking import Blocker
from src.entity_resolution.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, PROJECT_ROOT
from src.entity_resolution.features import extract_features

BUILD_OUT_DIR = os.path.join(PROJECT_ROOT, "build_out")
TRAIN_PAIRS = os.path.join(BUILD_OUT_DIR, "train_pairs.parquet")
VAL_PAIRS = os.path.join(BUILD_OUT_DIR, "val_pairs.parquet")
TRAIN_FEATS = os.path.join(BUILD_OUT_DIR, "train_features.parquet")
VAL_FEATS = os.path.join(BUILD_OUT_DIR, "val_features.parquet")

def load_lookup(path):
    print(f"Loading lookup for {path}...")
    df = pd.read_csv(path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'])
    df.set_index('entity_id', inplace=True)
    return df.to_dict('index')

def compute_features_for_split(pairs_path, out_path, s1_lookup, target_lookup, blocker, chunksize=1_000_000):
    print(f"Extracting features from {pairs_path} -> {out_path}")
    if not os.path.exists(pairs_path):
        print(f"File not found: {pairs_path}")
        return
    
    parquet_file = pq.ParquetFile(pairs_path)
    schema = None
    writer = None
    
    total = 0
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        df = batch.to_pandas()
        # Split df into 8 chunks for threading
        import concurrent.futures
        import numpy as np
        
        num_threads = 8
        splits = np.array_split(df, num_threads)
        
        def process_split(sub_df):
            return extract_features(sub_df, s1_lookup, target_lookup, blocker)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_split, splits))
            
        # Combine results
        X_list = [r[0] for r in results]
        f_names = results[0][1]
        X = np.vstack(X_list)
        
        # Add label and country and ids back
        res_df = pd.DataFrame(X, columns=f_names)
        res_df['s1_id'] = df['s1_id'].values
        res_df['c_id'] = df['c_id'].values
        res_df['label'] = df['label'].values
        res_df['country'] = df['country'].values
        
        table = pa.Table.from_pandas(res_df)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_path, schema)
        
        writer.write_table(table)
        total += len(df)
        print(f"Processed {total} pairs...")
        
        del df, res_df, X, table
        gc.collect()
        
    if writer:
        writer.close()
    print(f"Done extracting {total} pairs.")

def main():
    s1_lookup = load_lookup(TRAIN_SOURCE1)
    
    print("Loading S2 and S3 for lookup...")
    s2_lookup = load_lookup(TRAIN_SOURCE2)
    s3_lookup = load_lookup(TRAIN_SOURCE3)
    
    target_lookup = {**s2_lookup, **s3_lookup}
    del s2_lookup
    del s3_lookup
    gc.collect()
    
    print("Initializing Blocker for meta-features...")
    blocker = Blocker(s2_path=TRAIN_SOURCE2, s3_path=TRAIN_SOURCE3, rare_freq_threshold=50)
    
    # compute_features_for_split(VAL_PAIRS, VAL_FEATS, s1_lookup, target_lookup, blocker)
    compute_features_for_split(TRAIN_PAIRS, TRAIN_FEATS, s1_lookup, target_lookup, blocker)

if __name__ == "__main__":
    main()
