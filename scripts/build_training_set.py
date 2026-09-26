import os
import sys
import gc
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
import hashlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.entity_resolution.blocking import Blocker
from src.entity_resolution.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH, PROJECT_ROOT

BUILD_OUT_DIR = os.path.join(PROJECT_ROOT, "build_out")
os.makedirs(BUILD_OUT_DIR, exist_ok=True)

TRAIN_PAIRS_OUT = os.path.join(BUILD_OUT_DIR, "train_pairs.parquet")
VAL_PAIRS_OUT = os.path.join(BUILD_OUT_DIR, "val_pairs.parquet")

def load_ground_truth():
    print("Loading ground truth...")
    gt = {}
    chunk_iter = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str, chunksize=100_000)
    for chunk in chunk_iter:
        for row in chunk.itertuples(index=False):
            # row.matched_entity_ids could be NaN if empty
            if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids.strip():
                gt[row.source1_entity_id] = set()
            else:
                gt[row.source1_entity_id] = set(row.matched_entity_ids.split(','))
    print(f"Loaded ground truth for {len(gt)} S1 entities.")
    return gt

def get_split(s1_id):
    # Deterministic split: 85% train, 15% val
    # Prevents data leakage between train and val (same S1 entity shouldn't span both)
    h = int(hashlib.md5(s1_id.encode('utf-8')).hexdigest(), 16)
    return 'train' if (h % 100) < 85 else 'val'

def main(limit=None):
    print(f"Building Blocker indexes on S2 and S3... (limit={limit})")
    blocker = Blocker(s2_path=TRAIN_SOURCE2, s3_path=TRAIN_SOURCE3, rare_freq_threshold=50)
    
    gt = load_ground_truth()
    
    print("Generating candidate pairs...")
    
    # We will stream out to parquet to save memory
    train_schema = pa.schema([
        ('s1_id', pa.string()),
        ('c_id', pa.string()),
        ('label', pa.int8()),
        ('country', pa.string())
    ])
    
    train_writer = pq.ParquetWriter(TRAIN_PAIRS_OUT, train_schema)
    val_writer = pq.ParquetWriter(VAL_PAIRS_OUT, train_schema)
    
    total_s1 = 0
    total_candidates = 0
    total_positives = 0
    
    train_records = {'s1_id': [], 'c_id': [], 'label': [], 'country': []}
    val_records = {'s1_id': [], 'c_id': [], 'label': [], 'country': []}
    
    def flush_records(records, writer):
        if not records['s1_id']:
            return
        table = pa.Table.from_pydict(records, schema=train_schema)
        writer.write_table(table)
        records['s1_id'].clear()
        records['c_id'].clear()
        records['label'].clear()
        records['country'].clear()

    chunk_iter = pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str, chunksize=10_000)
    for idx, chunk in enumerate(chunk_iter):
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            country = row.country
            
            candidates = blocker.get_candidates(pd.Series({
                'entity_id': s1_id,
                'business_name': row.business_name,
                'business_address': row.business_address,
                'country': country
            }))
            
            true_matches = gt.get(s1_id, set())
            
            total_s1 += 1
            total_candidates += len(candidates)
            
            split = get_split(s1_id)
            target_records = train_records if split == 'train' else val_records
            
            for cid in candidates:
                label = 1 if cid in true_matches else 0
                if label == 1:
                    total_positives += 1
                
                target_records['s1_id'].append(s1_id)
                target_records['c_id'].append(cid)
                target_records['label'].append(label)
                target_records['country'].append(country)
                
        if limit and total_s1 >= limit:
            break
        
        # Flush every chunk to keep memory tight
        flush_records(train_records, train_writer)
        flush_records(val_records, val_writer)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {(idx + 1) * 10000} S1 entities... Avg candidates: {total_candidates / total_s1:.1f}")
            
    # Final flush
    flush_records(train_records, train_writer)
    flush_records(val_records, val_writer)

    train_writer.close()
    val_writer.close()
    
    print("Done!")
    print(f"Total S1 processed: {total_s1}")
    print(f"Total pairs generated: {total_candidates}")
    print(f"Total positives in pairs: {total_positives}")
    print(f"Positive rate: {total_positives / total_candidates if total_candidates > 0 else 0:.4f}")
    print(f"Avg candidates per S1: {total_candidates / total_s1 if total_s1 > 0 else 0:.2f}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
