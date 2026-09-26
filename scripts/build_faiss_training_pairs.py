import os
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from entity_resolution.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, PROJECT_ROOT
from entity_resolution.faiss_blocker import FaissBlocker

BUILD_OUT_DIR = os.path.join(PROJECT_ROOT, "build_out")

def run_faiss_pipeline():
    index_path = os.path.join(BUILD_OUT_DIR, "faiss_train.index")
    map_path = os.path.join(BUILD_OUT_DIR, "faiss_train_map.csv")
    out_parquet = os.path.join(BUILD_OUT_DIR, "train_pairs_faiss.parquet")
    
    print("=== FAISS PIPELINE START ===")
    
    # 1. Build the FAISS Index for S2 and S3 (Takes ~2.5 hours)
    blocker = FaissBlocker(TRAIN_SOURCE2, TRAIN_SOURCE3, index_path, map_path)
    if not os.path.exists(index_path):
        print("\n--- PHASE 1: Building FAISS Index ---")
        blocker.build()
    else:
        print("\n--- FAISS Index already exists. Skipping build. ---")
        
    # 2. Load the Index
    print("\n--- PHASE 2: Loading FAISS Index into RAM ---")
    blocker.load()
    
    # 3. Generate Pairs for S1 (Takes ~45 minutes)
    print("\n--- PHASE 3: Generating Candidate Pairs ---")
    
    if os.path.exists(out_parquet):
        os.remove(out_parquet)
        
    writer = None
    schema = pa.schema([
        ('s1_id', pa.string()),
        ('c_id', pa.string()),
        ('country', pa.string())
    ])
    
    chunksize = 50_000
    for i, chunk in enumerate(pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str, chunksize=chunksize, usecols=['entity_id', 'business_name', 'business_address', 'country'])):
        chunk.fillna('', inplace=True)
        s1_names = chunk['business_name'].tolist()
        s1_addrs = chunk['business_address'].tolist()
        s1_ids = chunk['entity_id'].tolist()
        countries = chunk['country'].tolist()
        
        # Query FAISS
        # We query top 15 to be safe (Amazon likes <50, 15 is excellent)
        candidate_lists = blocker.get_candidates(s1_names, s1_addrs, top_k=15)
        
        pairs = []
        for s1_id, c, cands in zip(s1_ids, countries, candidate_lists):
            for c_id in cands:
                pairs.append({'s1_id': s1_id, 'c_id': c_id, 'country': c})
                
        # Write to parquet
        df_pairs = pd.DataFrame(pairs)
        table = pa.Table.from_pandas(df_pairs, schema=schema)
        
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, schema)
        writer.write_table(table)
        
        print(f"Processed S1 chunk {i+1}. Pairs generated: {len(df_pairs)}")
        
        del chunk, s1_names, s1_addrs, candidate_lists, pairs, df_pairs, table
        gc.collect()
        
    if writer:
        writer.close()
        
    print("\n=== FAISS PIPELINE COMPLETE ===")
    print(f"Candidate pairs saved to {out_parquet}")

if __name__ == "__main__":
    run_faiss_pipeline()
