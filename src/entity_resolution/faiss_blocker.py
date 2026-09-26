import os
import gc
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class FaissBlocker:
    """
    Builds a Highly Optimized FAISS Vector Database for Candidate Generation.
    Uses IVFPQ (Product Quantization) to compress 15GB of raw vectors into <1GB of RAM.
    """
    def __init__(self, s2_path, s3_path, index_path="build_out/faiss.index", map_path="build_out/faiss_map.csv"):
        self.s2_path = s2_path
        self.s3_path = s3_path
        self.index_path = index_path
        self.map_path = map_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.id_mapping = []
        
    def build(self):
        d = 384
        nlist = 1024
        m = 48  # compresses 384 floats into 48 bytes
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        
        chunksize = 100_000
        is_trained = False
        
        for path in [self.s2_path, self.s3_path]:
            if not os.path.exists(path):
                continue
            
            print(f"\nProcessing {path} for FAISS...")
            for i, chunk in enumerate(pd.read_csv(path, sep='\t', dtype=str, chunksize=chunksize, usecols=['entity_id', 'business_name', 'business_address'])):
                chunk.fillna('', inplace=True)
                
                # Combine name and address for deep semantic embedding
                texts = (chunk['business_name'] + " " + chunk['business_address']).tolist()
                
                # Encode (this uses PyTorch under the hood and will max out CPU cores)
                embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
                
                if not is_trained:
                    print("Training FAISS IVFPQ Index (this only happens once)...")
                    self.index.train(embeddings)
                    is_trained = True
                
                self.index.add(embeddings)
                self.id_mapping.extend(chunk['entity_id'].tolist())
                
                print(f"Added chunk {i+1}. Total vectors in FAISS: {self.index.ntotal}")
                
                del chunk, texts, embeddings
                gc.collect()
                
        print("\nSaving FAISS index to disk...")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        pd.DataFrame({'entity_id': self.id_mapping}).to_csv(self.map_path, index=False)
        print("FAISS Build Complete!")

    def load(self):
        print("Loading FAISS index into RAM...")
        self.index = faiss.read_index(self.index_path)
        self.index.nprobe = 32  # Check 32 Voronoi cells for high recall
        
        print("Loading FAISS ID mapping...")
        df_map = pd.read_csv(self.map_path, dtype=str)
        self.id_mapping = df_map['entity_id'].tolist()
        
    def get_candidates(self, s1_names, s1_addrs, top_k=10):
        """
        Takes lists of S1 names and addresses.
        Returns a list of candidate ID sets (length top_k) for each S1 entity.
        """
        texts = [n + " " + a for n, a in zip(s1_names, s1_addrs)]
        embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        
        distances, indices = self.index.search(embeddings, top_k)
        
        results = []
        for row_idx in indices:
            cands = [self.id_mapping[i] for i in row_idx if i != -1]
            results.append(cands)
            
        return results

if __name__ == "__main__":
    # Test script to start building the FAISS index for the training set
    from config import TRAIN_SOURCE2, TRAIN_SOURCE3
    blocker = FaissBlocker(TRAIN_SOURCE2, TRAIN_SOURCE3)
    blocker.build()
