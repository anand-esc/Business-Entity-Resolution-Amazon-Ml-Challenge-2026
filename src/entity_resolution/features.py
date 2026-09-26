import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio, partial_ratio
import re
from src.entity_resolution.blocking import normalize_text, extract_pin_codes, tokenize_address, strip_legal_suffixes

def _first_token(text):
    if not isinstance(text, str) or not text:
        return ""
    return text.split()[0]

def _tokens(text):
    if not isinstance(text, str) or not text:
        return set()
    return set(text.split())

def _token_jaccard(tokens1, tokens2):
    if not tokens1 and not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0

def _extract_number(text):
    if not isinstance(text, str) or not text:
        return None
    match = re.search(r'\b\d+\b', text)
    return match.group() if match else None

def _longest_common_prefix(s1, s2):
    if not s1 or not s2:
        return 0
    mlen = min(len(s1), len(s2))
    for i in range(mlen):
        if s1[i] != s2[i]:
            return i
    return mlen

def extract_features(pairs_df, s1_lookup, target_lookup, blocker=None):
    """
    pairs_df: DataFrame with 's1_id', 'c_id', 'country'
    s1_lookup: dict mapping id -> dict/Series of {'business_name':..., 'business_address':...}
    target_lookup: dict mapping id -> dict/Series of {'business_name':..., 'business_address':...}
    blocker: Blocker instance for meta features
    """
    
    n = len(pairs_df)
    
    # 22 features matrix
    X = np.zeros((n, 22), dtype=np.float32)
    
    # We will use list comprehensions to extract arrays for rapidfuzz
    s1_ids = pairs_df['s1_id'].values
    c_ids = pairs_df['c_id'].values
    countries = pairs_df['country'].values
    
    # Fast vectorized lookup
    s1_names = s1_lookup.loc[s1_ids, 'business_name'].fillna('').values
    c_names = target_lookup.loc[c_ids, 'business_name'].fillna('').values
    
    s1_addrs = s1_lookup.loc[s1_ids, 'business_address'].fillna('').values
    c_addrs = target_lookup.loc[c_ids, 'business_address'].fillna('').values
    
    s1_countries = countries
    c_countries = target_lookup.loc[c_ids, 'country'].fillna('').values
    
    # Pre-process
    s1_names_norm = [normalize_text(n) for n in s1_names]
    c_names_norm = [normalize_text(n) for n in c_names]
    
    s1_names_stripped = [strip_legal_suffixes(n) for n in s1_names_norm]
    c_names_stripped = [strip_legal_suffixes(n) for n in c_names_norm]
    
    s1_addrs_norm = [normalize_text(a) for a in s1_addrs]
    c_addrs_norm = [normalize_text(a) for a in c_addrs]
    
    # --- Name features (7) ---
    # rapidfuzz processes lists natively but requires them as iterables
    # Note: rapidfuzz metric returns 0-1 or 0-100 depending on function
    name_jw = np.array([JaroWinkler.normalized_similarity(s1, c) for s1, c in zip(s1_names_norm, c_names_norm)])
    name_lev = np.array([Levenshtein.normalized_similarity(s1, c) for s1, c in zip(s1_names_norm, c_names_norm)])
    name_tsr = np.array([token_set_ratio(s1, c)/100.0 for s1, c in zip(s1_names_norm, c_names_norm)])
    name_tsort = np.array([token_sort_ratio(s1, c)/100.0 for s1, c in zip(s1_names_norm, c_names_norm)])
    name_partial = np.array([partial_ratio(s1, c)/100.0 for s1, c in zip(s1_names_norm, c_names_norm)])
    
    name_token_jaccard = np.array([_token_jaccard(_tokens(s1), _tokens(c)) for s1, c in zip(s1_names_stripped, c_names_stripped)])
    name_first_token_match = np.array([1.0 if _first_token(s1) == _first_token(c) and _first_token(s1) != "" else 0.0 for s1, c in zip(s1_names_norm, c_names_norm)])
    
    # --- Address features (6) ---
    addr_jw = np.array([JaroWinkler.normalized_similarity(s1, c) for s1, c in zip(s1_addrs_norm, c_addrs_norm)])
    addr_lev = np.array([Levenshtein.normalized_similarity(s1, c) for s1, c in zip(s1_addrs_norm, c_addrs_norm)])
    addr_tsr = np.array([token_set_ratio(s1, c)/100.0 for s1, c in zip(s1_addrs_norm, c_addrs_norm)])
    
    s1_addr_tokens = [tokenize_address(a) for a in s1_addrs]
    c_addr_tokens = [tokenize_address(a) for a in c_addrs]
    addr_token_jaccard = np.array([_token_jaccard(s1, c) for s1, c in zip(s1_addr_tokens, c_addr_tokens)])
    
    addr_len_ratio = np.array([min(len(s1), len(c)) / max(len(s1), len(c), 1) for s1, c in zip(s1_addrs_norm, c_addrs_norm)])
    
    s1_nums = [_extract_number(a) for a in s1_addrs_norm]
    c_nums = [_extract_number(a) for a in c_addrs_norm]
    street_num_match = np.array([1.0 if (s and c and s == c) else (-1.0 if s and c else 0.0) for s, c in zip(s1_nums, c_nums)])
    
    # --- Geographic features (5) ---
    s1_pins_list = [extract_pin_codes(a) for a in s1_addrs]
    c_pins_list = [extract_pin_codes(a) for a in c_addrs]
    
    s1_pins = [p[0] if p else "" for p in s1_pins_list]
    c_pins = [p[0] if p else "" for p in c_pins_list]
    
    pin_exact = np.array([1.0 if s and c and s == c else 0.0 for s, c in zip(s1_pins, c_pins)])
    pin_conflict = np.array([-1.0 if s and c and s != c else 0.0 for s, c in zip(s1_pins, c_pins)])
    pin_prefix = np.array([float(_longest_common_prefix(s, c)) for s, c in zip(s1_pins, c_pins)])
    country_match = np.array([1.0 if s.lower() == c.lower() and s else 0.0 for s, c in zip(s1_countries, c_countries)])
    
    # city token jaccard (non-numeric address tokens)
    s1_city_tokens = [{t for t in s if not t.isdigit()} for s in s1_addr_tokens]
    c_city_tokens = [{t for t in s if not t.isdigit()} for s in c_addr_tokens]
    city_token_jaccard = np.array([_token_jaccard(s1, c) for s1, c in zip(s1_city_tokens, c_city_tokens)])
    
    # --- Meta features (4) ---
    # These rely on the Blocker index to check if candidate is hit by index.
    # Note: To correctly do this during inference/training, we need to check if the candidate is in the index buckets for S1 keys.
    matched_exact_name = np.zeros(n)
    matched_pin = np.zeros(n)
    matched_rare_name = np.zeros(n)
    matched_rare_addr = np.zeros(n)
    
    if blocker is not None:
        for i in range(n):
            s1_name_stripped = s1_names_stripped[i]
            c = s1_countries[i]
            c_id = c_ids[i]
            
            # check exact name
            if s1_name_stripped and c_id in blocker.exact_name_idx.get((c, s1_name_stripped), []):
                matched_exact_name[i] = 1.0
                
            # check pin
            for pin in s1_pins_list[i]:
                if c_id in blocker.pin_idx.get((c, pin), []):
                    matched_pin[i] = 1.0
                    break
                    
            # check rare name
            for token in _tokens(s1_name_stripped):
                if c_id in blocker.rare_token_idx.get((c, token), []):
                    matched_rare_name[i] = 1.0
                    break
                    
            # check rare addr
            for token in s1_addr_tokens[i]:
                bucket = blocker.rare_addr_token_idx.get((c, token), [])
                if len(bucket) <= blocker.rare_freq_threshold and c_id in bucket:
                    matched_rare_addr[i] = 1.0
                    break

    # Assemble X
    X[:, 0] = name_jw
    X[:, 1] = name_lev
    X[:, 2] = name_tsr
    X[:, 3] = name_tsort
    X[:, 4] = name_partial
    X[:, 5] = name_token_jaccard
    X[:, 6] = name_first_token_match
    
    X[:, 7] = addr_jw
    X[:, 8] = addr_lev
    X[:, 9] = addr_tsr
    X[:, 10] = addr_token_jaccard
    X[:, 11] = addr_len_ratio
    X[:, 12] = street_num_match
    
    X[:, 13] = pin_exact
    X[:, 14] = pin_conflict
    X[:, 15] = pin_prefix
    X[:, 16] = country_match
    X[:, 17] = city_token_jaccard
    
    X[:, 18] = matched_exact_name
    X[:, 19] = matched_pin
    X[:, 20] = matched_rare_name
    X[:, 21] = matched_rare_addr
    
    # Feature names array for reference
    feature_names = [
        'name_jw', 'name_lev', 'name_tsr', 'name_tsort', 'name_partial', 'name_token_jaccard', 'name_first_token_match',
        'addr_jw', 'addr_lev', 'addr_tsr', 'addr_token_jaccard', 'addr_len_ratio', 'street_num_match',
        'pin_exact', 'pin_conflict', 'pin_prefix', 'country_match', 'city_token_jaccard',
        'matched_exact_name', 'matched_pin', 'matched_rare_name', 'matched_rare_addr'
    ]
    
    return X, feature_names
