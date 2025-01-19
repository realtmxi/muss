# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import faiss
from tqdm import tqdm

from muss.utils.submitit import get_executor
from muss.utils.helpers import get_file_hash, get_files_hash, log_action, yield_lines
from muss.resources.paths import get_dataset_dir
from muss.laser import get_laser_embeddings
from muss.mining.preprocessing import (
    get_subshard_paths,
    get_sentences_paths,
    sentence_tokenize_subshard,
    split_ccnet_shard,
    create_base_index,
    get_index_name,
)
from muss.mining.nn_search import (
    get_cache_dir,
    get_results_path,
    compute_and_save_nn_batched,
    get_paraphrase_pairs,
    get_pairs_path,
    compute_and_save_simplification_pairs,
    get_index_path,
    compute_and_save_embeddings,
    get_filter_string_representation,
    combine_simplifications_in_dataset,
    get_simplification_pairs_paths,
)
from muss.mining.filtering import SimplicityScorer

language = 'si'
cluster = 'local'
dataset_dir = Path("/home/realtmxi/Github/muss/sinhala/MADLAD_CulturaX_cleaned/data")
cache_dir = dataset_dir / "cache"
pairs_dir = cache_dir / "pairs"
nn_search_results_dir = cache_dir / "nn_search_results"
topk = 8
nprobe = 16
# For large jobs only
slurm_partition = 'debug'
slurm_array_parallelism = 1024

# Function to process custom dataset
def process_custom_dataset():
    # Embedding configuration
    embeddings_type_name = f"laser_{language}"
    get_embeddings = lambda sentences: get_laser_embeddings(
        sentences, max_tokens=1000, language=language
    )

    # Compute embeddings
    with log_action("Computing embeddings for Sinhala dataset"):
        dataset_files = list(dataset_dir.glob("*.txt"))
        if not dataset_files:
            raise FileNotFoundError("No text files found in the dataset directory.")
        
        for file_path in dataset_files:
            sentences = list(yield_lines(file_path))  # Read lines from the text file
            embeddings = get_embeddings(sentences)
            print(f"Computed embeddings for {file_path.name}: {embeddings.shape}")

    # Mine paraphrases
    with log_action("Mining paraphrases"):
        query_sentences_paths = dataset_files
        db_sentences_paths = dataset_files  # Assuming the same dataset for query and DB
        for query_sentences_path in query_sentences_paths:
            results_path = get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir)
            if results_path.exists():
                print(f"Results already exist for {query_sentences_path.name}, skipping...")
                continue
            compute_and_save_nn_batched(
                query_sentences_path, db_sentences_paths, topk, nprobe, cache_dir / "indexes", nn_search_results_dir
            )
    
    # Simplification scoring (if required)
    with log_action("Scoring simplifications"):
        filter_kwargs = {
            "density": 0.6,
            "distance": 0.05,
            "levenshtein": 0.2,
            "simplicity": 0.0,
        }
        simplification_pairs = get_simplification_pairs_paths(
            query_sentences_paths, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
        )
        dataset = f"sinhala_{embeddings_type_name}"
        combine_simplifications_in_dataset(simplification_pairs, dataset)