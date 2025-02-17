from pathlib import Path
import faiss
from tqdm import tqdm
from muss.utils.helpers import log_action, yield_lines
from muss.laser import get_laser_embeddings
from muss.mining.nn_search import (
    get_results_path,
    compute_and_save_nn_batched,
    get_simplification_pairs_paths,
    combine_simplifications_in_dataset,
)

# Configuration
language = 'sin'
dataset_dir = Path("/home/realtmxi/Github/muss/sinhala/MADLAD_CulturaX_cleaned/data/")
cache_dir = dataset_dir / "cache"
pairs_dir = cache_dir / "pairs"
nn_search_results_dir = cache_dir / "nn_search_results"
topk = 8
nprobe = 16

# Function to process custom dataset
def process_custom_dataset():
    print("Starting the Sinhala dataset mining process...")

    # Embedding configuration
    embeddings_type_name = f"laser_{language}"
    get_embeddings = lambda sentences: get_laser_embeddings(
        sentences, max_tokens=1000, language=language
    )

    # Compute embeddings
    with log_action("Computing embeddings for Sinhala dataset"):
        dataset_files = list(dataset_dir.glob("*.txt"))
        print(f"Dataset directory: {dataset_dir}")
        print(f"Found files: {dataset_files}")

        if not dataset_files:
            raise FileNotFoundError("No .txt files found in the dataset directory.")

        for file_path in dataset_files:
            print(f"Processing file: {file_path}")
            sentences = list(yield_lines(file_path))  # Read lines from the text file
            print(f"Loaded {len(sentences)} sentences from {file_path}")
            embeddings = get_embeddings(sentences)
            print(f"Computed embeddings for {file_path.name}: {embeddings.shape}")

    # Mine paraphrases
    with log_action("Mining paraphrases"):
        query_sentences_paths = dataset_files
        db_sentences_paths = dataset_files  # Assuming the same dataset for query and DB

        print("Starting paraphrase mining...")
        for query_sentences_path in query_sentences_paths:
            print(f"Processing query file: {query_sentences_path}")
            results_path = get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir)
            
            if results_path.exists():
                print(f"Results already exist for {query_sentences_path.name}, skipping...")
                continue

            compute_and_save_nn_batched(
                query_sentences_path, db_sentences_paths, topk, nprobe, cache_dir / "indexes", nn_search_results_dir
            )
            print(f"Saved nearest neighbor results for {query_sentences_path.name}")

    # Simplification scoring (if required)
    with log_action("Scoring simplifications"):
        print("Starting simplification scoring...")
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
        print(f"Final dataset saved as: {dataset}")

    print("Mining process completed successfully.")

# Entry point
if __name__ == "__main__":
    try:
        process_custom_dataset()
    except Exception as e:
        print(f"An error occurred: {e}")
