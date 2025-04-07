import faiss
import numpy as np
import pickle
import json
import os

# --- Configuration ---
# Input pickle file containing the embeddings dictionary {id: embedding}
input_pickle_path = 'cde_embeddings.pkl'
# Output FAISS index file path
output_index_path = 'cde_index.faiss'
# Output JSON file path for the ordered list of CDE IDs
output_ids_path = 'cde_ids_ordered.json'

# --- Helper Functions ---
def load_embeddings_dict(pickle_path):
    """Loads the embeddings dictionary from a pickle file."""
    print(f"Loading embeddings dictionary from {pickle_path}...")
    if not os.path.exists(pickle_path):
        print(f"Error: Embeddings file not found at {pickle_path}")
        return None
    try:
        with open(pickle_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"Successfully loaded {len(embeddings_dict)} embeddings.")
        return embeddings_dict
    except Exception as e:
        print(f"Error loading embeddings from pickle file: {e}")
        return None

def save_faiss_index(index, index_path):
    """Saves the FAISS index to a file."""
    print(f"Saving FAISS index to {index_path}...")
    try:
        faiss.write_index(index, index_path)
        print("Successfully saved FAISS index.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def save_ordered_ids(ids_list, ids_path):
    """Saves the ordered list of IDs to a JSON file."""
    print(f"Saving ordered CDE IDs to {ids_path}...")
    try:
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(ids_list, f, ensure_ascii=False, indent=4)
        print("Successfully saved ordered CDE IDs.")
    except Exception as e:
        print(f"Error saving ordered IDs: {e}")

# --- Main Indexing Logic ---
def create_faiss_index():
    """Loads embeddings, normalizes them, creates a FAISS index, and saves it."""
    # 1. Load Embeddings Dictionary
    embeddings_dict = load_embeddings_dict(input_pickle_path)
    if not embeddings_dict:
        return

    # 2. Prepare Data for FAISS
    # Extract IDs and embeddings into separate lists, maintaining order
    # It's important that the order of IDs matches the order of embeddings
    cde_ids_ordered = list(embeddings_dict.keys())
    embeddings_list = [embeddings_dict[doc_id] for doc_id in cde_ids_ordered]

    # Convert embeddings list to a NumPy array of float32 (required by FAISS)
    embeddings_np = np.array(embeddings_list).astype('float32')
    
    # Check if embeddings were loaded correctly
    if embeddings_np.size == 0:
        print("Error: Embeddings array is empty. Cannot create index.")
        return
        
    print(f"Embeddings loaded into NumPy array with shape: {embeddings_np.shape}")
    embedding_dim = embeddings_np.shape[1] # Get the dimension of the embeddings

    # 3. Normalize Embeddings for Cosine Similarity
    # Voyage embeddings are typically used with cosine similarity.
    # Normalizing vectors makes the inner product (IP) equivalent to cosine similarity.
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings_np)
    print("Embeddings normalized.")

    # 4. Create FAISS Index
    # Using IndexFlatIP because we normalized the vectors for cosine similarity (inner product).
    print(f"Creating FAISS index (IndexFlatIP) with dimension {embedding_dim}...")
    index = faiss.IndexFlatIP(embedding_dim)

    # 5. Add Embeddings to Index
    print(f"Adding {embeddings_np.shape[0]} embeddings to the index...")
    index.add(embeddings_np)
    print(f"Total embeddings in index: {index.ntotal}")

    # 6. Save the Index and Ordered IDs
    save_faiss_index(index, output_index_path)
    save_ordered_ids(cde_ids_ordered, output_ids_path)

    print("Indexing process completed.")

# --- Main Execution ---
if __name__ == "__main__":
    create_faiss_index() 