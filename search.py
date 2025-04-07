import faiss
import json
import numpy as np
import os

# Import the query embedding function from the previous step
from query_processing import get_query_embedding

# --- Configuration ---
# Path to the FAISS index file created in step 4
index_path = 'cde_index.faiss'
# Path to the JSON file containing the ordered CDE IDs corresponding to the index
ids_path = 'cde_ids_ordered.json'
# Path to the original cleaned data (optional, needed for retrieving text in testing)
cleaned_data_path = 'cleaned_cde_all.json' 

# --- Load Index and IDs ---
# Load the FAISS index only once when the module is imported for efficiency.
index = None
cde_ids_ordered = None

print(f"Loading FAISS index from {index_path}...")
if os.path.exists(index_path):
    try:
        index = faiss.read_index(index_path)
        print(f"Successfully loaded FAISS index with {index.ntotal} vectors.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
else:
    print(f"Error: FAISS index file not found at {index_path}")

print(f"Loading ordered CDE IDs from {ids_path}...")
if os.path.exists(ids_path):
    try:
        with open(ids_path, 'r', encoding='utf-8') as f:
            cde_ids_ordered = json.load(f)
        print(f"Successfully loaded {len(cde_ids_ordered)} ordered CDE IDs.")
        
        # Basic validation: Check if number of IDs matches index size
        if index and index.ntotal != len(cde_ids_ordered):
            print(f"Warning: FAISS index size ({index.ntotal}) does not match number of loaded IDs ({len(cde_ids_ordered)}). Results might be inconsistent.")
            
    except Exception as e:
        print(f"Error loading ordered CDE IDs: {e}")
else:
    print(f"Error: Ordered CDE IDs file not found at {ids_path}")

# --- Search Function ---
def search_similar_cdes(query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """
    Searches the FAISS index for the top k most similar CDEs to the query embedding.

    Args:
        query_embedding: A normalized NumPy array representing the query embedding.
        k: The number of top similar items to retrieve.

    Returns:
        A list of tuples, where each tuple contains (CDE ID, similarity_score).
        Returns an empty list if the index or IDs are not loaded, or if k is invalid.
    """
    # Check if index and IDs were loaded successfully
    if not index or not cde_ids_ordered:
        print("Error: FAISS index or CDE IDs not loaded. Cannot perform search.")
        return []
        
    # Validate k
    if not isinstance(k, int) or k <= 0:
        print("Error: k must be a positive integer.")
        return []
        
    # Ensure k is not greater than the total number of items in the index
    k = min(k, index.ntotal)

    # Ensure query embedding is a 2D numpy array of float32 for FAISS search
    if not isinstance(query_embedding, np.ndarray):
        print("Error: query_embedding must be a NumPy array.")
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.array([query_embedding]).astype('float32') # Reshape to (1, dim)
    elif query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype('float32')

    print(f"Searching for top {k} similar CDEs...")
    try:
        # Perform the search
        # D: distances (cosine similarities in our case due to normalization + IndexFlatIP)
        # I: indices of the nearest neighbors in the FAISS index
        distances, indices = index.search(query_embedding, k)

        # Process results
        results = []
        for i in range(k):
            idx = indices[0][i]  # Get the index of the i-th nearest neighbor
            score = distances[0][i] # Get the similarity score
            
            # Check for valid index before accessing cde_ids_ordered
            if 0 <= idx < len(cde_ids_ordered):
                cde_id = cde_ids_ordered[idx] # Map index back to CDE ID
                results.append((cde_id, float(score)))
            else:
                print(f"Warning: Invalid index {idx} returned by FAISS search.")

        print(f"Search completed. Found {len(results)} results.")
        return results
        
    except Exception as e:
        print(f"An error occurred during FAISS search: {e}")
        return []

# --- Main Execution (for testing purposes) ---
if __name__ == "__main__":
    # This block executes only when the script is run directly (python search.py)
    
    # Define a sample query for testing
    sample_query = "measurement of pregnancy body weight"
    num_results = 10 # Number of results to retrieve
    
    print(f"\n--- Testing Similarity Search --- K={num_results}")
    print(f"Test Query: \"{sample_query}\"")

    # 1. Get the query embedding (using the function from query_processing.py)
    query_vector = get_query_embedding(sample_query)

    # 2. Perform the search if embedding was successful
    if query_vector is not None:
        search_results = search_similar_cdes(query_vector, k=num_results)

        # 3. Display the results (IDs and Scores)
        if search_results:
            print(f"\nTop {len(search_results)} results:")
            for cde_id, score in search_results:
                print(f"  - ID: {cde_id}, Score: {score:.4f}")
                
            # Optional: Load original data to show text for context (requires cleaned_cde_all.json)
            if os.path.exists(cleaned_data_path):
                print("\nRetrieving text for context:")
                try:
                    with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                        cleaned_data = {item['id']: item['text'] for item in json.load(f)}
                    
                    for cde_id, score in search_results:
                         # Handle cases where ID might not be in the cleaned data dict (shouldn't happen if generated correctly)
                        text = cleaned_data.get(cde_id, "Text not found.") 
                        print(f"  - ID: {cde_id}, Score: {score:.4f}\n    Text: {text[:150]}...") # Print first 150 chars
                except Exception as e:
                    print(f"    Error loading or processing cleaned data for context: {e}")
        else:
            print("Search did not return any results.")
    else:
        print("Failed to generate query embedding. Cannot perform search.") 