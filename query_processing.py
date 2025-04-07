import voyageai
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables (.env file should be in the parent directory or project root where the script is run from)
# This assumes the .env file is in the 'semantic_clustering_NIH_CDE' directory
load_dotenv()
# Voyage AI model name (MUST match the one used for document embeddings in generate_embeddings.py)
model_name = "voyage-large-2"

# --- Initialize Voyage AI Client ---
# Attempt to initialize the client once when the module is loaded.
# This is more efficient than initializing it inside the function if called repeatedly.
api_key = os.getenv("VOYAGE_API_KEY")
vo = None # Initialize vo to None
if api_key:
    try:
        vo = voyageai.Client(api_key=api_key)
        print(f"Voyage AI client initialized successfully for query processing (model: {model_name}).")
    except Exception as e:
        print(f"Error: Failed to initialize Voyage AI client: {e}")
        # The get_query_embedding function will handle the case where vo is None.
else:
    print("Error: VOYAGE_API_KEY environment variable not set or found.")
    print("Please ensure the .env file with VOYAGE_API_KEY is in the C:\\Users\\Boyan\\Documents\\Code\\NLP\\semantic_clustering_NIH_CDE directory.")

# --- Query Embedding Function ---
def get_query_embedding(query_text: str) -> np.ndarray | None:
    """
    Generates a normalized embedding for a given text query using Voyage AI.

    Args:
        query_text: The user's query string.

    Returns:
        A normalized NumPy array representing the query embedding (float32),
        or None if an error occurs (e.g., client not initialized, API error, empty query).
    """
    global vo # Use the client initialized globally
    
    # Check if client was initialized successfully
    if not vo:
        print("Error: Voyage AI client is not available. Cannot generate query embedding.")
        return None
        
    # Check if the query text is valid
    if not query_text or not isinstance(query_text, str):
        print("Error: Invalid query text provided.")
        return None

    try:
        # Log the query being processed (truncated for brevity)
        print(f"Generating embedding for query: \"{query_text[:50]}...\"")
        
        # Call Voyage AI API - use input_type="query" for optimal search performance
        result = vo.embed(
            [query_text], # API expects a list of texts, even for a single query
            model=model_name,
            input_type="query", # Specify 'query' type
            truncation=True     # Allow truncation if query is too long
        )

        # Extract the single embedding from the result list
        query_embedding = result.embeddings[0]

        # Convert to NumPy array (float32) and ensure it's 2D for normalization
        # FAISS normalize_L2 expects a 2D array (n_vectors, dim)
        embedding_np = np.array([query_embedding]).astype('float32')

        # Normalize the embedding vector (inplace operation)
        # This is crucial for using cosine similarity with the IndexFlatIP index.
        print("Normalizing query embedding...")
        faiss.normalize_L2(embedding_np)
        print("Query embedding normalized.")

        # Return the normalized 1D embedding vector
        return embedding_np[0]

    except voyageai.error.VoyageError as e:
        # Handle specific Voyage AI API errors
        print(f"\nVoyage AI API error during query embedding: {e}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during the process
        print(f"\nAn unexpected error occurred during query embedding: {e}")
        return None

# --- Main Execution (for testing purposes) ---
if __name__ == "__main__":
    # This block executes only when the script is run directly (python query_processing.py)
    
    # Define a sample query for testing
    sample_query = "patient reported symptoms of high blood pressure"
    print(f"\n--- Testing get_query_embedding function ---")
    
    # Call the function to get the embedding
    query_vector = get_query_embedding(sample_query)

    # Check if the embedding was generated successfully
    if query_vector is not None:
        print(f"\nSuccessfully generated and normalized embedding for query: \"{sample_query}\"")
        print(f"Embedding Dimension: {query_vector.shape[0]}")
        # Optionally, print the first few values of the embedding vector
        print(f"Sample Embedding values (first 5): {query_vector[:5]}")
    else:
        # Notify if the embedding generation failed
        print(f"\nFailed to generate embedding for query: \"{sample_query}\"")

    # Example of testing with an invalid query
    print(f"\n--- Testing with empty query ---")
    get_query_embedding("") 