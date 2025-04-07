import json
import os
import numpy as np
import time 

# Import functions from our previous steps
from query_processing import get_query_embedding
from search import search_similar_cdes # Ensure search.py loads index/ids correctly

# --- Configuration ---
# List of test queries to evaluate the MVP
TEST_QUERIES = [
    "patient reported symptoms of high blood pressure",
    "assessment scale for patient pain level",
    "demographic information including age and gender",
    "measurement of body weight",
    "history of diabetes mellitus"
]
# Number of top results to retrieve for each query
NUM_RESULTS_K = 5
# Path to the cleaned CDE data (needed for retrieving text descriptions)
CLEANED_DATA_PATH = "cleaned_cde_all.json"
# Output file to save the test results
OUTPUT_FILE_PATH = "test_results.txt"

# --- Helper Function ---
def load_cleaned_data_dict(path):
    """Loads the cleaned CDE data and returns a dictionary {id: text}."""
    print(f"Loading cleaned data from {path} for testing...")
    if not os.path.exists(path):
        print(f"Error: Cleaned data file not found at {path}. Cannot retrieve text.")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            data_dict = {item['id']: item['text'] for item in data_list}
        print(f"Successfully loaded text for {len(data_dict)} CDEs.")
        return data_dict
    except Exception as e:
        print(f"Error loading cleaned data: {e}")
        return None

# --- Main Testing Logic ---
def run_mvp_tests():
    """Runs test queries against the search system and writes results to a file."""
    # Determine the number of queries upfront for delay logic
    num_queries = len(TEST_QUERIES)
    print(f"--- Starting MVP Test Run --- K={NUM_RESULTS_K} ({num_queries} queries) ---")
    
    # Load CDE text data
    cde_data_dict = load_cleaned_data_dict(CLEANED_DATA_PATH)
    if cde_data_dict is None:
        print("Aborting tests due to missing cleaned data.")
        return

    # Open the output file
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("NIH CDE Semantic Search MVP Test Results\n")
            f.write("=========================================\n")
            f.write(f"(Retrieving Top {NUM_RESULTS_K} results per query)\n\n")

            # Process each test query
            for i, query in enumerate(TEST_QUERIES, 1):
                print(f"\nProcessing Test Query {i}/{num_queries}: \"{query}\"")
                f.write(f"--- Query {i}: {query} ---\n")

                # 1. Get Query Embedding
                query_vector = get_query_embedding(query)

                if query_vector is not None:
                    # 2. Perform Similarity Search
                    search_results = search_similar_cdes(query_vector, k=NUM_RESULTS_K)

                    # 3. Write Results to File
                    if search_results:
                        f.write(f"Found {len(search_results)} results:\n")
                        for rank, (cde_id, score) in enumerate(search_results, 1):
                            cde_text = cde_data_dict.get(cde_id, "*Text not found*")
                            f.write(f"  {rank}. ID: {cde_id} (Score: {score:.4f})\n")
                            f.write(f"     Text: {cde_text}\n") # Indent text for readability
                        f.write("\n") # Add space between results
                    else:
                        f.write("  *Search returned no results.*\n\n")
                else:
                    f.write("  *Failed to generate query embedding for this query.*\n\n")
                
                print(f"Finished processing query {i}.")
                
                # Add delay between queries to respect API rate limits (except after the last query)
                if i < num_queries:
                    delay_seconds = 20
                    print(f"Waiting for {delay_seconds} seconds before next query to respect API limits...")
                    time.sleep(delay_seconds)

        print(f"\n--- Test Run Completed --- Results saved to {OUTPUT_FILE_PATH}")
        print("Please review the contents of test_results.txt to assess semantic relevance.")

    except Exception as e:
        print(f"An error occurred while writing the test results file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure .env is loaded if query_processing relies on it being loaded implicitly
    from dotenv import load_dotenv
    load_dotenv() # Load .env from the directory where the script is run
    
    run_mvp_tests() 