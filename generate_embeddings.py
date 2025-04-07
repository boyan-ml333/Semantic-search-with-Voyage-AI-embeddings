import voyageai
import json
import os
import pickle
import time
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file if it exists
load_dotenv()
# Add a check to see if the .env file was loaded and the key is present
print(f".env file loaded: {load_dotenv(verbose=True)}") # verbose=True provides more feedback
print(f"VOYAGE_API_KEY found after load_dotenv: {'Yes' if os.getenv('VOYAGE_API_KEY') else 'No'}")

# Input JSON file containing cleaned CDE data
input_json_path = 'cleaned_cde_all.json'
# Output pickle file to store embeddings
output_pickle_path = 'cde_embeddings.pkl'
# Voyage AI model name 
# Recommended options: voyage-large-2, voyage-3-lite, voyage-3, voyage-3-large
# See: https://docs.voyageai.com/docs/text-embedding-models
model_name = "voyage-large-2" 
# Batch size for sending requests to the API (max 128 for Voyage)
batch_size = 128 
# Delay between batches (in seconds) to respect potential rate limits
# Set to 20 seconds to stay under the 3 RPM limit (60 seconds / 3 requests = 20 seconds/request)
delay_between_batches = 20 

# --- Helper Functions ---
def load_cleaned_data(json_path):
    """Loads the cleaned CDE data from the specified JSON file."""
    print(f"Loading cleaned data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Expecting a list of dictionaries like [{"id": ..., "text": ...}]
        print(f"Successfully loaded {len(data)} items.")
        return data
    except FileNotFoundError:
        print(f"Error: Input file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        return None

def save_embeddings(embeddings_dict, pickle_path):
    """Saves the embeddings dictionary to a pickle file."""
    print(f"Saving embeddings to {pickle_path}...")
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print("Successfully saved embeddings.")
    except Exception as e:
        print(f"Error saving embeddings to pickle file: {e}")

# --- Main Embedding Generation Logic ---
def generate_embeddings():
    """Generates embeddings for CDE texts using Voyage AI API."""
    # 1. Load API Key
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("Error: VOYAGE_API_KEY environment variable not set.")
        print("Please set the environment variable before running.")
        return

    # 2. Initialize Voyage AI Client
    try:
        vo = voyageai.Client(api_key=api_key)
        print(f"Voyage AI client initialized with model: {model_name}")
    except Exception as e:
        print(f"Error initializing Voyage AI client: {e}")
        return

    # 3. Load Cleaned Data
    cde_data = load_cleaned_data(input_json_path)
    if not cde_data:
        return # Stop if data loading failed
        
    # Prepare lists for batch processing
    all_ids = [item['id'] for item in cde_data]
    all_texts = [item['text'] for item in cde_data]

    # 4. Generate Embeddings in Batches
    embeddings_dict = {}
    total_items = len(all_texts)
    print(f"Starting embedding generation for {total_items} items in batches of {batch_size}...")

    for i in range(0, total_items, batch_size):
        batch_ids = all_ids[i:i + batch_size]
        batch_texts = all_texts[i:i + batch_size]
        
        # Ensure batch is not empty
        if not batch_texts:
            continue
            
        print(f"Processing batch {i // batch_size + 1}/{ (total_items + batch_size - 1) // batch_size } ({len(batch_texts)} items)...", end='', flush=True)
        
        try:
            # Call Voyage AI API
            result = vo.embed(
                batch_texts, 
                model=model_name, 
                input_type="document", # Specify that these are documents to be searched
                truncation=True # Truncate long texts if necessary
            )
            
            # Store embeddings with their IDs
            for doc_id, embedding in zip(batch_ids, result.embeddings):
                embeddings_dict[doc_id] = embedding
                
            # Output the total tokens used in the API call
            # Safely check if usage and total_tokens attributes exist
            if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                print(f" Success. Tokens used: {result.usage.total_tokens}")
            else:
                # If usage info is not available, print a generic success message
                print(f" Success. (Token usage not available)")
            
            # Optional delay between batches
            if delay_between_batches > 0 and (i + batch_size < total_items):
                 time.sleep(delay_between_batches)
                 
        except voyageai.error.VoyageError as e: # Corrected exception type
            print(f"\nError processing batch: {e}")
            # Optionally, decide whether to stop or continue with the next batch
            # For now, we stop on error to avoid potential cascading issues
            print("Stopping embedding generation due to API error.")
            return # Stop processing further batches
        except Exception as e:
            print(f"\nAn unexpected error occurred during embedding: {e}")
            print("Stopping embedding generation due to unexpected error.")
            return # Stop processing further batches
            
    print("Embedding generation completed.")

    # 5. Save Embeddings
    if embeddings_dict:
        save_embeddings(embeddings_dict, output_pickle_path)
    else:
        print("No embeddings were generated.")

# --- Main Execution ---
if __name__ == "__main__":
    generate_embeddings() 