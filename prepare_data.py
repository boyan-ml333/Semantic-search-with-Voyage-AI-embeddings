import pandas as pd
import re
import json
import os

# Define the input CSV file path
# Use the full path provided by the user
csv_file_path = r'C:\Users\Boyan\Documents\Code\NLP\semantic_clustering_NIH_CDE\data\cde_all.csv'
# Define the output JSON file path, reflecting the input file name
json_output_path = 'cleaned_cde_all.json'

def clean_text(text):
    """
    Cleans the input text by removing HTML tags and extra whitespace.
    
    Args:
        text (str): The text string to clean.
        
    Returns:
        str: The cleaned text string, or None if the input is not a string.
    """
    if not isinstance(text, str):
        # Return None or an empty string if the input is not text (e.g., NaN)
        return None 
        
    # Remove HTML tags using regex
    # This is a simple regex; for complex HTML, BeautifulSoup might be better
    text = re.sub(r'<[^>]+>', '', text) 
    
    # Remove extra whitespace (leading/trailing, multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip() 
    
    return text

def prepare_cde_data(input_csv, output_json):
    """
    Loads CDE data from a CSV, cleans the 'Question Texts' column, 
    and saves the result as a JSON file.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_json (str): Path to the output JSON file.
    """
    print(f"Loading data from {input_csv}...")
    try:
        # Load the CSV file using pandas
        df = pd.read_csv(input_csv)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        print(f"Current working directory: {os.getcwd()}")
        # If running in a subdirectory, adjust the path like:
        # csv_file_path = '../cde_sample.csv' 
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Verify the required column exists
    if 'Question Texts' not in df.columns:
        print("Error: 'Question Texts' column not found in the CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return
        
    print("Cleaning 'Question Texts' column...")
    # Apply the cleaning function to the 'Question Texts' column
    # Create a new column 'cleaned_text' to store the results
    df['cleaned_text'] = df['Question Texts'].apply(clean_text)

    # Handle rows where cleaning resulted in None or empty string
    original_count = len(df)
    # Drop rows where 'cleaned_text' is null or empty
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'] != ''] 
    cleaned_count = len(df)
    print(f"Removed {original_count - cleaned_count} rows with missing or empty 'Question Texts'.")
    print(f"Proceeding with {cleaned_count} valid CDE entries.")

    # Prepare the data in the desired format: list of dictionaries
    # Use the DataFrame index as the 'id'
    output_data = []
    for index, row in df.iterrows():
        output_data.append({
            "id": index,  # Using DataFrame index as CDE ID
            "text": row['cleaned_text'] 
        })

    print(f"Saving cleaned data to {output_json}...")
    try:
        # Save the list of dictionaries to a JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print("Successfully saved cleaned data.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Check if the CSV file exists before starting
    if not os.path.exists(csv_file_path):
         print(f"Error: The file {csv_file_path} was not found.")
         print(f"Please ensure the file is in the correct directory: {os.getcwd()}")
    else:
        prepare_cde_data(csv_file_path, json_output_path) 