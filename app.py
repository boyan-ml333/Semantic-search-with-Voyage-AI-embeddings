import streamlit as st
import json
import os
import numpy as np

# Try to import authenticator, but don't fail if it's not available
try:
    import streamlit_authenticator as authenticator
    AUTHENTICATOR_AVAILABLE = True
except ImportError:
    AUTHENTICATOR_AVAILABLE = False
    st.warning("streamlit-authenticator package not installed. Authentication is disabled. Run 'pip install streamlit-authenticator==0.2.3' to enable.")

# Import functions from our previous steps
from query_processing import get_query_embedding
# Note: search.py loads the index and IDs when imported.
# Ensure search.py can find its required files (index, ids) relative to its own location.
from search import search_similar_cdes

# --- Configuration ---
# Path to the cleaned CDE data JSON file (needed to display text content)
# Assumes this script (app.py) is located in the semantic_clustering_NIH_CDE directory
CLEANED_DATA_PATH = "cleaned_cde_all.json"
DEFAULT_K = 10 # Default number of results to show

# --- Authentication ---
authentication_status = True  # Default if no authentication

if AUTHENTICATOR_AVAILABLE:
    # Get credentials from secrets
    try:
        credentials = {
            "usernames": st.secrets["authentication"]["usernames"]
        } if "authentication" in st.secrets else None

        # Create the authenticator
        if credentials:
            authenticator = authenticator.Authenticate(
                credentials,
                "nih_cde_search",
                "auth_key",
                cookie_expiry_days=30
            )
            
            # Generate login UI
            name, authentication_status, username = authenticator.login("Login", "main")
            
            # Redirect users based on authentication status
            if authentication_status == False:
                st.error("Username/password is incorrect")
                st.stop()
            elif authentication_status == None:
                st.warning("Please enter your username and password")
                st.stop()
    except Exception as e:
        st.warning(f"Authentication error: {e}. Proceeding without authentication.")
        authentication_status = True

# --- Helper Function with Caching ---
# Cache the loading of the cleaned data to avoid reloading on every interaction.
@st.cache_data # Use the correct decorator for caching data
def load_cleaned_data(path):
    """Loads the cleaned CDE data (list of dicts) from JSON.
       Returns a dictionary mapping ID to text for faster lookup.
    """
    print(f"Attempting to load cleaned data from: {path}") # Debug print
    if not os.path.exists(path):
        st.error(f"Error: Cleaned data file not found at {path}. Make sure it's in the same directory as app.py. Cannot display CDE text.")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            # Convert list of dicts to a dict {id: text} for efficient lookup
            data_dict = {item['id']: item['text'] for item in data_list}
        print(f"Successfully loaded cleaned data for {len(data_dict)} CDEs.") # Debug print
        return data_dict
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {path}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading cleaned data: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="CDE Semantic Search", layout="wide") # Configure page settings
st.title("NIH CDE Semantic Search MVP")
st.write("Enter a query to find semantically similar Common Data Elements using Voyage AI embeddings and FAISS.")

# User logout if authenticated
if AUTHENTICATOR_AVAILABLE and 'credentials' in locals() and credentials and authentication_status:
    try:
        authenticator.logout("Logout", "sidebar")
        st.sidebar.write(f"Welcome, {name}")
    except:
        pass

# Load the CDE text data once
# This should trigger only on the first run or if the file changes due to caching
cde_data_dict = load_cleaned_data(CLEANED_DATA_PATH)

# Check if data loading failed before proceeding
if cde_data_dict is None:
    st.stop() # Stop execution if data isn't available

# User Input: Query Text
query = st.text_input("Search Query:", placeholder="e.g., patient vital signs, blood pressure measurement, pain assessment scale")

# User Input: Number of results (k)
k = st.number_input("Number of results to display:", min_value=1, max_value=50, value=DEFAULT_K, step=1)

# --- Search Logic and Display ---
if query: # Only run search if query is not empty
    st.divider() # Add a visual separator
    st.subheader("Search Results")
    
    # 1. Get Query Embedding
    # Use a spinner to indicate processing
    with st.spinner(f'Generating embedding for "{query[:30]}..." using Voyage AI...'):
        query_vector = get_query_embedding(query)

    # Check if embedding generation was successful
    if query_vector is not None:
        # st.success("Query embedding generated.") # Can be a bit verbose, removed for cleaner UI
        
        # 2. Perform Similarity Search
        # Use a spinner to indicate search is in progress
        with st.spinner(f"Searching for top {k} similar CDEs using FAISS..."):
            search_results = search_similar_cdes(query_vector, k=k)

        # 3. Display Results
        if search_results:
            st.success(f"Found {len(search_results)} results matching your query.")
            
            # Prepare data for the table
            table_data = []
            for rank, (cde_id, score) in enumerate(search_results, 1):
                # Retrieve the text from the loaded dictionary
                cde_text = cde_data_dict.get(cde_id, "*Text not found in loaded data.*")
                table_data.append({
                    "Rank": rank,
                    "Similarity Score": f"{score:.4f}", # Format score
                    "CDE Text": cde_text,
                    "CDE ID": cde_id
                })
            
            # Define the desired column order for display
            column_order = ["Rank", "CDE Text", "Similarity Score", "CDE ID"]
            
            # Display data in a dataframe table
            # Specify column_order to ensure desired layout.
            # hide_index=True removes the default numerical index.
            st.dataframe(
                table_data, 
                column_order=column_order, 
                use_container_width=True, # Use container width for better layout
                hide_index=True # Hide the default dataframe index
            )

        else:
            # Handle case where search returns no results
            st.warning("No similar CDEs found for this query in the index.")
    else:
        # Handle case where embedding generation failed
        st.error("Failed to generate query embedding. Please check the API key and connection, and try again.")
else:
    # Show a gentle prompt if no query is entered yet
    st.info("Please enter a query above to start the search.") 