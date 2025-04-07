# NIH CDE Semantic Search

A semantic search tool for NIH Common Data Elements (CDE) using Voyage AI embeddings and FAISS similarity search.

## Project Overview

This MVP enables searching the NIH CDE repository (24,000+ elements) using natural language queries. The system generates embeddings with Voyage AI's voyage-large-2 model and performs fast similarity search via FAISS indexing.

## Core Components

- **prepare_data.py**: Cleans CDE data from CSV format
- **generate_embeddings.py**: Creates vector embeddings using Voyage AI API
- **index_embeddings.py**: Builds FAISS index for fast similarity search
- **query_processing.py**: Processes user queries into embeddings
- **search.py**: Performs similarity search using FAISS
- **app.py**: Streamlit web interface for the search tool
- **test_mvp.py**: Test script for validating search relevance

## Technical Details

- **Data Flow**: CSV → JSON → Embeddings → FAISS Index → Search Results
- **Embedding Model**: voyage-large-2 (Voyage AI)
- **Search Index**: FAISS with cosine similarity (IndexFlatIP)
- **UI**: Streamlit table display with Rank, CDE Text, Similarity Score, CDE ID
- **Security**: Optional user authentication via streamlit-authenticator

## Rate Limiting

- Voyage AI API free plan has a 3 requests per minute (RPM) limit
- Implementation: 20-second delays between batch API calls
- Batch Size: Configurable, default 128 texts per batch

## Dependencies

- voyageai
- faiss-cpu
- streamlit 
- python-dotenv
- numpy
- pandas
- streamlit-authenticator

## Setup and Usage

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Voyage AI API key:
   ```
   VOYAGE_API_KEY=your_api_key_here
   ```
4. (Optional) Configure authentication:
   - Create `.streamlit/secrets.toml` following the template
   - Update usernames and passwords

5. Run the Streamlit app: `streamlit run app.py`

## Deploying with Streamlit

### Local Sharing (Development)
```
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

### Streamlit Community Cloud (Recommended)
1. Push your code to GitHub (excluding secrets and large data files)
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Add your secrets via the Streamlit Cloud dashboard
5. Deploy the app

### Docker Deployment
For more controlled environments, use the provided Dockerfile:
```
docker build -t nih-cde-search .
docker run -p 8501:8501 nih-cde-search
```

## Data Processing Workflow

1. **Data Preparation**: `python prepare_data.py`
2. **Embedding Generation**: `python generate_embeddings.py`
3. **Index Creation**: `python index_embeddings.py`
4. **Testing**: `python test_mvp.py`
5. **Launch UI**: `streamlit run app.py`

## Memory Management

The system handles large embedding files (90MB+) with proper memory cleanup and error handling throughout the pipeline. 
