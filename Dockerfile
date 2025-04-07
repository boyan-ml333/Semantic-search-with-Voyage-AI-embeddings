FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY *.py .
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY README.md .

# Create directory for data
RUN mkdir -p data

# Copy example secrets file (will be overridden by mounted volume in production)
COPY .streamlit/secrets.toml.example ./.streamlit/secrets.toml

# Expose Streamlit port
EXPOSE 8501

# Command to run app when container starts
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 