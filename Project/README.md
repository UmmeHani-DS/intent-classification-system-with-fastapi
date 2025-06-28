# Intent Classification System

This project provides an end-to-end system for classifying user queries into predefined intent categories. It includes a FastAPI backend, a Streamlit frontend, and Docker-based deployment.

## How to Run

### Option 1: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/UmmeHani-DS/intent-classification-system-with-fastapi.git
cd intent-classification-system-with-fastapi
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Start the FastAPI backend:
```bash
uvicorn api.main:app --reload --port 8000
```

6. In a new terminal, start the Streamlit frontend:
```bash
streamlit run frontend/app.py --server.port 8501
```

Visit `http://localhost:8501` in your browser to use the application.

7. (Optional) Run API Tests:
To verify that the FastAPI endpoints are working correctly, run the test suite:
```bash
python -m pytest -v
```

This will execute all tests defined in `test_api.py` and display detailed results.

---

### Option 2: Run with Docker

1. Build the Docker image:
```bash
docker build -t intent-classifier .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 intent-classifier
```

The application will be accessible at `http://localhost:8501`.
