FROM python:3.10-slim

WORKDIR /app

COPY . .

# Install project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Expose both FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Run both backend and frontend when Docker starts
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501"]
