# 🛡️ Hybrid Vector-Based Fraud Detection API

A high-performance microservice for real-time transaction monitoring. The system uses vector similarity search to identify potential fraud patterns by comparing new transactions with a historical dataset of 500,000+ records.

## 🚀 Key Features
* **Real-time Scoring**: Built with **FastAPI** for low-latency inference.
* **Vector Search**: Utilizes **ChromaDB** to find nearest neighbors in high-dimensional space using $L2$ distance.
* **Production Pipeline**: Separate data ingestion script (`ingest_data.py`) and inference service (`main.py`).
* **Containerized**: Fully Dockerized for seamless deployment.
* **Validation**: Robust input validation using **Pydantic** models.

## 🛠️ Tech Stack
* **Backend**: Python 3.12, FastAPI, Uvicorn
* **Database**: ChromaDB (Vector Store)
* **ML/Math**: Scikit-learn (MinMaxScaler), NumPy, Joblib
* **DevOps**: Docker

## ⚙️ Installation & Running

### Using Docker (Recommended)
1. Build the image:
   ```bash
   docker build -t fraud-api .
2. Run the container:
    ```bash
   docker run -p 8000:8000 fraud-api
3. Access interactive API documentation (Swagger UI) at:
    ```bash
   http://localhost:8000/docs

### Manual Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Prepare the database and scaler:
   ```bash 
   python ingest_data.py
4. Start the server:
   ```bash 
   uvicorn main:app --reload
