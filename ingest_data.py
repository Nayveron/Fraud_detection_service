import os
import pandas as pd
import kagglehub
import chromadb
import joblib
from sklearn.preprocessing import MinMaxScaler

def run_ingestion():
    # 1. Download the data (it remains in the kagglehub cache)
    print("Отримання даних з Kaggle...")
    kaggle_path = kagglehub.dataset_download("ismetsemedov/transactions")
    csv_file = os.path.join(kaggle_path, os.listdir(kaggle_path)[0])

    # 2. Read the data (nrows for speed, as you wanted)
    needed_columns = ['amount', 'transaction_hour', 'distance_from_home', 'is_fraud', 'country']
    df = pd.read_csv(csv_file, usecols=needed_columns, nrows=500000)
    
    # 3. Prepare the database and scaler
    client = chromadb.PersistentClient(path="./fraud_indexed_db")
    collection = client.get_or_create_collection(name="real_transactions")
    scaler = MinMaxScaler()

    features = ['amount', 'transaction_hour', 'distance_from_home']
    scaler.fit(df[features])
    
    # Save the scaler's "memory"
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Скейлер збережено у scaler.pkl")

    # 4. Upload to ChromaDB
    batch_size = 5000 
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        vectors = scaler.transform(chunk[features])
        
        collection.upsert(
            embeddings=vectors.tolist(),
            metadatas=chunk[['is_fraud', 'country']].to_dict(orient='records'),
            ids=[str(idx) for idx in chunk.index]
        )
        
        if i % 50000 == 0:
            print(f"Прогрес: {i} рядків додано...")

    print(f"🚀 Готово! База створена, у ній {collection.count()} записів.")

if __name__ == "__main__":
    run_ingestion()