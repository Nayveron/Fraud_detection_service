import chromadb
import joblib

class FraudDetectionSystem:
    def __init__(self, db_path="./fraud_indexed_db"):
        # Connect to the existing database
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="real_transactions")
        
        # Load the saved scaler
        self.scaler = joblib.load('scaler.pkl')

    def predict(self, raw_data, n_results=3):
        vector = self.scaler.transform([raw_data])
        results = self.collection.query(
            query_embeddings=vector.tolist(),
            n_results=n_results,
            include=["metadatas", "distances"]
        )
    
        final_neighbors = []
        for i in range(len(results['distances'][0])):
            # Get ID and other metadata
            neighbor = results['metadatas'][0][i]
            neighbor['id'] = results['ids'][0][i] # Return the ID back!
            neighbor['distance'] = round(results['distances'][0][i], 5)
        
        # You can keep the filter, but make it softer, for example 1.5
        # Or remove it completely to see what distances actually come in
        if neighbor['distance'] < 2.0: 
            final_neighbors.append(neighbor)
        return final_neighbors