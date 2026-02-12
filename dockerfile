# 1. Use the official Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependencies file
COPY requirements.txt .

# 4. Install the required libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code and our artifacts (database and scaler)
COPY logic.py .
COPY main.py .
COPY scaler.pkl .
COPY fraud_indexed_db/ ./fraud_indexed_db/

# 6. Expose port 8000
EXPOSE 8000

# 7. Command to run our API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]