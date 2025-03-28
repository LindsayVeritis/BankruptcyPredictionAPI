FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the model files
COPY model/logistic_regression_model.pkl ./model/
COPY model/scaler.pkl ./model/
COPY model/top_15_features.csv ./model/

# Copy the application code

COPY . .

# Expose the port for the API
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
