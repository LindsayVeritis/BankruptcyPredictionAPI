# Bankruptcy Prediction API

This project is a Bankruptcy Prediction API built using FastAPI and Docker. It leverages a machine learning model to predict bankruptcy based on financial ratios.

## Features
- Predicts bankruptcy based on financial metrics
- Deployed as a Docker container
- Simple and efficient API interface

## Users
There are two types of users who may interact with this API:
1. **Data Scientists/Engineers:** They build and maintain the model and API, and are responsible for deployment and monitoring.
2. **End Users (e.g., Financial Analysts or Business Users):** They interact with the frontend or directly with the API to get bankruptcy predictions.

## Interaction Workflow
Users interact with the system in the following way:
1. **User Input:** End users enter financial data through the Streamlit frontend or via an API call.
2. **Data Processing:** The API processes the input data and scales it using a pre-trained scaler.
3. **Model Prediction:** The data is passed through a logistic regression model to predict the bankruptcy probability.
4. **Result Display:** The prediction and probability are displayed to the user returned as JSON via API.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LindsayVeritis/BankBankruptcyPredictionAPI.git
   cd bankruptcy-prediction-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API Locally
1. Build the Docker image:
   ```bash
   docker build -t bankruptcy-prediction-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 bankruptcy-prediction-api
   ```

3. Test the API:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample.json
   ```

## Streamlit Frontend
To run the Streamlit app locally:
```bash
streamlit run app.py
```

## Cloud Deployment (GCP)
1. Authenticate with GCP:
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   ```

2. Build the Docker image:
   ```bash
   docker build -t bankruptcy-prediction-api .
   ```

3. Tag the image for Google Container Registry (GCR):
   ```bash
   docker tag bankruptcy-prediction-api gcr.io/your-project-id/bankruptcy-prediction-api
   ```

4. Push the Docker image to GCR:
   ```bash
   docker push gcr.io/your-project-id/bankruptcy-prediction-api
   ```

5. Deploy the container using Google Cloud Run:
   ```bash
   gcloud run deploy bankruptcy-prediction-api \
       --image gcr.io/your-project-id/bankruptcy-prediction-api \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --port 8000
   ```

6. Monitor the deployment:
   ```bash
   gcloud run services describe bankruptcy-prediction-api --platform managed
   ```

7. Access the deployed API through the generated URL from Cloud Run.
Send a POST request to the deployed endpoint:
```bash
curl -X POST "https://your-cloud-run-url/predict" -H "Content-Type: application/json" -d @sample.json
```
