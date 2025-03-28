from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('model/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler used during training
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the top 15 features from the file and sanitize them
top_features = pd.read_csv("model/top_15_features.csv")['Feature'].tolist()

# Sanitize feature names to be valid Python identifiers
cleaned_features = [
    feature.replace(" ", "_").replace("%", "pct").replace("/", "_")
    .replace("(", "").replace(")", "").replace("¥", "").replace("-", "_")
    for feature in top_features
]

# Map cleaned feature names to original names
feature_map = dict(zip(cleaned_features, top_features))

# Dynamically define the Pydantic model with cleaned feature names
class BankruptcyPredictionInput(BaseModel):
    Equity_to_Liability: float = Field(..., ge=0, description="Ratio of equity to liability")
    Debt_ratio_pct: float = Field(..., ge=0, le=0, description="Debt ratio as a percentage")
    Per_Share_Net_profit_before_tax_Yuan_: float = Field(..., description="Per share net profit before tax in Yuan")
    Accounts_Receivable_Turnover: float = Field(..., ge=0, description="Turnover ratio of accounts receivable")
    Total_debt_Total_net_worth: float = Field(..., ge=0, description="Total debt to total net worth ratio")
    Cash_Flow_to_Liability: float = Field(..., ge=0, description="Cash flow to liability ratio")
    Operating_Profit_Rate: float = Field(..., ge=0, description="Operating profit rate")
    Net_Income_to_Total_Assets: float = Field(..., ge=0, description="Net income to total assets ratio")
    Cash_Total_Assets: float = Field(..., ge=0, description="Cash to total assets ratio")
    Inventory_and_accounts_receivable_Net_value: float = Field(..., ge=0, description="Ratio of inventory and accounts receivable to net value")
    Contingent_liabilities_Net_worth: float = Field(..., ge=0, description="Contingent liabilities to net worth ratio")
    Operating_profit_Paid_in_capital: float = Field(..., ge=0, description="Operating profit to paid-in capital ratio")
    Non_industry_income_and_expenditure_revenue: float = Field(..., ge=0, description="Non-industry income and expenditure as a ratio of revenue")
    Current_Liability_to_Liability: float = Field(..., ge=0, description="Current liability to total liability ratio")
    Current_Liability_to_Equity: float = Field(..., ge=0, description="Current liability to equity ratio")

class BankruptcyPredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted bankruptcy status (0 = Not Bankrupt, 1 = Bankrupt)")
    probability: float = Field(..., ge=0, le=1, description="Probability of bankruptcy")
    input_values: dict = Field(..., description="Input values used for prediction")


# Initialize the FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Bankruptcy Prediction API!"}

@app.post("/predict", response_model=BankruptcyPredictionResponse)
async def predict(input_data: BankruptcyPredictionInput):
    try:
        # Log received input data
        print("Received input data:", input_data)
        input_dict = input_data.dict()
        print("Input values:", input_dict)

        # Map cleaned features to original features
        original_features = [feature_map[cleaned_feature] for cleaned_feature in cleaned_features]
        original_input_dict = {feature_map[cleaned_feature]: value for cleaned_feature, value in input_dict.items()}

        # Convert input data to a DataFrame using the original feature names
        X = pd.DataFrame([original_input_dict])

        # Scale the input data
        print("Features before scaling:", X)
        X_scaled = scaler.transform(X)
        print("Scaled features:", X_scaled)

        # Make prediction using the scaled data
        prediction = model.predict(X_scaled)[0]
        print("Prediction:", prediction)
        probability = model.predict_proba(X_scaled)[0][1]
        print("Probability:", probability)

        return BankruptcyPredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            input_values=input_dict
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
