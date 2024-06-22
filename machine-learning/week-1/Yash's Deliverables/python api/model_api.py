import pickle
from fastapi import FastAPI
import uvicorn
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np
app = FastAPI()

@app.get("/")
async def root():
    response = {"message": "Hello World"}
    return response["message"]

with open('C:\MLSoC\creditcard_fraud_lgb(1).pkl', 'rb') as f:
  model_12 = pickle.load(f)
@app.get("/predict")
async def predict(input_data: dict):
    input_data = pd.DataFrame(input_data)
    amount = input_data['Amount'] 
    input_data_num = input_data.drop(['Name', 'Amount'], axis=1) 
    pca = PCA(n_components=1)
    input_data_pca = pca.fit_transform(input_data_num)
    input_data_final = np.column_stack((input_data_pca, amount)) 
    prediction = model_12.predict(input_data_final)
    if prediction[0] == 1:
      result = "Fraud"
    else:
      result = "Not Fraud"
    return { 'prediction' : result}

if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=3000)
