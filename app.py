from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Define input data model
class InputData(BaseModel):
    DeductibleAmtPaid: float
    NumClaimsPerBene: int
    NumDaysToSettle: int
    NumDiagnosisCodes: int
    NumProcedureCodes: int
    Gender: int
    Race: int
    State: int
    County: int
    ChronicCond_Alzheimer: int
    ChronicCond_ObstrPulmonary: int
    ChronicCond_Depression: int
    ChronicCond_Osteoporasis: int
    ChronicCond_rheumatoidarthritis: int
    ChronicCond_stroke: int
    IPAnnualReimbursementAmt: int
    IPAnnualDeductibleAmt: int
    OPAnnualReimbursementAmt: int
    OPAnnualDeductibleAmt: int
    TotalChronicCond: int

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(input_data: InputData): 
    features = [
        input_data.DeductibleAmtPaid,
        input_data.NumClaimsPerBene,
        input_data.NumDaysToSettle,
        input_data.NumDiagnosisCodes,
        input_data.NumProcedureCodes,
        input_data.Gender,
        input_data.Race,
        input_data.State,
        input_data.County,
        input_data.ChronicCond_Alzheimer,
        input_data.ChronicCond_ObstrPulmonary,
        input_data.ChronicCond_Depression,
        input_data.ChronicCond_Osteoporasis,
        input_data.ChronicCond_rheumatoidarthritis,
        input_data.ChronicCond_stroke,
        input_data.IPAnnualReimbursementAmt,
        input_data.IPAnnualDeductibleAmt,
        input_data.OPAnnualReimbursementAmt,
        input_data.OPAnnualDeductibleAmt,
        input_data.TotalChronicCond
    ]
    # Make prediction using the loaded model
    prediction = model.predict([features])[0]
    return {"prediction": prediction}

# Run the FastAPI server with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
