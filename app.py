from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model = joblib.load("house_price_pipeline.pkl")


class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    FullBath: int
    YearBuilt: int


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(data: HouseData):

    input_data = pd.DataFrame([{
        "OverallQual": data.OverallQual,
        "GrLivArea": data.GrLivArea,
        "GarageCars": data.GarageCars,
        "TotalBsmtSF": data.TotalBsmtSF,
        "FullBath": data.FullBath,
        "YearBuilt": data.YearBuilt,

        "SaleCondition": "Normal",
        "BldgType": "1Fam",
        "Utilities": "AllPub",
        "1stFlrSF": data.GrLivArea * 0.6,
        "GarageArea": data.GarageCars * 250,
        "Heating": "GasA",
        "MSZoning": "RL",
        "KitchenQual": "TA",
        "LandSlope": "Gtl",
        "YearRemodAdd": data.YearBuilt,
        "TotRmsAbvGrd": 6
    }])

    prediction = model.predict(input_data)[0]

    return {
        "Predicted House Price": round(float(prediction), 2)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)