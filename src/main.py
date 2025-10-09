from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="CRED Score Predictor (v2)")

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    upvote_ratio: float
    num_comments: int
    post_length: int
    text_word_count: int
    flair: str

@app.post("/predict")
def predict(data: InputData):
    sample = pd.DataFrame([{
        "Upvote Ratio": data.upvote_ratio,
        "Number of Comments": data.num_comments,
        "Post Length": data.post_length,
        "Text Word Count": data.text_word_count,
        "Flair": data.flair
    }])
    pred = model.predict(sample)
    return {"predicted_score": float(pred[0])}
