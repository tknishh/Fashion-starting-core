from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from utils import load_lace_data
from model import get_recommendation, get_colors
import os
import dotenv
from io import BytesIO

dotenv.load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Load lace data once at startup
laces_data = load_lace_data('data/laces', 'data/descriptions.txt')

class RecommendationRequest(BaseModel):
    user_input: str

class ColorResponse(BaseModel):
    name: str
    code: str

@app.post("/recommend_lace")
async def recommend_lace(request: RecommendationRequest):
    try:
        recommended_laces = get_recommendation(request.user_input, laces_data, API_KEY)
        if recommended_laces:
            return JSONResponse(content=recommended_laces)
        else:
            raise HTTPException(status_code=404, detail="No recommendations found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/find_colors", response_model=List[ColorResponse])
async def find_colors(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        image_stream = BytesIO(image_bytes)
        colors = get_colors(image_stream, API_KEY)
        if colors:
            return colors
        else:
            raise HTTPException(status_code=404, detail="No colors identified.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")