from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from controller.model import Model, get_model

app = FastAPI()


class NewsRequest(BaseModel):
    text: str


class NewsResponse(BaseModel):
    sentiment: str
    

@app.post("/multilingual_news_detect", response_model=NewsResponse)
def multilingual_news_detect(request: NewsRequest, model: Model = Depends(get_model)):
    sentiment = model.multilingual_news_detect(request.text)
    return NewsResponse(
        sentiment=sentiment
    )
