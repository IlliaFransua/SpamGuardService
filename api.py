
import os
import sys
from fastapi import FastAPI, Response
from pydantic import BaseModel
from spam_detection_model import SpamDetectionModel

_model = SpamDetectionModel()

def is_spam(text: str) -> bool:
    return _model.predict(text)

app = FastAPI(title="SpamGuardServiceAPI")

class Req(BaseModel):
    text: str

class Resp(BaseModel):
    result: bool

@app.post("/is_spam", response_model=Resp)
def endpoint_is_spam(req: Req, response: Response):
    result = is_spam(req.text)
    response.headers["SpamResult"] = str(result).lower()
    return Resp(result=result)

