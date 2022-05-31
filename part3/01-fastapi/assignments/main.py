from unittest import result
from fastapi import FastAPI
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Optional

from assignments.model import MyDALLE, get_model, txt2img

app = FastAPI()

orders = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str


class InferenceTextProduct(Product):
    name: str = "inference_text_product"
    result: Optional[List] # imagearray.tolist


@app.post("/order", description="주문을 요청합니다")
async def make_order(text: str,
                     model: MyDALLE = Depends(get_model)):

    inference_result = txt2img(model=model, text=text)
    product = InferenceTextProduct(result=inference_result.tolist())

    return product[result]
