from fastapi import APIRouter

order_router = APIRouter(prefix="/orders")

@order_router.get("/", tags=["orders"])
def read_orders():
    return [{"order": "Taco"}, {"order": "Burritto"}]


@order_router.get("/me", tags=["orders"])
def read_order_me():
    return {"my_order": "taco"}


@order_router.get("/{order_id}", tags=["orders"])
def read_order_id(order_id: str):
    return {"order_id": order_id}