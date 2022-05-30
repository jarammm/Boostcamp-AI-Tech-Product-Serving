from fastapi import FastAPI
import uvicorn
from user import user_router
from order import order_router

app = FastAPI()

if __name__ == '__main__':
    app.include_router(user_router)
    app.include_router(order_router)
    uvicorn.run(app, host="0.0.0.0", port=30002)