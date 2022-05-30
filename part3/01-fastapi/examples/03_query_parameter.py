from fastapi import FastAPI
import uvicorn

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip: skip + limit]

# example
# 1) {ipaddress}/items/?limit=1 (o)
# 2) {ipaddress}/items/?limit=1/ (x)
# 3) {ipaddress}/items?limit=1 (o) --> 자동으로 1)과 같이 주소가 바뀜
# *3) 실행 시 출력되는 INFO : "GET /items?limit=1 HTTP/1.1" 307 Temporary Redirect

if __name__ == '__main__':
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=30002)