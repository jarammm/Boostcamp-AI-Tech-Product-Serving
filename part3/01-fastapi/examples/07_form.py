from fastapi import FastAPI, Form, Request # Request : request할 때의 객체로, request 데이터를 가지고 있음
from fastapi.templating import Jinja2Templates # python으로 front 구성 시 간단하게 사용 가능

import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='./')


@app.get("/login/")
def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})

# [파이썬 세 개의 점, ELLIPSIS 객체는 무엇인가요?](https://tech.madup.com/python-ellipsis/)

@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)): # ... : required, 필수의 의미
    return {"username": username}


if __name__ == '__main__':
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=30002)