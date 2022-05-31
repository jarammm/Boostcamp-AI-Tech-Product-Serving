if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=30002, reload=True)  # reload=True : 새로운 정보가 입력될 때마다, 변화가 발생할 때마다 다시 로딩됨