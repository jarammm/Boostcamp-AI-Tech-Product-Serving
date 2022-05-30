from fastapi import APIRouter

user_router = APIRouter(prefix="/users")


@user_router.get("/", tags=["users"])
def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]


@user_router.get("/me", tags=["users"])
def read_user_me():
    return {"username": "fakecurrentuser"}


@user_router.get("/{username}", tags=["users"])
def read_user(username: str):
    return {"username": username}