from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.model import MyEfficientNet, get_model, get_config, predict_from_image_byte

app = FastAPI()


@app.get("/")
def hello_world():
    return {"hello": "world"}


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # uuid : 고유 식별자, Universally Unique Identifier
    # default_factory : Product Class가 처음 만들어질 때 호출되는 함수를 uuid4로 하겠다 => Product 클래스를 생성하면 uuid4를 만들어서 id에 저장
    name: str
    price: float


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    # 최초에 빈 list를 만들어서 저장한다
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        # add_product는 Product를 인자로 받아서, 해당 id가 이미 존재하는지 체크 => 없다면 products 필드에 추가
        # 업데이트할 때 updated_at을 현재 시각으로 업데이트
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


orders = []
# 실무에서는 보통 이 경우에 데이터베이스를 이용해서 주문을 저장하지만, 데이터베이스를 따로 학습하지 않았으므로 In Memory인 리스트에 저장


@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders


@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    # order_id를 기반으로 order를 가져온다
    order = get_order_by_id(order_id=order_id)
    # 만약 get_order_by_id에서 아무런 데이터가 없다면? 빈 리스트가 나오면?
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id), None)
    # 제네레이터
    # iter, next 키워드로 검색
    # 제네레이터를 사용한 이유 : 메모리를 더 절약해서 사용할 수 있음
    # 이터레이터, 이터러블, 제네레이터 => 파이썬 면접에서 많이 나오는 소재. GIL
    # iter는 반복 가능한 객체에서 이터레이터를 반환
    # next는 이터레이터에서 값을 차례대로 꺼냄


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


@app.post("/order", description="주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MyEfficientNet = Depends(get_model),
                     config: Dict[str, Any] = Depends(get_config)):
    # Depends : 의존성 주입
    # 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # 모델을 Load, Config Load
    # async, Depends 검색해서 또 학습해보기!
    products = []
    for file in files:
        image_bytes = await file.read()
        inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
        # InferenceImageProduct Class 생성해서 product로 정의
        product = InferenceImageProduct(result=inference_result)
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]:
    """
    Order를 업데이트 합니다

    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)

    return updated_order


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)

    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill


# TODO: 주문 구현, 상품 구현, 결제 구현
    # TODO: 주문(Order) = Request
    # TODO: 상품(Product) = 마스크 분류 모델 결과
    # TODO: 결제 = Order.bill
    # 2개의 컴포넌트
# TODO: Order, Product Class 구현
    # TODO: Order의 products 필드로 Product의 List(하나의 주문에 여러 제품이 있을 수 있음)

# TODO: get_orders(GET) : 모든 Order를 가져옴
# TODO: get_order(GET) : order_id를 사용해 Order를 가져옴
# TODO: get_order_by_id : get_order에서 사용할 함수
# TODO: make_order(POST) : model, config를 가져온 후 predict => Order products에 넣고 return
# TODO: update_order(PATCH) : order_id를 사용해 order를 가져온 후, update
# TODO: update_order_by_id : update_order에서 사용할 함수
# TODO: get_bill(GET) : order_id를 사용해 order를 가져온 후, order.bill return
from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from app.model import MyEfficientNet, get_model, get_config, predict_from_image_byte

app = FastAPI()

orders = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    # Field : 모델 스키마 또는 복잡한 Validation 검사를 위해 필드에 대한 추가 정보를 제공할 때 사용
    # uuid : 고유 식별자, Universally Unique IDentifier / uuid4(uuid ver4. 고유 식별자 랜덤 생성 기능)
    # default_factory : Product Class가 처음 만들어질 때 호출되는 함수를 uuid4로 하겠다!는 의미
    # ==> Product 클래스를 생성하면 uuid4를 만들어서 id에 저장
    name: str
    price: float
    # Product(name, price) 같은 식으로 객체 생성


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    # 빈 최초의 리스트를 만들어 저장
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self


class OrderUpdate(BaseModel):
    products: List[Product] = Field(default_factory=list)


class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]


# 실무에서는 보통 이 경우에 데이터베이스를 이용해서 주문을 저장하지만, 데이터베이스를 따로 학습하지 않은 관계로, In Memory인 리스트에 저장할 것입니다.
orders = []

@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders

# order_id를 기반으로 order를 가져온다
@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    # order_id 를 기반으로 order를 가져온다.
    order = get_order_by_id(order_id=order_id)
    # 만약 get_order_by_id에서 아무런 데이터가 없다면? 빈 리스트가 나오면?
    # None을 반환하지 말고 명시적으로 정보가 없다는 사실을 전달하자!
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order


def get_order_by_id(order_id: UUID) -> Optional[Order]:
    # 제너레이터
    # None: 조건에 맞는 값이 없으면 None 반환하겠다는 의미
    # 제네레이터를 사용한 이유 : 메모리를 더 절약해서 사용할 수 있음
    # 이터레이터, 이터러블, 제네레이터 => 파이썬 면접에서 많이 나오는 소재. GIL
    # iter는 반복 가능한 객체에서 이터레이터를 반환
    # next는 이터레이터에서 값을 차례대로 꺼냄
    return next((order for order in orders if order.id == order_id), None)


# async : 비동기
# async와 await를 사용해 비동기를 할 수 있음
@app.post("/order", description="주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...),
                     model: MyEfficientNet = Depends(get_model),
                     config: Dict[str, Any] = Depends(get_config)):
    # Depends : 의존성 주입해주는 함수
    # 반복적이고 공통적인 로직이 필요할 때 사용할 수 있음
    # 모델을 Load, Config Load
    # async, Depends 검색해서 심화학습 추천
    products = []
    for file in files:
        image_bytes = await file.read()
        inference_result = predict_from_image_byte(model=model, image_bytes=image_bytes, config=config)
        product = InferenceImageProduct(result=inference_result)
        products.append(product)

    new_order = Order(products=products)
    orders.append(new_order)
    return new_order


def update_order_by_id(order_id: UUID, order_update: OrderUpdate) -> Optional[Order]:
    """
    Order를 업데이트 합니다

    Args:
        order_id (UUID): order id
        order_update (OrderUpdate): Order Update DTO

    Returns:
        Optional[Order]: 업데이트 된 Order 또는 None
    """
    existing_order = get_order_by_id(order_id=order_id)
    if not existing_order:
        return

    updated_order = existing_order.copy()
    for next_product in order_update.products:
        updated_order = existing_order.add_product(next_product)

    return updated_order


@app.patch("/order/{order_id}", description="주문을 수정합니다")
async def update_order(order_id: UUID, order_update: OrderUpdate):
    updated_order = update_order_by_id(order_id=order_id, order_update=order_update)

    if not updated_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return updated_order


@app.get("/bill/{order_id}", description="계산을 요청합니다")
async def get_bill(order_id: UUID):
    found_order = get_order_by_id(order_id=order_id)
    if not found_order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return found_order.bill

"""
TODO: 주문 구현, 상품 구현, 결제 구현
    TODO: 주문(Order) = Request
    TODO: 상품(Product) = 마스크 분류 모델 결과
    TODO: 결과 = Order.bill
    # 2개의 컴포넌트
TODO: Order, Product Class 구현
    TODO: Order의 products 필드로 Product의 List(하나의 주문에 여러 제품이 있을 수 있음음

TODO: get_orders(GET) : 모든 Order를 가져옴
TODO: get_order(GET) : order_id를 사용해 Order를 가져옴
TODO: get_order_by_id : get_order에서 사용할 함수
TODO: make_order(POST) : model, config를 가져온 후 predict => Order products에 넣고 return
TODO: update_order(PATCH) : order_id를 사용해 order를 가져온 후, update
TODO: update_order_by_id : update_order에서 사용할 함수
TODO: get_bill(GET) : order_id를 사용해 order를 가져온 후, order.bill return
"""