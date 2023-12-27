from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.models import DSSM, AutoEncoder, LightFM, MultiVae, TopPopular, TwoStageModel, UserKNN


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()

# Инициализация моделей
user_knn = UserKNN()
top_popular_model = TopPopular()
lightfm = LightFM()
dssm_model = DSSM()
autoencoder = AutoEncoder()
mvae = MultiVae()
two_stage_model = TwoStageModel()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name == "top_popular":
        recs = top_popular_model.recommend(user_id)
    elif model_name == "user_knn":
        recs = user_knn.recommend(user_id)
    elif model_name == "lightfm":
        recs = lightfm.recommend(user_id)
    elif model_name == "dssm":
        recs = dssm_model.recommend(user_id)
    elif model_name == "ae":
        recs = autoencoder.recommend(user_id)
    elif model_name == "multi_vae":
        recs = mvae.recommend(user_id)
    elif model_name == "two_stage":
        recs = two_stage_model.recommend(user_id)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    return RecoResponse(user_id=user_id, items=recs)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
