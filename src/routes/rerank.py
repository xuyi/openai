import uuid
from fastapi import APIRouter

from ..models import get_model
from ..models.rerank import RerankModel
from ..utils.request import raise_if_invalid_model

from ..type import (
    RerankRequest,
    RerankResponse,
    CohereRerankResult,
    CohereRerankMeta,
    CohereRerankMetaApiVersion,
    CohereRerankMetaBilledUnits,
    CohereRerankResultDocument,
)

rerank_router = APIRouter(prefix="/rerank")

# URL like cohere api
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-large"


@rerank_router.post("")
async def create_rerank(request: RerankRequest, model_name: str = None):
    if request.model is None:
        request.model = model_name

    rerank_model = get_model(request.model)
    raise_if_invalid_model(rerank_model, RerankModel)

    inputs = _process_inputs(request)

    scores = rerank_model.compute_score(inputs)
    # scores is  list
    sorted_scores = sorted(
        zip(range(len(scores)), scores, request.documents),
        key=lambda x: x[1],
        reverse=True,
    )[0 : request.top_n]
    results = []

    for row in sorted_scores:
        results.append(
            CohereRerankResult(
                document=(
                    None
                    if not request.return_documents
                    else CohereRerankResultDocument(text=row[2])
                ),
                index=row[0],
                relevance_score=row[1],
            )
        )

    return RerankResponse(
        id=str(uuid.uuid4()),
        results=results,
        meta=CohereRerankMeta(
            api_version=CohereRerankMetaApiVersion(
                version="1", is_deprecated=None, is_experimental=None
            ),
            billed_units=CohereRerankMetaBilledUnits(
                input_tokens=None,
                output_tokens=None,
                search_units=1,
                classifications=None,
            ),
            tokens=None,
            warnings=None,
        ),
    )


def _process_inputs(request: RerankRequest):
    documents = request.documents
    query = request.query

    inputs = []
    for doc in documents:
        inputs.append([query, doc])

    return inputs
