import os

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any, List

from ..utils.logger import get_logger

from .baichuan import HANDLERS as BAICHUAN_HANDLERS
from .chatglm import HANDLERS as CHATGLM_HANDLERS
from .internlm import HANDLERS as INTERNLM_HANDLERS
from .llama import HANDLERS as LLAMA_HANDLERS
from .freewilly import HANDLERS as FREE_WILLY_HANDLERS

logger = get_logger(__name__)

models = {
    "chatglm-6b": CHATGLM_HANDLERS,
    "chatglm2-6b": CHATGLM_HANDLERS,
    "internlm-chat-7b": INTERNLM_HANDLERS,
    "internlm-chat-7b-8k": INTERNLM_HANDLERS,
    "Baichuan-13B-Chat": BAICHUAN_HANDLERS,
    "Llama-2-7b-chat-hf": LLAMA_HANDLERS,
    "Llama-2-13b-chat-hf": LLAMA_HANDLERS,
    "FreeWilly2": FREE_WILLY_HANDLERS,
}

load_dotenv()
LLMS_DISABLED = os.environ.get("LLMS_DISABLED")
if LLMS_DISABLED is not None and LLMS_DISABLED.strip() != "":
    for name in [name.strip() for name in LLMS_DISABLED.split(",")]:
        if name in models:
            del models[name]


class LLM(BaseModel):
    id: str
    tokenizer: Any
    model: Any


def get_models():
    return list(models.keys())


_llms: List[LLM] = []


def get_model(model_id: str):
    global _llms

    if models.get(model_id) is None:
        raise ValueError(f"Model {model_id} not found")

    llm = next((l for l in _llms if l.id == model_id), None)

    if llm is None:
        handlers = models.get(model_id)

        logger.info(f"Loading model {model_id} ...")
        model, tokenizer = handlers.get("load")(model_id)
        logger.info(f"Model {model_id} loaded!")

        llm = LLM(id=model_id, tokenizer=tokenizer, model=model)
        _llms.append(llm)

    return llm.model, llm.tokenizer
