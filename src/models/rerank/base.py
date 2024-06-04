from FlagEmbedding import FlagReranker
from typing import Any, Optional

from src.utils.env import compose_model_id
from ..base import Model


class RerankModel(Model):
    encode_args: dict = {}
    model: Any

    def __init__(self, model: str, name: Optional[str]=None, owner: Optional[str]=None, **kwargs):
        super().__init__(model, name=name, owner=owner)
        self.encode_args = kwargs
    
    def load(self):
        model_id = compose_model_id(self.id, prefix=self.org)
        print(f"Loading model {model_id}")
        self.model = FlagReranker(model_id, use_fp16=True)
        print(f"Model {model_id} loaded!")


    def compute_score(self, sentences):
        return self.model.compute_score(sentences, **self.encode_args)
