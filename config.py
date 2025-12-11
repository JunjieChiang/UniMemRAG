from dataclasses import dataclass
from qdrant_client.http import models as qmodels
from typing import Optional

@dataclass
class Config:
    qdrant_host: str = "localhost"
    qdrant_port: int = 6336
    qdrant_grpc_port: int = 6337
    prefer_grpc: bool = False
    collection: str = "mm_rag_clip"

    distance: qmodels.Distance = qmodels.Distance.COSINE
    on_disk: bool = True
    batch_size: int = 1024

    top_k: int = 5
    timeout_sec = 1200
