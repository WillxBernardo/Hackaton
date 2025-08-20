import os
import base64
import tempfile
from typing import Tuple

from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore


def _resolve_tls_kwargs() -> dict:
    """Monta kwargs de TLS conforme .env."""
    verify = os.getenv("ES_VERIFY", "false").lower() == "true"
    ca_path = os.getenv("ES_CA_PATH")
    ca_b64 = os.getenv("ES_CA_BASE64")

    if not verify:
        # modo inseguro (útil para CA self-signed)
        return {"verify_certs": False, "ssl_show_warn": True}

    if ca_path and os.path.exists(ca_path):
        return {"verify_certs": True, "ca_certs": ca_path}

    if ca_b64:
        pem_tmp = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        try:
            pem_tmp.write(base64.b64decode(ca_b64))
            pem_tmp.flush()
            return {"verify_certs": True, "ca_certs": pem_tmp.name}
        except Exception:
            return {"verify_certs": True}

    # tenta trust store do SO
    return {"verify_certs": True}


def create_es_client() -> Elasticsearch:
    """Cria cliente Elasticsearch com base na .env (sem interações)."""
    url = os.getenv("ES_URL")
    user = os.getenv("ES_USER")
    pwd = os.getenv("ES_PASS")
    if not url or not user or not pwd:
        raise RuntimeError("ES_URL/ES_USER/ES_PASS não configurados na .env")

    tls_kwargs = _resolve_tls_kwargs()

    return Elasticsearch(
        hosts=[url],               # URL completa: https://host:port
        basic_auth=(user, pwd),
        request_timeout=60,
        **tls_kwargs,
    )


def create_embeddings() -> HuggingFaceEmbeddings:
    model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)


def create_vector_store(
    es_client: Elasticsearch,
    embeddings: HuggingFaceEmbeddings,
    index_name: str,
) -> Tuple[ElasticsearchStore, str]:
    """Cria um VectorStore para o índice informado."""
    store = ElasticsearchStore(
        embedding=embeddings,
        index_name=index_name,
        es_connection=es_client,   # compat com versões antigas do pacote
    )
    return store, index_name
