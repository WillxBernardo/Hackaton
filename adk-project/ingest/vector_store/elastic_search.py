import os
import base64
import tempfile
from typing import Optional, Tuple

from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore


def _resolve_tls_kwargs() -> dict:
    """Monta kwargs de TLS conforme .env."""
    verify = os.getenv("ES_VERIFY", "false").lower() == "true"
    ca_path = os.getenv("ES_CA_PATH")
    ca_b64 = os.getenv("ES_CA_BASE64")

    if not verify:
        # modo inseguro (sem verificar cadeia) – útil para self-signed
        return {"verify_certs": False, "ssl_show_warn": True}

    # verify=True
    if ca_path and os.path.exists(ca_path):
        return {"verify_certs": True, "ca_certs": ca_path}

    if ca_b64:
        # materializa CA base64 em arquivo temporário
        pem_tmp = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
        try:
            pem_tmp.write(base64.b64decode(ca_b64))
            pem_tmp.flush()
            return {"verify_certs": True, "ca_certs": pem_tmp.name}
        except Exception:
            # fallback: verifica sem CA explícito (pode falhar)
            return {"verify_certs": True}
    # sem CA fornecido; tenta verificar usando trust store do SO
    return {"verify_certs": True}


def create_es_client() -> Elasticsearch:
    """Cria cliente Elasticsearch com base na .env (sem interações)."""
    url = os.getenv("ES_URL")
    user = os.getenv("ES_USER")
    pwd = os.getenv("ES_PASS")

    if not url or not user or not pwd:
        raise RuntimeError("ES_URL/ES_USER/ES_PASS não configurados na .env")

    tls_kwargs = _resolve_tls_kwargs()

    # Você pode passar a URL completa (https://host:port) em hosts=[url]
    return Elasticsearch(
        hosts=[url],
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
) -> Tuple[ElasticsearchStore, str]:
    index_name = os.getenv("ES_INDEX", "your_index_name")
    store = ElasticsearchStore(
        embedding=embeddings,
        index_name=index_name,
        es_connection=es_client,  # compat com versões antigas do langchain-elasticsearch
    )
    return store, index_name
