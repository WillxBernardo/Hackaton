import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

from dotenv import load_dotenv
from pypdf import PdfReader

from splitter.text_splitter import build_text_splitter
from vector_store.elastic_search import (
    create_es_client,
    create_embeddings,
    create_vector_store,
)

# ===== util =====
def batched(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

# ===== loaders =====
def read_pdf(file_path: Path) -> List[Tuple[str, Dict]]:
    out = []
    reader = PdfReader(str(file_path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, {"source": str(file_path), "page": i + 1}))
    return out

def read_txt(file_path: Path) -> List[Tuple[str, Dict]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [(text, {"source": str(file_path)})] if text.strip() else []

def read_json(file_path: Path) -> List[Tuple[str, Dict]]:
    try:
        data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    chunks = []
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            try:
                text = json.dumps(obj, ensure_ascii=False)
                if text.strip():
                    chunks.append((text, {"source": str(file_path), "item": idx}))
            except Exception:
                continue
    else:
        text = json.dumps(data, ensure_ascii=False)
        if text.strip():
            chunks.append((text, {"source": str(file_path)}))
    return chunks

def load_documents(base_dir: str) -> List[Tuple[str, Dict]]:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {base_dir}")
    pairs: List[Tuple[str, Dict]] = []
    for fp in base.rglob("*"):
        if not fp.is_file():
            continue
        sfx = fp.suffix.lower()
        try:
            if sfx == ".pdf":
                pairs.extend(read_pdf(fp))
            elif sfx in {".txt", ".md"}:
                pairs.extend(read_txt(fp))
            elif sfx == ".json":
                pairs.extend(read_json(fp))
        except Exception as e:
            print(f"[WARN] Falha ao ler {fp}: {e}")
    return pairs

# ===== pipeline =====
def main():
    load_dotenv()  # carrega .env

    docs_dir = os.getenv("DOCS_DIR", "adk-project/ingest/data/docs")
    batch_size = int(os.getenv("BATCH_SIZE", "128"))

    print("Carregando documentos...")
    pairs = load_documents(docs_dir)
    print(f"Arquivos carregados (após parsing): {len(pairs)}")

    splitter = build_text_splitter()

    print("Gerando chunks...")
    texts: List[str] = []
    metas: List[Dict] = []
    for text, meta in pairs:
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metas.append(meta)

    total = len(texts)
    print(f"Total de chunks: {total}")

    print("Criando embeddings (HuggingFace)...")
    embeddings = create_embeddings()

    print("Conectando no Elasticsearch...")
    es_client = create_es_client()
    try:
        info = es_client.info()
        print(f"Cluster: {info.get('cluster_name')} | Version: {info.get('version', {}).get('number')}")
    except Exception as e:
        print(f"[WARN] es_client.info(): {e}")

    print("Criando VectorStore...")
    store, index_name = create_vector_store(es_client, embeddings)

    print(f"Ingerindo em lotes de {batch_size}...")
    inserted = 0
    for t_batch, m_batch in zip(batched(texts, batch_size), batched(metas, batch_size)):
        store.add_texts(texts=t_batch, metadatas=m_batch)
        inserted += len(t_batch)
        if inserted % (batch_size * 5) == 0 or inserted == total:
            print(f"Inseridos: {inserted}/{total}")

    es_client.indices.refresh(index=index_name)
    print(f"OK! Index '{index_name}' populado com {inserted} chunks.")


if __name__ == "__main__":
    main()
