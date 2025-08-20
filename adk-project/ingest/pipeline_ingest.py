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

# ===== utils =====
def batched(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]

def list_overlay_dirs(base: Path) -> List[Path]:
    """Lista subdiretórios imediatos em base (cada um vira um índice)."""
    return sorted([p for p in base.iterdir() if p.is_dir()])


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

def load_documents(dir_path: Path) -> List[Tuple[str, Dict]]:
    """Carrega documentos de UM overlay (dir_path)."""
    pairs: List[Tuple[str, Dict]] = []
    if not dir_path.exists():
        return pairs
    for fp in dir_path.rglob("*"):
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
    load_dotenv()  # lê .env

    # Base com overlays (cada subpasta = índice)
    base_dir = Path(os.getenv("DATA_BASE_DIR", "adk-project/ingest/data")).resolve()
    batch_size = int(os.getenv("BATCH_SIZE", "128"))

    if not base_dir.exists():
        raise FileNotFoundError(f"Base de dados não existe: {base_dir}")

    overlay_dirs = list_overlay_dirs(base_dir)
    if not overlay_dirs:
        print(f"Nenhum overlay encontrado em {base_dir}. Crie subpastas (ex.: hipoteses, diretrizes).")
        return

    print("Criando embeddings (HuggingFace)...")
    embeddings = create_embeddings()

    print("Conectando no Elasticsearch...")
    es_client = create_es_client()
    try:
        info = es_client.info()
        print(f"Cluster: {info.get('cluster_name')} | Version: {info.get('version', {}).get('number')}")
    except Exception as e:
        print(f"[WARN] es_client.info(): {e}")

    splitter = build_text_splitter()

    # === loop por overlay/índice ===
    for overlay_dir in overlay_dirs:
        index_name = overlay_dir.name  # nome do índice = nome da pasta
        print(f"\n==== Overlay: {overlay_dir}  ->  Index: '{index_name}' ====")

        pairs = load_documents(overlay_dir)
        print(f"Arquivos carregados (após parsing): {len(pairs)}")

        if not pairs:
            print("Nada para ingerir; pulando...")
            continue

        # chunking
        texts: List[str] = []
        metas: List[Dict] = []
        for text, meta in pairs:
            for chunk in splitter.split_text(text):
                texts.append(chunk)
                metas.append(meta)

        total = len(texts)
        print(f"Total de chunks para '{index_name}': {total}")

        # vectorstore para ESTE índice
        store, _ = create_vector_store(es_client, embeddings, index_name)

        print(f"Ingerindo '{index_name}' em lotes de {batch_size}...")
        inserted = 0
        for t_batch, m_batch in zip(batched(texts, batch_size), batched(metas, batch_size)):
            store.add_texts(texts=t_batch, metadatas=m_batch)
            inserted += len(t_batch)
            if inserted % (batch_size * 5) == 0 or inserted == total:
                print(f"[{index_name}] Inseridos: {inserted}/{total}")

        es_client.indices.refresh(index=index_name)
        print(f"[OK] Index '{index_name}' populado com {inserted} chunks.")

    print("\nTudo pronto! 🎉")


if __name__ == "__main__":
    main()
