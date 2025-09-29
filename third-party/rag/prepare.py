from typing import List
from sentence_transformers import SentenceTransformer

def split_text(file_path: str) -> List[str]:
    """
    把文本切分成多个块，每个块的长度不超过 chunk_size，且块与块之间有重叠的部分（重叠长度为 chunk_overlap）。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        return [chunk for chunk intext.split('\n')]

# for i, chunk in enumerate(split_text('test.txt')):
#     print(f"[{i}] {chunk}\n")

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
def embed_chunk(chunk: str) -> List[float]:
    """
    把文本块嵌入到向量空间中。
    """
    embeddings = embedding_model.encode(chunk, normalize_embeddings=True)
    return embeddings.tolist()


import chromadb
chromadb_client = chromadb.PersistentClient(path="./chromadb")
collection = chromadb_client.get_or_create_collection(name="defaut")

def save_embeddings(chunks: List[str], embeddings: List[List[float]]):
    """
    把文本块添加到 ChromaDB 集合中。
    """
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"[{i}] {chunk} -> {embedding}")
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"str(i)"]
        )

chunks = split_text('test.txt')
embeddings = [embed_chunk(chunk) for chunk in chunks]
save_embeddings(chunks, embeddings)
