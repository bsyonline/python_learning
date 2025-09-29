import chromadb
from prepare import embed_chunk
from sentence_transformers import CrossEncoder

def retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    从 ChromaDB 集合中检索与查询最相关的文本块。
    """
    embedding = embed_chunk(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results['documents'][0]

query = "你好"
retrieve_chunks = retrieve(query)
for i, chunk in enumerate(retrieve_chunks):
    print(f"[{i}] {chunk}\n")

def rerank(query: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """
    对检索到的文本块进行重新排序，根据与查询的相关性。
    """
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    scores = cross_encoder.predict([(query, chunk) for chunk in chunks])
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)][:top_k]
    return sorted_chunks

rerank_chunks = rerank(query, retrieve_chunks)
for i, chunk in enumerate(rerank_chunks):
    print(f"[{i}] {chunk}\n")




def generate(query: str, chunks: List[str]) -> str:
    """
    生成基于查询的文本。
    """
    prompt = f"""根据以下内容回答问题.
    用户问题：{query}
    
    相关片段：
    {'\n'.join(chunks)}

    请根据以上内容回答，不要编造信息。
    """
    response = llm.generate(
        model="qwen-plus",
        prompt=prompt
    )
    return response.content

response = generate(query, rerank_chunks)
print(response)