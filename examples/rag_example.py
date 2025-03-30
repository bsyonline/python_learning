# RAG (Retrieval Augmented Generation) 示例
# 展示如何结合文档检索和语言模型来回答问题

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Document:
    """文档类，用于存储文本片段"""
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None

class DocumentStore:
    """文档存储类，用于管理和检索文档"""
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到存储"""
        self.documents.extend(documents)
        # 计算文档嵌入
        contents = [doc.content for doc in documents]
        embeddings = self.encoder.encode(contents)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """基于相似度搜索文档"""
        # 计算查询的嵌入
        query_embedding = self.encoder.encode([query])[0]
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 获取最相似的k个文档
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

class RAGSystem:
    """RAG系统类"""
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
    
    def _format_context(self, relevant_docs: List[Document]) -> str:
        """格式化上下文信息"""
        return "\n\n".join([f"文档内容: {doc.content}" for doc in relevant_docs])
    
    def _generate_prompt(self, query: str, context: str) -> str:
        """生成提示"""
        return f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文信息:
{context}

问题: {query}

回答:"""
    
    def answer(self, query: str) -> Dict[str, Any]:
        """回答问题"""
        # 1. 检索相关文档
        relevant_docs = self.document_store.similarity_search(query)
        
        # 2. 构建上下文
        context = self._format_context(relevant_docs)
        
        # 3. 生成提示
        prompt = self._generate_prompt(query, context)
        
        # 4. 在实际应用中，这里会调用语言模型生成回答
        # 这里我们模拟一个简单的回答
        answer = "这是基于检索到的文档生成的回答..."
        
        return {
            "query": query,
            "answer": answer,
            "relevant_documents": relevant_docs,
            "prompt": prompt
        }

def rag_demo():
    """RAG系统示例"""
    print("RAG系统示例：")
    
    # 1. 准备示例文档
    documents = [
        Document(
            content="Python是一种高级编程语言，以其简洁的语法和丰富的库而闻名。",
            metadata={"source": "programming_guide", "page": 1}
        ),
        Document(
            content="机器学习是人工智能的一个子领域，主要研究如何让计算机从数据中学习。",
            metadata={"source": "ml_textbook", "page": 15}
        ),
        Document(
            content="深度学习是机器学习的一个分支，使用多层神经网络进行学习。",
            metadata={"source": "dl_guide", "page": 3}
        ),
    ]
    
    # 2. 初始化文档存储和RAG系统
    print("\n1. 初始化系统...")
    doc_store = DocumentStore()
    doc_store.add_documents(documents)
    rag = RAGSystem(doc_store)
    
    # 3. 测试问答
    print("\n2. 测试问答:")
    questions = [
        "什么是Python？",
        "机器学习和深度学习有什么关系？",
        "谁发明了Python？"  # 文档中没有这个信息
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = rag.answer(question)
        print(f"找到的相关文档:")
        for doc in result["relevant_documents"]:
            print(f"- {doc.content}")
        print(f"生成的回答: {result['answer']}")

if __name__ == "__main__":
    rag_demo() 