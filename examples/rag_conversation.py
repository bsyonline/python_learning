# RAG系统的多轮对话示例
# 展示如何在RAG中处理对话历史和上下文

from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Message:
    """对话消息"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class Conversation:
    """对话类，用于管理对话历史"""
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[Message] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息到对话历史"""
        message = Message(role=role, content=content)
        self.messages.append(message)
    
    def get_history(self, max_turns: int = None) -> List[Message]:
        """获取对话历史"""
        if max_turns is None:
            return self.messages
        return self.messages[-max_turns*2:]  # 每轮包含用户和助手的消息
    
    def format_history(self, max_turns: int = None) -> str:
        """格式化对话历史"""
        messages = self.get_history(max_turns)
        formatted = []
        for msg in messages:
            role = "用户" if msg.role == "user" else "助手"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

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
        query_embedding = self.encoder.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

class ConversationalRAG:
    """支持多轮对话的RAG系统"""
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
        self.conversations: Dict[str, Conversation] = {}
    
    def _get_or_create_conversation(self, conversation_id: str) -> Conversation:
        """获取或创建对话"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(conversation_id)
        return self.conversations[conversation_id]
    
    def _format_context(self, conversation: Conversation, relevant_docs: List[Document]) -> str:
        """格式化上下文信息，包括对话历史和相关文档"""
        parts = []
        
        # 添加对话历史
        history = conversation.format_history(max_turns=3)  # 最近3轮对话
        if history:
            parts.append("对话历史:\n" + history)
        
        # 添加相关文档
        docs_text = "\n\n".join([f"文档内容: {doc.content}" for doc in relevant_docs])
        if docs_text:
            parts.append("相关文档:\n" + docs_text)
        
        return "\n\n".join(parts)
    
    def _generate_prompt(self, query: str, context: str) -> str:
        """生成提示"""
        return f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法回答。
请保持回答的连贯性，考虑对话的上下文。

{context}

当前问题: {query}

回答:"""
    
    def answer(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """回答问题"""
        # 获取或创建对话
        conversation = self._get_or_create_conversation(conversation_id)
        
        # 添加用户问题到对话历史
        conversation.add_message("user", query)
        
        # 检索相关文档
        relevant_docs = self.document_store.similarity_search(query)
        
        # 构建上下文（包括对话历史和相关文档）
        context = self._format_context(conversation, relevant_docs)
        
        # 生成提示
        prompt = self._generate_prompt(query, context)
        
        # 模拟生成回答
        # 在实际应用中，这里会调用语言模型生成回答
        answer = "这是考虑了对话历史的回答..."
        
        # 添加回答到对话历史
        conversation.add_message("assistant", answer)
        
        return {
            "conversation_id": conversation_id,
            "query": query,
            "answer": answer,
            "relevant_documents": relevant_docs,
            "prompt": prompt,
            "conversation_history": conversation.get_history()
        }

def conversation_demo():
    """多轮对话示例"""
    print("多轮对话RAG系统示例：")
    
    # 1. 准备知识库文档
    documents = [
        Document(
            content="Python是由Guido van Rossum创造的编程语言，诞生于1991年。",
            metadata={"source": "python_history", "page": 1}
        ),
        Document(
            content="Python的设计哲学强调代码的可读性和简洁性，其中最重要的特征是使用缩进来表示代码块。",
            metadata={"source": "python_features", "page": 1}
        ),
        Document(
            content="Python支持多种编程范式，包括面向对象编程、命令式编程和函数式编程。",
            metadata={"source": "python_paradigms", "page": 1}
        ),
    ]
    
    # 2. 初始化系统
    print("\n1. 初始化系统...")
    doc_store = DocumentStore()
    doc_store.add_documents(documents)
    rag = ConversationalRAG(doc_store)
    
    # 3. 模拟多轮对话
    print("\n2. 开始多轮对话:")
    conversation_id = "conv_001"
    
    # 第一轮：询问Python的创造者
    query1 = "谁创造了Python？"
    print(f"\n用户: {query1}")
    result1 = rag.answer(query1, conversation_id)
    print(f"助手: {result1['answer']}")
    print("\n相关文档:")
    for doc in result1['relevant_documents']:
        print(f"- {doc.content}")
    
    # 第二轮：询问Python的特点
    query2 = "这门语言有什么特点？"
    print(f"\n用户: {query2}")
    result2 = rag.answer(query2, conversation_id)
    print(f"助手: {result2['answer']}")
    print("\n相关文档:")
    for doc in result2['relevant_documents']:
        print(f"- {doc.content}")
    
    # 第三轮：询问具体的编程范式
    query3 = "你提到的这些编程范式是什么意思？"
    print(f"\n用户: {query3}")
    result3 = rag.answer(query3, conversation_id)
    print(f"助手: {result3['answer']}")
    print("\n相关文档:")
    for doc in result3['relevant_documents']:
        print(f"- {doc.content}")
    
    # 显示完整的对话历史
    print("\n3. 完整对话历史:")
    conversation = rag.conversations[conversation_id]
    print(conversation.format_history())

if __name__ == "__main__":
    conversation_demo() 