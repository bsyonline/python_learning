from langchain_core.prompts import (
    PromptTemplate, 
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

def demo_prompt_template():
    """演示基础PromptTemplate"""
    print("\n演示 1: PromptTemplate (基础提示模板)")
    print("=" * 60)
    
    # 1. 基础PromptTemplate
    simple_template = "请用中文回答: {question}"
    prompt_template = PromptTemplate.from_template(simple_template)
    
    print("模板内容:", simple_template)
    print("模板变量:", prompt_template.input_variables)
    
    # 格式化提示
    formatted_prompt = prompt_template.format(question="什么是人工智能?")
    print("格式化后的提示:", formatted_prompt)
    
    # 2. 多变量模板
    multi_var_template = """
    请根据以下信息回答问题:
    主题: {topic}
    问题: {question}
    语言: {language}
    """
    
    multi_prompt = PromptTemplate(
        template=multi_var_template,
        input_variables=["topic", "question", "language"]
    )
    
    formatted_multi = multi_prompt.format(
        topic="机器学习",
        question="请解释监督学习和无监督学习的区别",
        language="中文"
    )
    
    print("\n多变量模板:")
    print(formatted_multi)

def demo_chat_prompt_template():
    """演示ChatPromptTemplate"""
    print("\n\n演示 2: ChatPromptTemplate (聊天提示模板)")
    print("=" * 60)
    
    # 1. 基础ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的AI助手，请用{language}回答。"),
        ("human", "请解释一下{concept}的概念")
    ])
    
    print("ChatPromptTemplate结构:")
    print("消息数量:", len(chat_template.messages))
    
    # 格式化聊天提示
    formatted_chat = chat_template.format_messages(
        language="中文",
        concept="深度学习"
    )
    
    print("\n格式化后的聊天消息:")
    for i, msg in enumerate(formatted_chat):
        print(f"{i+1}. {type(msg).__name__}: {msg.content}")
    
    # 2. 复杂聊天模板
    complex_chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，请用{language}回答。"),
        ("human", "用户问题: {question}"),
        ("ai", "让我来帮你分析这个问题..."),
        ("human", "{follow_up}")
    ])
    
    complex_messages = complex_chat_template.format_messages(
        role="机器学习专家",
        language="中文",
        question="什么是神经网络?",
        follow_up="它能解决哪些实际问题?"
    )
    
    print("\n复杂聊天模板:")
    for i, msg in enumerate(complex_messages):
        print(f"{i+1}. {type(msg).__name__}: {msg.content}")

def demo_message_templates():
    """演示各种消息模板"""
    print("\n\n演示 3: 各种消息模板")
    print("=" * 60)
    
    # 1. SystemMessagePromptTemplate
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{expert_type}专家，请用{language}回答。"
    )
    
    system_message = system_template.format(
        expert_type="人工智能",
        language="中文"
    )
    
    print("SystemMessagePromptTemplate:")
    print("类型:", type(system_message))
    print("内容:", system_message.content)
    
    # 2. HumanMessagePromptTemplate  
    human_template = HumanMessagePromptTemplate.from_template(
        "请解释{concept}的工作原理"
    )
    
    human_message = human_template.format(concept="Transformer模型")
    
    print("\nHumanMessagePromptTemplate:")
    print("类型:", type(human_message))
    print("内容:", human_message.content)
    
    # 3. AIMessagePromptTemplate
    ai_template = AIMessagePromptTemplate.from_template(
        "根据我的分析，{concept}的主要特点是: {features}"
    )
    
    ai_message = ai_template.format(
        concept="卷积神经网络",
        features="局部连接、权重共享、池化操作"
    )
    
    print("\nAIMessagePromptTemplate:")
    print("类型:", type(ai_message))
    print("内容:", ai_message.content)

def demo_advanced_prompt_techniques():
    """演示高级提示技术"""
    print("\n\n演示 4: 高级提示技术")
    print("=" * 60)
    
    # 1. 部分格式化
    partial_template = PromptTemplate.from_template(
        "用{language}解释{concept}"
    )
    
    # 部分格式化
    partial_prompt = partial_template.partial(language="中文")
    full_prompt = partial_prompt.format(concept="循环神经网络")
    
    print("部分格式化:")
    print("模板:", partial_template.template)
    print("部分格式化后:", full_prompt)
    
    # 2. 模板组合
    combined_template = ChatPromptTemplate.from_messages([
        ("system", "请用中文回答"),
        ("human", "主题: {topic}\n问题: {question}")
    ])
    
    combined_messages = combined_template.format_messages(
        topic="自然语言处理",
        question="什么是BERT模型?"
    )
    
    print("\n模板组合:")
    for msg in combined_messages:
        print(f"{type(msg).__name__}: {msg.content}")

def main():
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_message_templates()
    demo_advanced_prompt_techniques()
    
    print("\n" + "=" * 70)
    print("• PromptTemplate用于基础文本提示")
    print("• ChatPromptTemplate用于多轮对话场景")
    print("• 各种MessageTemplate用于特定类型的消息")
    print("• 支持部分格式化和模板组合")
    print("• 可以与LLM链式组合使用")

if __name__ == "__main__":
    main()