
import asyncio
from openai import OpenAI
from pydantic import BaseModel,Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.providers.ollama import OllamaProvider
from typing import Any
import tools

ollama_model = OpenAIModel(
    model_name='qwen2.5:0.5b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(ollama_model,
              system_prompt="You are an experienced programmer",
              tools=[tools.read_file, tools.list_files, tools.rename_file])

def main():
    history = []
    while True:
        user_input = input("Input: ")
        resp = agent.run_sync(user_input,
                              message_history=history)
        history = list(resp.all_messages())
        print(resp.output)
    # files = tools.list_files()
    # print(files)


if __name__ == "__main__":
    main()