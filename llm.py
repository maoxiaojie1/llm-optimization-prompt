import sys
import os
import json
import folder_paths

custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
llm_opt_prompt_path = os.path.join(custom_nodes_path, "llm-optimization-prompt")

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

qwen2_model_list = ["qwen-max", "qwen-plus"]

qwen2_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

openai_model_list = ["gpt-3.5-turbo", "gpt4-o"]
ollama_model_list = ["qwen2:7b"] 

model_list = openai_model_list + qwen2_model_list

json_schema = {
    "title": "Prompt",
    "description": "prompt optimization",
    "type": "object",
    "properties": {
        "input": {"title": "Input", "description": "input prompt", "type": "string"},
        "output": {"title": "Output", "description": "optimized prompt", "type": "string"},
    },
    "required": ["input", "output"],
}

example = {
        "input": "一个漂亮的女孩，面向着镜头",
        "output": "A beautiful girl, with her captivating charm becoming even more enchanting under the moonlight's soft glow. She held the attention of every onlooker, her graceful silhouette against the sky making her resemble a living masterpiece. The atmosphere around her was filled with silent admiration as she observed the scene, adding an aura of mystique to the entire environment."
    }

messages = [
    ("system", "你是一位资深的ai绘画的prompt工程师，需要按照要求，提供绘画prompt优化服务，要求：1. 将human消息翻译成英文; 2. 对human消息添加一些形容词，补充细节描述，使得最终语句在80词左右; 3. 按照下面的JSON格式输出\n {schema} \n这里给出个例子, human: 一个漂亮的女孩，面向着镜头, 你输出：{example}"),
    ("human","{prompt}")
]

def chat(llm, prompt):
    template = ChatPromptTemplate.from_messages(messages)
    chain = template | llm | JsonOutputParser()
    print(f"orgin prompt: {prompt}")
    prompt = chain.invoke({
        "prompt": prompt, 
        "schema": json.dumps(json_schema, indent=2, ensure_ascii=False),
        "example": json.dumps(example, indent=2, ensure_ascii=False),
    })
    print(f"after llm opt, prompt: {prompt}")
    return (prompt['output'],)

class OpenAi:
    def __init__(self):
      pass
    
    @classmethod
    def INPUT_TYPES(s):
      return {
        "required": {
            "prompt": ("STRING", {"default": '', "multiline": True}),
            "model": (model_list,),
            "api_key": ("STRING", {"default": ''}),
        },
        "optional": {
          "base_url": ("STRING", {"default": ''}),
        }
      }  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_prompt",)
    
    FUNCTION = "chat"
    
    #OUTPUT_NODE = False
    
    CATEGORY = "LLM优化提示词"
    
    def chat(self, prompt, model, api_key, base_url):
      if base_url == "":
        if model in qwen2_model_list:
          base_url = qwen2_base_url
        else:
          base_url = None
          
      llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
      )
      return chat(llm, prompt)

class Ollama:
    def __init__(self):
      pass
    
    @classmethod
    def INPUT_TYPES(s):
      return {
        "required": {
            "prompt": ("STRING", {"default": '', "multiline": True}),
            "model": (ollama_model_list,),
        },
        "optional": {
          "base_url": ("STRING", {"default": ''}),
        }
      }  
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_prompt",)
    
    FUNCTION = "chat"
    
    #OUTPUT_NODE = False
    
    CATEGORY = "LLM优化提示词"
    
    def chat(self, prompt, model, base_url):
      if base_url == "":
        base_url = "http://localhost:11434"
          
      llm = ChatOllama(
        model=model,
        format="json",
        base_url=base_url,
      )

      return chat(llm, prompt)

NODE_CLASS_MAPPINGS = {
  "OpenAi": OpenAi,
  "Ollama": Ollama,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "OpenAi": "OpenAi",
  "Ollama": "Ollama",
}
