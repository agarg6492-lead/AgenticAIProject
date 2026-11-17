#%%
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
import os
import base64
import json
import re
#from huggingface_hub import login
#from transformers import AutoTokenizer, AutoModel
#from langchain_google_genai import ChatGoogleGenerativeAI
#from huggingface_hub import InferenceClient
#from diffusers import StableDiffusionPipeline

load_dotenv()
#%%
with open("Aladdin_Adventures.png", "rb") as f:
    image_b64= base64.b64encode(f.read()).decode()

#image_b64[:200]

@tool
def storyImagePrompt(prompt: str):
    """extract only the image generation prompt and save it to a text file."""
    with open("story_image.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    return prompt


#%%
#llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm = ChatGroq(model="qwen/qwen3.8-27b")

messages= HumanMessage(content= [{"type":"image_url", "image_url":{"url": f"data:image/png;base64,{image_b64}"}},
                                {"type":"text", "text": """Analyse this story image. 
                                Create one 'colorful image generation prompt' which shows main characters and moral of the story".
                                Return only valid JSON"""
                                }
                                ])

agent= create_agent(
    llm,
    tools=[storyImagePrompt],
    system_prompt= """when a user ask for story image prompt, use storyImagePrompt. The generated output must not contain any words that cause third party copyright conflicts.
                    You have to strictly fetch all info from the user provided image only. Do not provide response if storyImagePrompt is not called.
                    """
                    )

def run(message: list):
    #config= {"configurable":{"thread_id":"user_aashu_session_id"}}
    response= agent.invoke({"messages": [{"role":"user", "content": message}]})
    #print(response["messages"][-1].response_metadata)

run(messages.content)


#%%
# result= response['messages']
# print(response['messages'][-1].response_metadata)
# print(re.sub(r'\s+', ' ', messages.content[-1]["text"]).strip())
