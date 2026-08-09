# import os
# import sys
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# # Clean up environment to prevent OpenRouter/OpenAI hijack
# if "OPENAI_API_BASE" in os.environ: del os.environ["OPENAI_API_BASE"]
# if "OPENAI_BASE_URL" in os.environ: del os.environ["OPENAI_BASE_URL"]
# if "OPENAI_API_KEY" in os.environ: del os.environ["OPENAI_API_KEY"]

# NVIDIA_KEY = os.getenv("NVIDIA_API_KEY").strip() if os.getenv("NVIDIA_API_KEY") else None

# MODELS_TO_TEST = [
#     "gpt-oss-120b", # Adding exactly what you requested to check if Nvidia supports it
#     "meta/llama-3.1-70b-instruct",
#     "meta/llama-3.1-405b-instruct",
#     "nvidia/llama-3.1-nemotron-70b-instruct",
#     "nvidia/nemotron-4-340b-instruct"
# ]

# if not NVIDIA_KEY:
#     print("ERROR: NVIDIA_API_KEY not found in .env file!")
#     sys.exit(1)

# print("Testing NVIDIA NIM Models via OpenAI Compatible Endpoint...")
# print("-" * 50)

# working_models = []

# for model_name in MODELS_TO_TEST:
#     print(f"\nTesting: {model_name}")
#     try:
#         llm = ChatOpenAI(
#             api_key=NVIDIA_KEY,
#             base_url="https://integrate.api.nvidia.com/v1",
#             model=model_name,
#             temperature=0,
#             max_tokens=100
#         )
#         response = llm.invoke("Hi")
#         print(f"SUCCESS! -> {model_name} is a valid Chat model.")
#         working_models.append(model_name)
        
#     except Exception as e:
#         print(f"FAILED -> {e}")

# print("\n" + "=" * 50)
# print("AVAILABLE WORKING NVIDIA MODELS:")
# for m in working_models:
#     print(f"- {m}")
# print("=" * 50)

# import os
# from dotenv import load_dotenv

# from openai import OpenAI

# load_dotenv()

# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = os.getenv('GPT-OSS-120B-API_KEY')
# )

# completion = client.chat.completions.create(
#   model="openai/gpt-oss-120b",
#   messages=[{"role":"user","content":"Hi Bro?"}],
#   temperature=1,
#   top_p=1,
#   max_tokens=4096,
#   stream=False
# )

# reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
# if reasoning:
#   print(reasoning)
# print(completion.choices[0].message.content)


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Initialize the LangChain model
# We map your base_url, api_key, and generation parameters directly here
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv('GPT-OSS-120B-API_KEY'),
    model="openai/gpt-oss-120b",
)

# 2. Create the message prompt
messages = [
    HumanMessage(content="Hi Bro?")
]

# 3. Invoke the model
response = llm.invoke(messages)

# 4. Extract reasoning and content
# LangChain stores non-standard API fields (like reasoning_content) inside additional_kwargs
reasoning = response.additional_kwargs.get("reasoning_content")

if reasoning:
    print(reasoning)

print(response.content)