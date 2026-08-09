import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Add parent directory to path to load .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

MODELS_TO_TEST = [
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

print("Testing HuggingFace Chat Models...")
print("-" * 50)

working_models = []

for repo_id in MODELS_TO_TEST:
    print(f"\nTesting: {repo_id}")
    try:
        model = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
        )
        llm = ChatHuggingFace(llm=model)
        
        # Test a simple chat invocation
        response = llm.invoke("Hi")
        print(f"SUCCESS! -> {repo_id} is a valid Chat model.")
        working_models.append(repo_id)
        
    except Exception as e:
        print(f"FAILED -> {e}")

print("\n" + "=" * 50)
print("AVAILABLE WORKING MODELS:")
for m in working_models:
    print(f"- {m}")
print("=" * 50)
