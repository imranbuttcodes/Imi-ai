import os
import time
from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
except ImportError:
    print("Please install required packages:")
    print("pip install langchain-groq python-dotenv")
    exit(1)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODELS = [
    ("General Assistant", "llama-3.3-70b-versatile"),
    ("Fast Router", "llama-3.1-8b-instant"),
    ("GPT OSS 120B", "openai/gpt-oss-120b"),
    ("GPT OSS 20B", "openai/gpt-oss-20b"),
    ("Qwen 3 32B", "qwen/qwen3-32b"),
    ("DeepSeek R1 Llama 70B", "deepseek-r1-distill-llama-70b"),
    ("DeepSeek R1 Qwen 32B", "deepseek-r1-distill-qwen-32b"),
    ("Gemma 2 9B", "gemma2-9b-it"),
]


def test_model(role_name, model_name):
    print("\n" + "=" * 70)
    print(f"{role_name}")
    print(f"Model : {model_name}")
    print("=" * 70)

    start = time.time()

    try:
        llm = ChatGroq(
            model=model_name,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=20,
        )

        response = llm.invoke([
            HumanMessage(content="Reply with exactly one word: Success")
        ])

        elapsed = time.time() - start

        print("Status : PASS")
        print(f"Speed  : {elapsed:.2f} sec")
        print(f"Reply  : {response.content.strip()}")

        return {
            "model": model_name,
            "status": "PASS",
            "speed": elapsed,
        }

    except Exception as e:

        print("Status : FAIL")
        print(f"Error  : {e}")

        return {
            "model": model_name,
            "status": "FAIL",
            "speed": None,
        }


if __name__ == "__main__":

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env")
        exit(1)

    results = []

    print("\nStarting Groq Model Benchmarks...\n")

    for role, model in MODELS:
        results.append(test_model(role, model))

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] == "FAIL"]

    passed.sort(key=lambda x: x["speed"])

    print(f"\nPassed : {len(passed)}")
    print(f"Failed : {len(failed)}")

    print("\nFastest Models")

    for r in passed:
        print(f"{r['speed']:.2f}s   {r['model']}")

    if failed:
        print("\nUnavailable Models")

        for r in failed:
            print(f"- {r['model']}")