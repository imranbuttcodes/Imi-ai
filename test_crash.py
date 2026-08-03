from app import *
from graphs.nexus_graph import build_master_graph

def test():
    app_graph = build_master_graph()
    state = {
        "query": "What is the company's Policy?",
        "next_agent": "",
        "final_response": "",
        "agent_outputs": {}
    }
    result = app_graph.invoke(state)
    print(result)

if __name__ == "__main__":
    test()
