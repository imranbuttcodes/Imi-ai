# ==============================================================
# app.py — The Terminal Interface
# ==============================================================
# Run this file to chat with Nexus AI.
# It boots up the master graph and provides a CLI loop.
# ==============================================================

import sys
import os

# ULTIMATE FIX FOR defusedxml ON PYTHON 3.10: 
# Python's security patch blocks defusedxml imports if it thinks it's in the CWD.
# We temporarily change the CWD, import all necessary defusedxml modules, and then switch back.
_old_cwd = os.getcwd()
try:
    os.chdir(os.path.dirname(sys.executable))
    import defusedxml
    import defusedxml.ElementTree
    import defusedxml.minidom
    import defusedxml.cElementTree
finally:
    os.chdir(_old_cwd)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from core.config import settings
from graphs.nexus_graph import build_master_graph

console = Console()

def main():
    # 1. Boot up the system & Auto-Ingest
    console.print(f"[bold blue]Booting {settings.app_name}...[/bold blue]")
    try:
        #from data.ingest import ingest_all
        # Sync the database exactly once on startup
       # ingest_all()
        app = build_master_graph()
    except Exception as e:
        console.print(f"[bold red]Failed to boot system:[/bold red] {e}")
        sys.exit(1)

    console.print("[bold green]System Online.[/bold green] (Type 'exit' to quit)\n")

    # 2. Main Chat Loop
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[dim]Shutting down Nexus AI... Goodbye.[/dim]")
                break

            # Prepare the state update
            # We use `messages` and let LangGraph's reducer handle appending it
            update_state = {
                "query": user_input,
                "messages": [("user", user_input)],
                "next_agent": "",
                "final_response": "",
                "agent_outputs": {}
            }
            
            # Use a consistent thread_id for persistence across boots
            config = {"configurable": {"thread_id": "nexus_default_user"}}

            # 3. Run the graph with streaming for live debugging traces
            with console.status("[dim]Nexus AI is thinking...[/dim]", spinner="dots"):
                for event in app.stream(update_state, config=config):
                    for node_name, state_update in event.items():
                        console.print(f"[dim]  [Graph Trace] Finished node: [bold]{node_name}[/bold][/dim]")

            # Fetch the complete final state from the checkpointer
            final_state = app.get_state(config).values

            # 4. Extract the answer
            messages = final_state.get("messages", [])
            answer = messages[-1].content if messages else "I'm not sure how to respond to that."
            source = "Main AI"

            # 5. Print the result nicely
            console.print(Panel(Markdown(answer), title=f"[bold magenta]{source}[/bold magenta]", border_style="magenta"))
            console.print()  # empty line for spacing

            # 6. Fire and Forget: Background Semantic Reflection
            # We run this in a separate thread so the user can instantly type their next message
            # while the LLM extracts facts in the background (~2 seconds).
            import threading
            from infrastructure.memory.reflection import extract_and_store_facts
            messages_snapshot = final_state.get("messages", [])
            threading.Thread(target=extract_and_store_facts, args=(messages_snapshot,), daemon=True).start()

        except KeyboardInterrupt:
            console.print("\n[dim]Shutting down Nexus AI... Goodbye.[/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
