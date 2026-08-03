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
        from data.ingest import ingest_all
        # Sync the database exactly once on startup
        ingest_all()
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

            # Prepare the initial state
            initial_state = {
                "query": user_input,
                "next_agent": "",
                "final_response": "",
                "agent_outputs": {}
            }

            # 3. Run the graph
            with console.status("[dim]Nexus AI is thinking...[/dim]", spinner="dots"):
                final_state = app.invoke(initial_state)

            # 4. Extract the answer
            agent_outputs = final_state.get("agent_outputs", {})
            next_agent    = final_state.get("next_agent", "END")

            # Did a specialist answer?
            if agent_outputs and next_agent != "END" and next_agent in agent_outputs:
                answer = agent_outputs[next_agent]
                source = f"Specialist: {next_agent.capitalize()}"
            
            # Or did the Main AI answer directly?
            else:
                answer = final_state.get("final_response") or "I'm not sure how to respond to that."
                source = "Main AI"

            # 5. Print the result nicely
            console.print(Panel(Markdown(answer), title=f"[bold magenta]{source}[/bold magenta]", border_style="magenta"))
            console.print()  # empty line for spacing

        except KeyboardInterrupt:
            console.print("\n[dim]Shutting down Nexus AI... Goodbye.[/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
