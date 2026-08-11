import os
import shutil
from langchain_core.tools import tool
from langgraph.types import interrupt

@tool
def delete_path(path: str) -> str:
    """
    Deletes a file or directory at the specified absolute path.
    Requires human confirmation before executing the deletion.
    """
    # Yield control back to the state machine for explicit human approval
    confirmation = interrupt(
        f"\n[WARNING] The Imi AI is requesting to delete the following path:\n"
        f"Path: {path}\n"
        f"Do you approve this deletion? (y/n): "
    )
    
    if str(confirmation).strip().lower() != 'y':
        return f"Deletion cancelled: The user denied permission to delete {path}."
        
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            return f"Successfully deleted file: {path}"
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return f"Successfully deleted directory: {path}"
        else:
            return f"Error: Path does not exist: {path}"
    except Exception as e:
        return f"Error occurred while trying to delete {path}: {str(e)}"
