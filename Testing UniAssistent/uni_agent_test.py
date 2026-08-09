import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from uni_db_manager import UniDatabaseManager

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

print("Initializing University Database Manager (Full Capabilities)...")
db_manager = UniDatabaseManager(headless=True)
print("Database Manager Ready!")

# ==========================================
# UNIFIED GATEKEEPER TOOLS
# These return the EXACT, FULL JSON payloads from the scraper so the LLM has ZERO blind spots.
# ==========================================

@tool
def sync_university_data(endpoint: str, course_name: str = None) -> str:
    """Forces a fresh live scrape of the portal to bypass the cache. Use this ONLY if the user explicitly asks to refresh, update, or sync their data.
    Endpoint must be one of: "dashboard", "timetable", "grades", "invoices", "notifications", "datesheet", or "course".
    If endpoint is "course", you MUST provide the course_name."""
    try:
        db_manager.force_refresh(endpoint, course_name)
        return f"Successfully refreshed {endpoint} data from the portal!"
    except Exception as e:
        return f"Error refreshing data: {str(e)}"

@tool
def get_student_dashboard() -> str:
    """Fetches the student's main Dashboard overview. 
    Use this tool to find: the student's Name, Roll Number, Department, current CGPA, Earned/Total Credits, Scholarships, a quick list of Today's Classes, and a list of all currently Enrolled Courses."""
    try:
        return db_manager.get_dashboard().model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_detailed_profile() -> str:
    """Fetches the student's Detailed Personal Profile.
    Use this tool to find private information like: CNIC, Phone Number, Email, Date of Birth, Gender, Blood Group, Religion, Nationality, Present/Permanent Addresses, and Father/Guardian details."""
    try:
        return db_manager.get_profile().model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_full_timetable() -> str:
    """Fetches the student's complete Weekly Class Schedule.
    Use this tool to find out what classes the student has on ANY given day (Monday through Sunday), including start/end times, the Teacher's name, and the Room/Venue."""
    try:
        return db_manager.get_timetable().model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_academic_history() -> str:
    """Fetches the student's complete Past Academic Transcript / Grades.
    Use this tool to find historical data for past semesters, including: past CGPA/SGPA, past attempted/earned credit hours, and the final grades (e.g., A, B+, C) obtained in every past course."""
    try:
        return db_manager.get_academic_history().model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_invoices() -> str:
    """Fetches the student's Financial Invoices and Fee Challans.
    Use this tool to check for any pending or paid university fees, invoice dates, payable amounts, and due dates."""
    try:
        return db_manager.get_invoices().model_dump_json(indent=2, by_alias=True)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_notifications() -> str:
    """Fetches the student's unread Portal Notifications and Alerts.
    Use this tool if the user asks about new alerts, messages, or notifications from the university."""
    try:
        return db_manager.get_notifications().model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_exam_datesheet() -> str:
    """Fetches the student's Exam Datesheet.
    Use this tool to find the schedule for upcoming Midterm or Final exams, including exact dates, times, and exam venues/rooms."""
    try:
        return db_manager.get_datesheet().model_dump_json(indent=2, by_alias=True)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_full_course_details(course_name: str) -> str:
    """Fetches EVERYTHING related to a specific course. 
    Use this tool if the user asks about a specific class (e.g., 'Functional English'). It returns:
    1. metadata (Section, Semester, Teacher Name via raw_text)
    2. attendance (Percentages and day-by-day logs)
    3. gradebook (Marks/percentages for past quizzes and assignments)
    4. materials (Uploaded files and their filenames)
    5. submissions & assessments (Upcoming assignment due dates)
    6. outline (Syllabus, reference books, and grading weights)."""
    try:
        details = db_manager.get_course_details(course_name)
        if not details: return "Course not found."
        return details.model_dump_json(indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def download_file(course_name: str, filename: str) -> str:
    """Downloads a physical file from a course's materials to the user's local hard drive.
    Args:
        course_name: The exact name of the course.
        filename: The exact filename to download. (You MUST run get_full_course_details first to find the exact filename)."""
    try:
        details = db_manager.get_course_details(course_name)
        if not details: return "Course not found."
        
        for m in details.materials:
            if m.filename == filename:
                if not m.download_url:
                    return f"File '{filename}' exists but has no download link."
                
                print(f"\n[Agent] Triggering physical download for: {filename}")
                filepath = db_manager.scraper.download_specific_file(m.download_url, filename)
                if filepath:
                    return f"SUCCESS: File physically saved to {filepath}"
                else:
                    return "FAILED: Scraper encountered an error downloading the file."
        return "File not found in course materials."
    except Exception as e:
        return f"Error: {str(e)}"


tools = [
    sync_university_data,
    get_student_dashboard,
    get_detailed_profile,
    get_full_timetable,
    get_academic_history,
    get_invoices,
    get_notifications,
    get_exam_datesheet,
    get_full_course_details,
    download_file
]

# Initialize the LLM via Groq (Free and Lightning Fast)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

system_prompt = """You are 'Uni-Assistant', an elite, highly intelligent AI dedicated to managing the user's university portal data.

# YOUR CAPABILITIES (TOOLS)
You are equipped with 10 powerful tools that fetch real-time JSON data from the university database. 
The tool docstrings explain EXACTLY what data each tool returns. Read them carefully before deciding which tool to call.

# CRITICAL DIRECTIVES
1. BE CONVERSATIONAL & HELPFUL: Do not act like a robotic query engine. If the user says "Hello", greet them warmly. If they ask for advice, give it. 
2. NEVER HALLUCINATE: If the user asks for their grades, attendance, schedule, or personal info, you MUST call the appropriate tool(s) first. NEVER guess their data.
3. MULTI-TOOL MASTERY: Don't hesitate to chain tools. If the user asks "What is my teacher's name for English, and what is my CGPA?", you should call `get_full_course_details("English")` AND `get_student_dashboard()` to gather all the facts before responding.
4. DOWNLOADING FILES: If the user asks you to download a file, you MUST first use `get_full_course_details` to find the exact `filename`, and then pass that exact string to the `download_file` tool.
5. NO RAW JSON: When you output information to the user, format it beautifully using Markdown (bullet points, bold text, tables). NEVER just spit raw JSON strings back at the user."""

agent = create_react_agent(llm, tools=tools)

def chat_loop():
    print("\n" + "="*50)
    print("🎓 UNI-ASSISTANT SANDBOX READY (ULTIMATE EDITION)")
    print("="*50)
    print("I now have UNRESTRICTED access to 100% of the scraper's raw JSON data.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    chat_history = [{"role": "system", "content": system_prompt}]
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            chat_history.append({"role": "user", "content": user_input})
            
            print("\n[dim]Thinking...[/dim]")
            
            response = agent.invoke({"messages": chat_history})
            final_message = response["messages"][-1].content
            
            # Print Reasoning if available
            if hasattr(response["messages"][-1], "additional_kwargs") and "reasoning_content" in response["messages"][-1].additional_kwargs:
                print(f"[Reasoning]: {response['messages'][-1].additional_kwargs['reasoning_content']}")
                
            print(f"\nAssistant: {final_message}")
            chat_history.append({"role": "assistant", "content": final_message})
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_loop()
