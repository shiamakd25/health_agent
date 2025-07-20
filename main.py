from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import os
import asyncio
import pymupdf
import streamlit as st

load_dotenv(override=True)

# Model
gemini_client = AsyncOpenAI(
    base_url=os.getenv("GEMINI_BASE_URL"), api_key=os.getenv("GEMINI_API_KEY")
)
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=gemini_client
)

# Agents

diagnoser_instructions = (
    "You are a health agent who makes diagnoses based on medical information you receive. "
    "There are three categories under which your work will fall. You will receive data from a blood report in the form of a JSON, "
    "image data from an X-Ray, or a list of symptoms. Based on this information you will diagnose possible medical conditions the user may have. "
    "A user may also ask you what certain medical terms mean, to which you will reply with a simple definition which can be  understandable without a medical background. "
    "When giving medical information, make sure to tell the user to speak with a qualified medical professional at the end of whatever you are saying."
)
diagnoser_agent = Agent(
    name="Diagnoser", instructions=diagnoser_instructions, model=gemini_model
)


@function_tool
def extract_text(file_path: str):
    """Extract text from a pdf file at the given path"""
    text = ""
    try:
            doc = pymupdf.open(file_path)
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                text += page.get_text()
    except Exception as e:
            print(f"Error extracting text: {e}")
            return {"status": "failure"}
    return text


blood_report_agent_tools = [extract_text]

blood_report_instructions = (
    "You are responsible for taking a given blood report PDF file, extracting text, and then processing the data from the blood report and its extracted text and putting it into a JSON format. "
    "When you introduce yourself, do not introduce yourself by name, simply say something along the lines of 'I can help you with that'. "
    "To both open and extract text, you must use the extract_text tool. This tool will either return the text or a failure message. In the case of a failure message, simply hand that message off to the diagnoser_agent. "
    "If you receive the extracted text, you must process the text and remove any unnecessary information. The only necessary information is the actual information about the blood. All other information can be removed. "
    "Once you process the information, you must put it into a JSON format. This JSON will then be handed off to the diagnoser_agent. Your job is complete after sending the JSON to the diagnoser_agent."
)

blood_report_agent = Agent(
    name="Blood Report Parser",
    instructions=blood_report_instructions,
    handoffs=[diagnoser_agent],
    tools=blood_report_agent_tools,
    model=gemini_model,
)

router_instructions = (
    "You are the entry point agent for a medical diagnosing chatbot. "
    "You receive a user text input which will identify what kind of diagnosis they are looking for. This will be either diagnosing a blood report in the form of a PDF, X-Ray in an image format, or user-inputted symptoms in natural language. "
    "If the user wants a blood report to be analyzed, you must handoff to the blood_report_agent. Otherwise, you can handoff to the diagnoser agent. "
    "You do not make diagnoses by yourself, you simply route requests to the appropriate agent. "
    "Once routing to the appropriate agent, your job is complete. If you are not able to route a request, then you can ask the user to enter an appropriate input. "
)

router_agent = Agent(
    name="Router",
    instructions=router_instructions,
    handoffs=[blood_report_agent, diagnoser_agent],
    model=gemini_model,
)

# UI

st.title("Medical Diagnosis Chatbot")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "current_agent_name" not in st.session_state:
    st.session_state.current_agent_name = "Router"

user_input = st.text_input(
    "Describe your issue (e.g., 'I have a blood report to diagnose',  'I have the following symptoms:...')",
    key="input",
)
submit_text = st.button("Submit Text")

if submit_text and user_input.strip():
    result = asyncio.run(Runner.run(router_agent, user_input))
    print(dir(result))
    st.session_state.current_agent_name = result.last_agent.name
    st.session_state.conversation_history.append(f"🧑‍💻 You: {user_input}")
    st.session_state.conversation_history.append(
        f"🤖 {result.last_agent.name}: {result.final_output}"
    )
    st.write(f"🤖 {result.last_agent.name}: {result.final_output}")


if st.session_state.current_agent_name == "Blood Report Parser":
    uploaded_file = st.file_uploader("Upload your blood report (PDF)", type=["pdf"])
    if uploaded_file is not None and st.button("Submit File"):
        try:
            file_path = "upload/report.pdf"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            input_data = f"The blood report is located at: {file_path}"
            result = asyncio.run(Runner.run(router_agent, input_data))
            st.session_state.current_agent_name = result.last_agent.name
            st.session_state.conversation_history.append("📄 File uploaded: blood report")
            st.session_state.conversation_history.append(f"🤖 {result.last_agent.name}: {result.final_output}")
            st.write(f"🤖 {result.last_agent.name}: {result.final_output}")

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


st.markdown("---")
st.subheader("Conversation History")
for message in st.session_state.conversation_history:
    st.write(message)

if st.button("Restart"):
    st.session_state.current_agent_name = "Router"
    st.session_state.conversation_history = []
