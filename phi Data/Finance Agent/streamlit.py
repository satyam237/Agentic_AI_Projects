import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Define Agents
web_agent = Agent(
    name="Web Agent",
    model=Gemini(id="gemini-2.0-flash", provider="google"),
    tools=[DuckDuckGo()],
    instructions=[
        "You are a Web Search Assistant specializing in retrieving and summarizing information from the internet.",
        "Always provide sources and links when answering queries.",
        "Prioritize recent and reliable sources for information.",
        "Summarize information concisely but provide details when necessary.",
        "If no relevant results are found, state that explicitly instead of guessing.",
        "For technical queries, include code snippets or structured outputs when applicable.",
        "When asked about yourself, explain that you are a Web Search Agent designed to find and summarize web content.",
    ],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=Gemini(id="gemini-2.0-flash", provider="google"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=[
        "You are a Finance Assistant specializing in financial data, stock market trends, and company information.",
        "Use tables to display financial data clearly.",
        "Include recent stock prices, analyst recommendations, and market trends where relevant.",
        "Avoid making financial predictions; instead, present historical data and expert analyses.",
        "When providing company news, include publication date and source.",
        "Use percentage changes and comparisons when analyzing stock trends.",
        "When asked about yourself, explain that you are a Finance Assistant providing stock and market insights.",
    ],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

agents = {"Web Agent": web_agent, "Finance Agent": finance_agent}

# Streamlit UI
st.set_page_config(page_title="AI Agent Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
    }
    .stButton > button {
        background-color: #4A90E2 !important;
        color: #FFFFFF !important;
        font-size: 18px !important;
        border-radius: 10px !important;
        padding: 10px !important;
        display: block !important;
        margin: 0 auto !important;
        border: 2px solid #FFFFFF !important;
    }

    .stButton > button:focus,
    .stButton > button:active {
        background-color: #4A90E2 !important;
        color: #FFFFFF !important;
        border: 2px solid #FFFFFF !important;
        outline: none !important;
    }

    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        padding: 10px;
    }
    .stMarkdown, .response-text {
        font-size: 20px;
        color: #ffffff;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🤖 AI Agent Assistant</div>", unsafe_allow_html=True)

st.sidebar.header("Select Agent")
selected_agent_name = st.sidebar.radio("Choose an agent:", list(agents.keys()))
selected_agent = agents[selected_agent_name]

st.sidebar.markdown("### Instructions")
st.sidebar.info("Select an agent, enter your prompt, and get an AI-generated response.")

st.markdown(f"### You are using: **{selected_agent_name}**")

prompt = st.text_area("Enter your query:", height=150)

if st.button("Get Response"):
    if prompt.strip():
        with st.spinner("Processing..."):
            response = selected_agent.run(prompt)
        
        st.markdown("## Response:")
        st.markdown(f"<div class='response-text'>{response.messages[-1].content}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a query before submitting.")
