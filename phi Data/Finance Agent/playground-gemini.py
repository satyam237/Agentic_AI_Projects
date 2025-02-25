from phi.agent import Agent, RunResponse
from phi.model.google import Gemini
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
import os
from phi.tools.duckduckgo import DuckDuckGo

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


# Web Agent using Gemini
web_agent = Agent(
    name="Web Agent",
    model=Gemini(id="gemini-2.0-flash", provider="google"),
    tools=[DuckDuckGo()],
    instructions=[
        "Always provide sources and links when answering queries.",
        "Prioritize recent and reliable sources for information.",
        "Summarize information concisely but provide details when necessary.",
        "If no relevant results are found, state that explicitly instead of guessing.",
        "For technical queries, include code snippets or structured outputs when applicable.",
    ],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

# Finance Agent using Gemini
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
        "Use tables to display financial data clearly.",
        "Include recent stock prices, analyst recommendations, and market trends where relevant.",
        "Avoid making financial predictions; instead, present historical data and expert analyses.",
        "When providing company news, include publication date and source.",
        "Use percentage changes and comparisons when analyzing stock trends.",
    ],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)
finance_agent.print_response("Which stock is better, apple or tesla for tomorrow.")