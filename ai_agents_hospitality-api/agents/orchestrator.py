import asyncio
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from util.logger_config import logger
from config.agent_config import get_agent_config
from agents.bookings_sql_agent import run_bookings_analytics

try:
    # LangChain v0.2+
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    # LangChain v0.1 fallback
    from langchain_classic.prompts import ChatPromptTemplate


async def handle_orchestrator(user_query: str) -> str:
    """
    Async wrapper for orchestrator to be used in WebSocket / API.
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, orchestrator_agent, user_query
    )
    return response


def orchestrator_agent(user_query: str) -> str:
    """
    Orchestrator agent that routes user queries to the appropriate agent.
    """

    try:
        # ============================================================
        # 🔥 FAST PATH — Direct routing to SQL Agent (no LLM required)
        # ============================================================
        if "booking" in user_query.lower():
            logger.info("Routing directly to Bookings SQL Agent")
            return run_bookings_analytics(user_query)

        # ============================================================
        # 🔵 LLM-based routing (RAG / future agents)
        # ============================================================
        chain = _create_orchestrator_chain()
        logger.info(f"Processing RAG question: {user_query[:100]}...")

        result = chain.invoke({"user_query": user_query})

        # Defensive parsing: LLM may return text, JSON, or garbage
        try:
            decision = json.loads(result.content)
            agent = decision.get("agent")
        except Exception:
            logger.warning("Invalid JSON returned by orchestrator LLM")
            return "❌ Unable to determine the correct agent for this question."

        if agent == "Bookings SQL Agent":
            return run_bookings_analytics(user_query)

        elif agent == "RAG Agent":
            return "⚠️ RAG Agent not available (hotel data not loaded)."

        else:
            return "❌ Unknown agent selected by orchestrator."

    except Exception as e:
        logger.error("Error processing orchestrator question", exc_info=True)
        return f"❌ **Error**: {str(e)}"


def _create_orchestrator_chain():
    """
    Creates the LLM chain used only to decide which agent to run.
    """
    config = get_agent_config()

    llm = ChatGoogleGenerativeAI(
        model=config.model,
        temperature=0,
        google_api_key=config.api_key
    )

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a smart Orchestrator Agent.

Available agents:
1. Bookings SQL Agent
2. RAG Agent

Task:
- Given a user query, decide which agent is the best fit.
- Only choose ONE agent.
- Return ONLY a JSON with this exact structure:
{
  "agent": "Bookings SQL Agent" | "RAG Agent"
}
"""
        ),
        ("human", "{user_query}")
    ])

    return prompt_template | llm
