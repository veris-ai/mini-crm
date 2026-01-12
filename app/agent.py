"""Mini CRM Agent - supports any frontier LLM via LiteLLM.

Configure your model via the MODEL environment variable:
  - OpenAI:    MODEL=gpt-4o (or MODEL=openai/gpt-4o)
  - Anthropic: MODEL=litellm/anthropic/claude-sonnet-4-20250514
  - Google:    MODEL=litellm/gemini/gemini-2.0-flash
  - Mistral:   MODEL=litellm/mistral/mistral-large-latest
  - DeepSeek:  MODEL=litellm/deepseek/deepseek-chat

Set the corresponding API key for your provider:
  - OpenAI:    OPENAI_API_KEY
  - Anthropic: ANTHROPIC_API_KEY
  - Google:    GEMINI_API_KEY
  - Mistral:   MISTRAL_API_KEY
  - DeepSeek:  DEEPSEEK_API_KEY
"""

import os

from agents import Agent
from agents.models.multi_provider import MultiProvider

from .tools import get_leads, lookup_lead, score_lead_industry, write_lead_update


# Default model - can be overridden via MODEL env var
DEFAULT_MODEL = "gpt-4o"


def get_model_name() -> str:
    """Get model name from environment, defaulting to gpt-4o."""
    return os.getenv("MODEL", DEFAULT_MODEL)


# Create the multi-provider that routes based on model prefix
model_provider = MultiProvider()


instructions = (
    "You are a sales assistant. Use tools to find leads, apply a simple industry score, "
    "update status/notes, then confirm by listing relevant leads with `get_leads`. "
    "Keep responses short and businesslike."
)


agent = Agent(
    name="Mini CRM Lead Qualifier",
    instructions=instructions,
    tools=[lookup_lead, score_lead_industry, write_lead_update, get_leads],
    model=get_model_name(),
)


