"""Mini CRM Lead Qualifier demo app."""

import json
from typing import Dict, Any, List

from agents import RunContextWrapper, TResponseInputItem, Runner

from .schema import CRMRunContext
from .agent import agent as crm_agent


class CRMChatService:
    """Session-scoped chat service that maintains multi-turn message history.

    Only manages model input history; tool calls and data are captured in the
    per-run context provided by the Veris SDK / agents context.
    """

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.input_items: List[TResponseInputItem] = []
        self.ctx = RunContextWrapper(CRMRunContext())

    async def process_message(self, message: str) -> Dict[str, Any]:
        # Append user message to rolling transcript
        self.input_items.append({"content": message, "role": "user"})

        # Run with full history so the agent has context
        result = await Runner.run(
            starting_agent=crm_agent,
            input=self.input_items,
            context=self.ctx.context,
        )

        # Refresh canonical history to include assistant/model outputs
        self.input_items = result.to_input_list()

        print("result.to_input_list()\n", json.dumps(result.to_input_list(), indent=2))

        reply_text = str(result.final_output) if result.final_output is not None else ""

        return {
            "reply": reply_text,
            "tool_calls": list(self.ctx.context.tool_calls),
            "data": dict(self.ctx.context.data),
        }


# In-memory registry of session services
_session_services: Dict[str, CRMChatService] = {}


def get_crm_chat_service(session_id: str) -> CRMChatService:
    service = _session_services.get(session_id)
    if service is None:
        service = CRMChatService(session_id)
        _session_services[session_id] = service
    return service


