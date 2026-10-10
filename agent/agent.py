from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response_classes import StreamEventType
from context.context_manager import ContextManager


class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()

    async def run(self,message:str):
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)

        final_response: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

        
    async def _agentic_loop(self):
        # messages = [{"role":"user","content":"write a short para on cricket"}]
        response_text=""
        async for event in self.client.chat_completion(self.context_manager.get_messages(), True):
            # print("---event in agent.py",event)
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text+=content
                    yield AgentEvent.text_delta(content)
                
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(
                    event.error or "Unknown error occured"
                )
        
        self.context_manager.add_assistant_message(response_text or "")
        
        if response_text:
            yield AgentEvent.text_complete(response_text)
                    
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client:
            await self.client.close()
            self.client = None