from client.llm_client import LLMClient
import asyncio

async def main():
    client = LLMClient()
    messages = [
        {"role":"user","content":"Whats up"}
    ]

    await client.chat_completion(messages,False)

asyncio.run(main())