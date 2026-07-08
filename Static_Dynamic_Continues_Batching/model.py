from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="dummy",
    base_url="http://localhost:12434/engines/llama.cpp/v1",
)



class Chat():
    def __init__(self):
        self.messages = []
    
    async def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    async def get_response(self):
        response = await client.chat.completions.create(
            model="llama3.2",
            messages=self.messages,
        )
        return response.choices[0].message.content