# pyrefly: ignore [missing-import]
import httpx



reply = httpx.get("http://localhost:8000/chat/hello, my name is nikki")

print(reply.json())

reply = httpx.get("http://localhost:8000/chat/hello,what is my name")

print(reply.json())