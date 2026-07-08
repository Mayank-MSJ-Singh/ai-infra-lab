import asyncio
import httpx
import time

BASE_URL = "http://localhost:8000"
TOTAL_USERS = 1000
MAX_CONCURRENT_USERS = 1000  # Semaphore limit to prevent port/DB connection exhaustion

# Create a Semaphore to throttle active concurrent users
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_USERS)

# Track progress
completed_count = 0
progress_lock = asyncio.Lock()

async def simulate_user(client: httpx.AsyncClient, user_id: int):
    global completed_count
    
    async with SEMAPHORE:
        user_name = f"User_{user_id}"
        
        # Step 1: Create a unique conversation for this user
        try:
            conv_response = await client.post(f"{BASE_URL}/conversation/")
            conv_response.raise_for_status()
            conversation_id = conv_response.json()["conversation_id"]
        except Exception as e:
            print(f"[{user_name}] Failed to create conversation: {e}")
            return

        # Step 2: Introduce name
        try:
            response1 = await client.post(
                f"{BASE_URL}/chat/{conversation_id}",
                json={"message": f"Hello, my name is {user_name}"}
            )
            response1.raise_for_status()
        except Exception as e:
            print(f"[{user_name}] Msg 1 failed: {e}")
            return

        # Wait 1 second between messages
        await asyncio.sleep(1.0)

        # Step 3: Ask the AI for the name to verify session isolation
        try:
            response2 = await client.post(
                f"{BASE_URL}/chat/{conversation_id}",
                json={"message": "What is my name?"}
            )
            response2.raise_for_status()
            res2_json = response2.json()
            
            # Print periodically or if name verification fails
            response_text = res2_json.get('response', '')
            if user_name not in response_text:
                print(f"⚠️ [WARNING] Context leak? {user_name} got response: {response_text}")
                
        except Exception as e:
            print(f"[{user_name}] Msg 2 failed: {e}")
            return

        # Log progress
        async with progress_lock:
            completed_count += 1
            if completed_count % 50 == 0:
                print(f"Progress: {completed_count}/{TOTAL_USERS} users completed.")

async def main():
    print(f"Starting simulation of {TOTAL_USERS} users (throttled to {MAX_CONCURRENT_USERS} max concurrent requests)...")
    
    # Large timeout to handle queueing at the LLM backend
    timeout = httpx.Timeout(300.0, connect=30.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [simulate_user(client, i) for i in range(1, TOTAL_USERS + 1)]
        start_time = time.time()
        await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        print(f"\nSimulation complete in {total_duration:.2f}s.")

if __name__ == "__main__":
    asyncio.run(main())
