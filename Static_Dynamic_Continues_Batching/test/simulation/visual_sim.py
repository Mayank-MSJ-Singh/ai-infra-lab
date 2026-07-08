import asyncio
import httpx
import time
import argparse
import sys
from tqdm.asyncio import tqdm

BASE_URL = "http://localhost:8000"

async def simulate_user(client: httpx.AsyncClient, user_id: int, sem: asyncio.Semaphore, stats: dict):
    user_name = f"User_{user_id}"
    
    async with sem:
        # Step 1: Create conversation
        try:
            resp = await client.post(f"{BASE_URL}/conversation/")
            resp.raise_for_status()
            conv_id = resp.json()["conversation_id"]
            stats["conv_created"] += 1
        except Exception as e:
            stats["errors"] += 1
            return

        # Step 2: Introduce name
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/{conv_id}",
                json={"message": f"Hello, my name is {user_name}. Please remember it carefully."}
            )
            resp.raise_for_status()
            stats["msg1_success"] += 1
        except Exception as e:
            stats["errors"] += 1
            return

        # Wait a tiny bit between messages to simulate human typing
        await asyncio.sleep(0.5)

        # Step 3: Verify name isolation
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/{conv_id}",
                json={"message": "What is my name? Please just say my name and nothing else."}
            )
            resp.raise_for_status()
            stats["msg2_success"] += 1
            
            response_text = resp.json().get('response', '')
            if user_name not in response_text:
                stats["context_leaks"] += 1
            else:
                stats["isolated"] += 1
                
        except Exception as e:
            stats["errors"] += 1

async def main():
    parser = argparse.ArgumentParser(description="Visual LLM Batching & Concurrency Simulation")
    parser.add_argument("--users", type=int, default=100, help="Total number of simulated users")
    parser.add_argument("--concurrency", type=int, default=30, help="Max concurrent connections (throttle)")
    
    # Parse known args so it doesn't fail in interactive environments
    args, _ = parser.parse_known_args()

    print("="*60)
    print(f"🚀 VISUAL SIMULATION: {args.users} Users | {args.concurrency} Concurrency")
    print("="*60)
    
    sem = asyncio.Semaphore(args.concurrency)
    stats = {
        "conv_created": 0,
        "msg1_success": 0,
        "msg2_success": 0,
        "context_leaks": 0,
        "isolated": 0,
        "errors": 0
    }
    
    timeout = httpx.Timeout(300.0, connect=30.0)
    
    start_time = time.time()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [simulate_user(client, i, sem, stats) for i in range(1, args.users + 1)]
        
        # Use tqdm to show a visual progress bar
        # For standard terminals, ASCII mode usually looks best and most reliable
        await tqdm.gather(*tasks, desc="Simulating Users", unit="user", ascii=False, dynamic_ncols=True)
        
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("📊 SIMULATION REPORT")
    print("="*60)
    print(f"Total Time Taken     : {total_time:.2f} seconds")
    print(f"Users Attempted      : {args.users}")
    print(f"Conversations Created: {stats['conv_created']}")
    print(f"Successful Chats     : {stats['msg2_success']}")
    
    if stats['context_leaks'] == 0:
        print(f"Context Leaks        : 0 ✅ (Perfect Isolation)")
    else:
        print(f"Context Leaks        : {stats['context_leaks']} ❌ (Data leakage detected!)")
        
    print(f"Perfect Isolations   : {stats['isolated']}")
    print(f"Connection Errors    : {stats['errors']}")
    print("="*60)

if __name__ == "__main__":
    # Fix for windows / specific async environments
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
