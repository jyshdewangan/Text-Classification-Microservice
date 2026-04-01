import asyncio
import aiohttp
import time
import numpy as np

async def fetch(session, url, data):
    start = time.time()
    async with session.post(url, json=data) as response:
        result = await response.json()
        # Also grab the process time from headers if available
        # server_time = float(response.headers.get("X-Process-Time", 0))
        return (time.time() - start) * 1000  # Return roundtrip latency in ms

async def main():
    url = "http://127.0.0.1:8000/predict"
    data = {
        "texts": [
            "This movie was absolutely wonderful, I loved every minute. A true masterpiece of cinema.", 
            "Terrible acting and the plot made no sense. Complete waste of time, do not watch.",
            "It was an okay film, not great but not terrible either. The acting was acceptable.",
            "I was on the edge of my seat the entire time. A true masterpiece.",
            "Such a boring experience. I fell asleep halfway through.",
            "The visual effects were stunning, but the storyline lacked depth.",
            "A brilliant performance by the lead actor, totally deserves an Oscar.",
            "One of the worst movies I have ever seen. Save your money.",
            "It started off slow but the climax was incredible! Highly recommended.",
            "Utter garbage. I regret spending time on this."
        ]
    }
    
    # Fire off concurrent requests to simulate load
    concurrency_level = 50
    requests_per_worker = 4
    total_requests = concurrency_level * requests_per_worker
    
    print(f"Stress testing ML Inference Service...")
    print(f"Target: {url}")
    print(f"Concurrency: {concurrency_level} parallel workers")
    print(f"Total Requests: {total_requests}")
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = [fetch(session, url, data) for _ in range(total_requests)]
        latencies = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Calculate latency statistics
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"\nCompleted {total_requests} requests in {total_time:.2f}s")
    print("\n--- Latency Under Concurrent Load ---")
    print(f"Average: {avg:.2f} ms")
    print(f"P50:     {p50:.2f} ms")
    print(f"P95:     {p95:.2f} ms")
    print(f"P99:     {p99:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
