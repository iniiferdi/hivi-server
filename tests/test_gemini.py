import asyncio
import time

from gemini import MODEL, generate_response


async def main():
    print(f"Model: {MODEL}")

    started = time.perf_counter()
    answer = await generate_response("Halo, perkenalkan dirimu singkat.")
    elapsed = time.perf_counter() - started

    print(f"Jawaban ({elapsed:.1f} detik): {answer}")


if __name__ == "__main__":
    asyncio.run(main())