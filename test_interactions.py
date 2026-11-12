import asyncio
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

async def main():
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"timeout": 60000}
    )

    try:
        print("Mengirim request melalui Interactions API...")

        interaction = await asyncio.wait_for(
            client.aio.interactions.create(
                model="gemini-3.8-flash",
                input="Halo Hivi, perkenalkan dirimu dalam satu kalimat."
            ),
            timeout=70
        )

        print("\nSTATUS:", interaction.status)
        print("RESPONS:", interaction.output_text)

    except asyncio.TimeoutError:
        print("\nERROR: Request mengalami timeout.")

    except Exception as e:
        print("\nERROR:", type(e).__name__)
        print("DETAIL:", repr(e))

    finally:
        await client.aio.aclose()

asyncio.run(main())
