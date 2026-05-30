
import os
from openai import AsyncOpenAI

# Initialisierung des asynchronen Clients (erwartet OPENAI_API_KEY in den Umgebungsvariablen)
client = AsyncOpenAI(api_key=os.environ.get(os.getenv("OPENAI_API_KEY")))

async def transform_go_query(prompt,) -> str:
    try:
        response = await client.responses.create(
            model="gpt-5.5",
            input=prompt,
            temperature = 0.1,
            max_tokens = 40,
        )
        print(response.output_text)
        transformed_concepts = response.output_text
        return transformed_concepts
    except Exception as e:
        print(f"Error during OpenAI Query Transformation: {e}")
        return f"biological process, {label}"




