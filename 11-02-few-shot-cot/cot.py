import os,re
from collections import Counter
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")

def call_llm(client, model, system, user, temperature=None):
    # The Claude API no longer accepts temperature. The argument stays only
    # so the lesson's functions can still pass it without errors.
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[
            {"role": "user", "content": user},
        ],
    )
    return response.content[0].text

if __name__ == "__main__":
    print(call_llm(client, MODEL, "Reply in one word.", "Say hello."))