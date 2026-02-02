import openai;
from dotenv import load_dotenv;

load_dotenv();

def generate_answer(context: list[str], query: str) -> str:
    context_str = "\n\n".join(context);

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": "You answer questions based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content;