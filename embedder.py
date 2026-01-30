import openai;
from dotenv import load_dotenv;

load_dotenv();

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        model = "text-embedding-3-small",
        input = texts,
    )
    return [item.embedding for item in response.data];

if __name__ == "__main__":
    texts = ["Hello, world!", "Hello, world!", "Hello, world!"];
    embeddings = get_embeddings(texts);
    print(embeddings);