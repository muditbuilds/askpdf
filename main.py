from retriever import retrieve;
from generator import generate_answer;
from ingest import ingest;
import psycopg2;
from pgvector.psycopg2 import register_vector;

import os;
from dotenv import load_dotenv;
load_dotenv();

def ask(conn, question: str) -> str:
    context = retrieve(conn, question);
    if not context:
        return "No relevant information found."
    answer = generate_answer(context, question);
    return answer;

if __name__ == "__main__":
    print("Welcome to the PDF Q&A system! (\q to quit)");
    pdf_path = input("Enter the path to the PDF file: ");
    if pdf_path == "\q":
        exit();
    conn = psycopg2.connect(os.getenv("DATABASE_URL"));
    register_vector(conn);
    ingest(conn, pdf_path);
    while True:
        question = input("Enter a question: ");
        if question == "\q":
            break;
        answer = ask(conn,question);
        print(answer);
    conn.close();