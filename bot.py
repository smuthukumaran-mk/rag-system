"""
Main chatbot application for the College Information Assistant.

Run:
    python app.py
"""



from transformers import AutoTokenizer
from langchain_ollama import ChatOllama
from query import retrieve_documents


from config import (
    OLLAMA_MODEL,
)



# ==========================================================
# Connect to Ollama
# ==========================================================

print("Connecting to Ollama...")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

# ==========================================================
# Load Tokenizer
# ==========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

print("\n==========================================")
print(" College Information Assistant")
print(" Type 'exit' to quit")
print("==========================================")

# ==========================================================
# Chat Loop
# ==========================================================

while True:

    question = input("\nYou : ").strip()

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    # --------------------------------------------
    # Retrieve Top-K Similar Chunks
    # --------------------------------------------

    context, sources = retrieve_documents(question)


    # ---------- DEBUG ----------
    print("\n========== Retrieved Context ==========\n")
    print(context)
    print("=======================================\n")


    # --------------------------------------------
    # Prompt
    # --------------------------------------------

    prompt = f"""
You are a helpful College Information Assistant.

Instructions:
- Answer ONLY from the Context.
- Do NOT use external knowledge.
- If the Context does not contain the answer, reply exactly:
  "I couldn't find this information in the college documents."
- If the question is about a person, organization, department, course, event, or document mentioned in the Context, answer using only those details.
- Keep the answer concise (2-5 sentences).
- Do not mention the Context or these instructions in your response.

Context:
{context}

Question:
{question}

Answer:
"""

    # --------------------------------------------
    # Count Input Tokens
    # --------------------------------------------

    input_tokens = len(tokenizer.encode(prompt))

    # --------------------------------------------
    # Generate Answer
    # --------------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------------
    # Count Output Tokens
    # --------------------------------------------

    output_tokens = len(tokenizer.encode(response.content))
    total_tokens = input_tokens + output_tokens

    # --------------------------------------------
    # Display Answer
    # --------------------------------------------

    print("\nAssistant:\n")
    print(response.content)

    print(f"\nInput Tokens  : {input_tokens}")
    print(f"Output Tokens : {output_tokens}")
    print(f"Total Tokens  : {total_tokens}")

    print("\nSources:")

    for index, source in enumerate(dict.fromkeys(sources), start=1):
        print(f"{index}. {source}")