import ollama


def main():
    stream= ollama.chat(
        model="mistral",
        messages=[
            {"role": "user", "content": "tell me a joke"}],
        stream=True
    )
    
    for chunk in stream:
        print(chunk["message"]['content'], end="", flush=True)  # Print each chunk's content
    
    print("\n")

if __name__ == "__main__":
    main()