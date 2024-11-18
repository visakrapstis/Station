import ollama

conversation = []
conversation.append({"role": "user", "content": "## you are the crypto sentiment expert. "
                                                "##your task is to analyse the sentiment of the question and:"
                                                "ANSWER ONLY positive, neutral or negative."
                                                "##NEVER answer anything else than one of those three words. "})

def stream_response(prompt):
    conversation.append({"role": "user", "content": prompt})

    response = ""
    stream = ollama.chat(model="mistral", messages=conversation, stream=True)
    print("\nAssistant: ")

    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="")

    conversation.append({"role": "ai", "content": response})


while True:
    print("\nUser: ")
    prompt = input()
    if prompt == "-quit":
        quit()
    elif "-print -conversation" in prompt:
        for saying in conversation:
            print(saying)
    else:
        stream_response(prompt)
