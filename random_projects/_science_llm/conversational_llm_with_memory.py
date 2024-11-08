#   Goal.: local llm with who to talk to .

#   Sidebar: possibility to upload pdf/txt:
#       1. to 'train llm:' on paper (.txt)
#       2. and to talk about the paper with memory

import ollama

conversation = []

def stream_response(prompt):
    conversation.append({"role": "user", "content":prompt})
    response = ""
    stream = ollama.chat(model="llama3.2", messages=conversation, stream=True)
    print("\nAssistant: ")

    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)


    print("\n")
    conversation.append({"role": "user", "content":response})


while True:
    prompt = input("User: \n")
    stream_response(prompt)



#embeddings of previous conversations... (Memory)