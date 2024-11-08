import streamlit as st
from langchain_ollama import ChatOllama


st.set_page_config(
    page_title="Llama Chat",
    layout="wide",
    initial_sidebar_state="auto"
)

def main():
    st.subheader("Ollama Playground", divider=True, anchor=False)
    client = ChatOllama(model="llama3.2")
    message_container = st.container(height=500, border=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        avatar = '🦕' if message["role"] == "assistant" else "😅"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter message.. "):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="😅").markdown(prompt)

            with message_container.chat_message("assistant", avatar='🦕'):
                with st.spinner("Llama thinks..."):
                    response = ""
                    for chunk in client.stream(input=[{"role":m["role"], "content": m["content"]}
                                                  for m in st.session_state.messages],
                                           stream=True):
                        response += chunk.content
                    st.write(response)
            st.session_state.messages.append(
                {"role":"assistant", "content":response}
            )
        except Exception as e:
            st.error(e)

if __name__ == "__main__":
    main()

