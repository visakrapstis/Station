import ollama
import time

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#   Initialize the Paper that you like
loader = TextLoader("C:/Users/V. Stasiunaitis/Desktop/Projects/Data Science/Data Science. Projects/___FAISS_to_optimize/publication.txt")
documents = loader.load()

#   Splitting the Document
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=0,
)

#print(documents[0].page_content)
docs = text_splitter.split_documents(documents)

# Initialize the HuggingFace embedding model (IDK why, but after 1,5days it is the only model that functioned)


from langchain.embeddings import HuggingFaceEmbeddings
"""#!!!   from langchain_huggingface import HuggingFaceEmbeddings"""

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Use Chroma with this embedding model (had a lot of problems with this database earlier, but it fixed itself now...)
library = Chroma.from_documents(documents=docs, embedding=embedding_model)
#print(library)

retriever = library.as_retriever()

#   Initialize the llm
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.2")

#   Prompting:  The fun Part (RAG=feeding the papers..)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


"""# 3. simple Answer
print("The answer before RAG\n")
start_time = time.time()
before_rag_template = "What is {topic}"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | llm | StrOutputParser()
print(before_rag_chain.invoke({"topic": "summary of recent Advancement in biomedical research"}))
end_time = time.time()
print("the answer took {:.2f} seconds".format(end_time-start_time))"""
from langchain_core.runnables import RunnablePassthrough


# 4. After RAG
print("\n########\nThe answer after RAG\n")
start_time = time.time()
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | llm
    | StrOutputParser()
)
print(after_rag_chain.invoke("Please give me a summary on recent Advancement in biomedical research"))
end_time = time.time()
print("the answer took {:.2f} seconds".format(end_time-start_time))


#       Learn about the code.
#       To upgrade: Memory