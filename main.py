from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatMistralAI(model = "mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""You are a helpful AI assistant that answers questions based on the provided context.
         if the answer is not in the context, say: "I could not find the answer in the document."
         """),(
            "human",
            """
            Context: {context}
            Question: {question}
            """
         )
         
    ]
)

print("RAG system created")
while True:
    query = input("you:")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)
    print("AI:", response.content)