from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = '/home/jacob/senior_design/Langchain-RAG/chromaDB'
embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small",
)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

query_text = "What is the capital of France?"
results = db.similarity_search_with_score(query_text, k=5)
print(results)

