from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from generate_embedding_function import get_embedding_function

CHROMA_PATH = '/home/jacob/senior_design/Langchain-RAG/chromaDB'

embedding_function = get_embedding_function()


db = Chroma(collection_name='first_collection', persist_directory=CHROMA_PATH, embedding_function=embedding_function)
docs = db.get()['documents']

for i, doc in enumerate(docs):
    print(f"Document {i}: {doc}")
    print("<------------------------------------------------------->")

query = input('Enter a question: ')

result = db.similarity_search_with_score(query, k=1)
print(result[0][0].page_content)
