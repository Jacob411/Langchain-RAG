from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from generate_embedding_function import get_embedding_function


CHROMA_PATH = '/Users/jakesimmons/repos/Langchain-RAG/chromaDB'

embedding_function = get_embedding_function()


db = Chroma(collection_name="osp_initial_docs_V1", persist_directory=CHROMA_PATH, embedding_function=embedding_function)
docs = db.get()['documents']
#
# for i, doc in enumerate(docs):
#     print(f"Document {i}: {doc}")
#     print("<------------------------------------------------------->")
#
query = input('Enter a question: ')

result = db.similarity_search_with_score(query, k=5)
    
for doc in result:
    print("Metadata: ", doc[0].metadata)
    print("Content: ",doc[0].page_content)
    print("Score: ", doc[1])
    print('----')
