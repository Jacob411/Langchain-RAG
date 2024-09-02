from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from generate_embedding_function import get_embedding_function


CHROMA_PATH = '/Users/jakesimmons/repos/Langchain-RAG/chromaDB'

embedding_function = get_embedding_function()


db = Chroma(collection_name="osp_initial_docs_V1", persist_directory=CHROMA_PATH, embedding_function=embedding_function)
docs = db.get(include=['metadatas'])
     
# for doc in docs:
#     print("Content: ",doc)
#     print("Score: ", doc[1])
#     print('----')


res = db.get(where={'id' : 341})
print(res)

