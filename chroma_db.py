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
def overlap(a, b):
    return max(i for i in range(len(b)+1) if a.endswith(b[:i]))


def get_neigboring_context(db : Chroma, id : int, k = 1, chunk_overlap=80):
    retrieval_ids  = [i for i in range(id - k, id + k + 1)]
    print(retrieval_ids)
    # grab each id and concat
    res = ""

    for id in retrieval_ids:
        content = db.get(where={'id':id})['documents'][0]
        overlap1 = overlap(res, content)
        res += content[overlap1:]

    return res

res = get_neigboring_context(db, 150)
print(res)
