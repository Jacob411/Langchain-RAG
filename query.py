import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from generate_embedding_function import get_embedding_function
from langchain_openai import ChatOpenAI
import ollama




PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question} """
def overlap(a, b):
    return max(i for i in range(len(b)+1) if a.endswith(b[:i]))

def get_neighboring_context(db : Chroma, id : int, k = 1):
    retrieval_ids  = [i for i in range(id - k, id + k + 1)]
    print(retrieval_ids)
    # grab each id and concat
    res = ""

    for id in retrieval_ids:
        response = db.get(where={'id':id})
        if len(response['documents']) == 0:
            continue
        content = response['documents'][0]
        met = response['metadatas'][0]
        print(met)
        overlap1 = overlap(res, content)
        res += content[overlap1:]

    return res

def query(query_text : str, collection_name : str, view_context=True):

    print("Query:\n")
    print(query_text)
    CHROMA_PATH = '/Users/jakesimmons/repos/Langchain-RAG/chromaDB'

    embedding_function = get_embedding_function()
    db = Chroma(collection_name=collection_name, persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    results = db.similarity_search_with_score(query_text, k=3)
    ids = [doc.metadata['id'] for doc, _ in results]
    ids = set(ids)

    context_text = "\n\n---\n\n".join([get_neighboring_context(db, id, k=3) for id in ids])
    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    #
    # stream = ollama.chat(
    #     model='llama3.1',
    #     messages=[{'role': 'user', 'content': prompt}],
    #     stream=True,
    # )
    #
    # #print()
    # #print("Response:\n")
    # total = ''
    # for chunk in stream:
    #     #print(chunk['message']['content'], end='', flush=True)
    #     total += chunk['message']['content']
    # print()

    sources = [doc[0].metadata['source'] for doc in results]
    display_text = ""
    if view_context: 
        print(context_text)
        for doc in results:
            doc_name = doc[0].metadata['source'].split("/")[-1]
            page_num = doc[0].metadata['page']
            display_text += f"### Source \n  {doc_name}\n"
            display_text += f" **Page**  **{page_num}**\n"
            display_text += f"[Link to source](file://{doc[0].metadata['source']})\n"
            display_text += f"#### Selected text: \n  {doc[0].page_content}\n"
            display_text += f"Score:  {doc[1]}\n"
            display_text += "\n\n---\n\n"


    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(prompt)

    #print()
    # print(response.content)

    return response.content, display_text, sources


def main():
    collection_name = "osp_initial_docs_V1"
    query_text = input("Enter a question: ")

    reply = query(query_text, collection_name=collection_name)
    print(reply)

if __name__ == '__main__':
    main() 
