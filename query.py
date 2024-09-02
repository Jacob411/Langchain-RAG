import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from generate_embedding_function import get_embedding_function
import ollama




PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def query(query_text : str, collection_name : str, view_context=False):

    print("Query:\n")
    print(query_text)
    CHROMA_PATH = '/Users/jakesimmons/repos/Langchain-RAG/chromaDB'

    embedding_function = get_embedding_function()
    db = Chroma(collection_name=collection_name, persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)


    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    print()
    print("Response:\n")
    total = ''
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        total += chunk['message']['content']


    print()
    view_context = input("view context? (y/n)")
    if view_context == 'y':
        for doc in results:
            print("Metadata: ", doc[0].metadata)
            print("Content: ",doc[0].page_content)
            print("Score: ", doc[1])
            print('<=====================================================>')       

    # from langchain_openai import ChatOpenAI

    # model = ChatOpenAI(model="gpt-4o-mini")
    # response = model.invoke(prompt)

    # print()
    # print(response.content)

    return total


def main():
    collection_name = "osp_initial_docs_V1"
    query_text = input("Enter a question: ")

    query(query_text, collection_name=collection_name)

if __name__ == '__main__':
    main() 
