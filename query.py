import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
#from langchain_community.llms.ollama import Ollama
from generate_embedding_function import get_embedding_function
import ollama


query_text = input("Enter a question: ")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

CHROMA_PATH = '/home/jacob/senior_design/Langchain-RAG/chromaDB'

embedding_function = get_embedding_function()
db = Chroma(collection_name='first_collection', persist_directory=CHROMA_PATH, embedding_function=embedding_function)


results = db.similarity_search_with_score(query_text, k=5)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)

print(prompt)

# stream = ollama.chat(
#     model='llama3.1:8b',
#     messages=[{'role': 'user', 'content': prompt}],
#     stream=True,
# )
#
# print()
# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)
#

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(prompt)

print()
print(response.content)



