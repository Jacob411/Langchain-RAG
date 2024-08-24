import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import glob


CHROMA_PATH = '/home/jacob/senior_design/Langchain-RAG/chromaDB'

def display_files():
    files = glob.glob('**/*.pdf', recursive=True)
    print(f'Found {len(files)} pdf files')
    files = glob.glob('**/*.docx', recursive=True)
    print(f'Found {len(files)} docx files')

def load_documents(directory):
    docx_loader = DirectoryLoader(directory, glob="**/*.docx")
    docxs = docx_loader.load()
    pdf_loader = DirectoryLoader(directory, glob="**/*.pdf")
    pdfs = pdf_loader.load()

    print(f'Loaded {len(docxs)} docx files')
    print(f'Loaded {len(pdfs)} pdf files')

    all_files = docxs + pdfs

    print(f'Loaded {len(all_files)} documents')
    return all_files

def chunk_documents(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Chunked into {len(chunks)} chunks')
    return chunks

def create_chroma_db(chunks : list[Document], persist_directory : str):
    embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
    )

    vector_store = Chroma(
        collection_name="first_collection",
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    vector_store.add_documents(documents=chunks)

    return vector_store



def main():
    directory = '/home/jacob/senior_design/Langchain-RAG/confidential_data/Projects/'
    docs = load_documents(directory)
    chunks = chunk_documents(docs)
    chunks = chunks[:10]
    db = create_chroma_db(chunks, CHROMA_PATH)
    print('Created Chroma DB')

    query_text = "What is the capital of France?"
    results = db.similarity_search_with_score(query_text, k=5)

    print('Closest matches to query: ', query_text)
    print(results)

    retrieved = db.get(include=[])
    print('retrieved: ',retrieved)

if __name__ == '__main__':
    main()
