from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

from generate_embedding_function import get_embedding_function

import glob


CHROMA_PATH = '/Users/jakesimmons/repos/Langchain-RAG/chromaDB'

def display_files(directory):
    files = glob.glob(f'{directory}/**/*.pdf', recursive=True)
    print(f'Found {len(files)} pdf files')
    files = glob.glob(f'{directory}/**/*.docx', recursive=True)
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

def pypdf_load_documents(directory):

    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdfs = loader.load()

    print(pdfs)
    print("len of pdfs: ", len(pdfs))
    return pdfs

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

def add_chunk_ids(chunks : list[Document]):
    for i, chunk in enumerate(chunks):
        chunk.metadata['id'] = i
    return chunks

def create_chroma_db(chunks : list[Document], persist_directory : str, collection_name : str):
    embedding_function = get_embedding_function()

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    vector_store.add_documents(documents=chunks)

    return vector_store



def main():
    directory = '/Users/jakesimmons/repos/Langchain-RAG/docs/osp_docs'
    collection_name = "osp_initial_docs_V1"
    docs = pypdf_load_documents(directory)
    chunks = chunk_documents(docs)
    chunks = add_chunk_ids(chunks)
    db = create_chroma_db(chunks, CHROMA_PATH, collection_name=collection_name)

    query = input('enter a query: ')
    res = db.similarity_search_with_score(query, k=5)

    print(res)

if __name__ == '__main__':
    main()
