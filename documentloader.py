import os
from langchain_community.document_loaders import DirectoryLoader
import glob

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

def main():
    directory = '/home/jacob/senior_design/Langchain-RAG/confidential_data/Projects/'
