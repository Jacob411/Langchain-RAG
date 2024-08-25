from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
    )
    # embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    return embedding_function
