from langchain_openai import OpenAIEmbeddings


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        )

# Embedding of a single text
text = "Hello, world!"
embedding = embeddings.embed_query(text)
print(len(embedding))
