import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from langchain_openai import OpenAIEmbeddings
import json

# Open json file which contains a list of texts
texts = json.load(open("visualization/genres.json"))
texts = [text["name"] for text in texts]
print(texts)

embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        )

# Embedding of a single text
embeddings = embeddings.embed_documents(texts)
embeddings = torch.tensor(embeddings)
print(embeddings.shape)

experiment_name = "Spotify_Genres_run1"
writer = SummaryWriter(log_dir=f"runs/{experiment_name}")



#convert to tensor
embeddings = torch.tensor(embeddings)

writer.add_embedding(embeddings, metadata=texts)

writer.close()
