import torch
import tiktoken
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
enc=tiktoken.get_encoding("gpt2")

checkpoint=torch.load("gpt2_step_11500.pt",map_location="cpu")
E=checkpoint['_orig_mod.transformer.wte.weight']

tokens=[
    " health",
    " medical",
    " disease",
    " cancer",
   " nutrition",
   " wellness",

   " research",
    " study",
   " science",
    " data",
    " information",
    " evidence",

    " students",
    " children",
    " learning",
    " education",
    " school",

    " software",
    " computer",
    " internet",
    " technology",
    " database",
]

token_ids = []

for token in tokens:
    ids = enc.encode(token)

    if len(ids) != 1:
        print(f"{token} -> {ids}")

    token_ids.append(ids[0])

embeddings=E[token_ids]

pca=PCA(n_components=2)

embeddings_2d= pca.fit_transform(embeddings.numpy())

plt.figure(figsize=(12,8))

for i, token in enumerate(tokens):
    x=embeddings_2d[i,0]
    y=embeddings_2d[i,1]

    plt.scatter(x,y)
    plt.annotate(token.strip(),(x,y))

plt.title("PCA of GPT Token Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()




