import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("catalogo_fixo.json", "r", encoding="utf-8") as f:
    catalogo_fixo = json.load(f)
try:
    with open("catalogo_variavel.json", "r", encoding="utf-8") as f:
        catalogo_variavel = json.load(f)
except FileNotFoundError:
    catalogo_variavel = []

todos_filmes = catalogo_fixo + catalogo_variavel

try:
    with open("embeddings_fixos.pkl", "rb") as f:
        embeddings_fixos = pickle.load(f)
except FileNotFoundError:
    print("Erro: O ficheiro 'embeddings_fixos.pkl' não foi encontrado!")
    print("Por favor, corre o script 'gerar_cache_fixo.py' primeiro para criar o cache.")
    exit()

    
sinopses_variaveis = [filme["overview"] for filme in catalogo_variavel]

if sinopses_variaveis:
    print(f"A processar {len(sinopses_variaveis)} novos filmes do catálogo variável na memória...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_variaveis = model.encode(sinopses_variaveis, show_progress_bar=False)
else:
    embeddings_variaveis = np.empty((0, 384))

matriz_embeddings = np.vstack([embeddings_fixos, embeddings_variaveis])
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(matriz_embeddings)


nome_arquivo = "usuario_atual.json"

try:
    with open("usuario_atual.json", "r", encoding="utf-8") as f:
        dados_usuario = json.load(f)
    
    id_para_linha_matriz = {filme["id"]: idx for idx, filme in enumerate(todos_filmes)}

except FileNotFoundError:
    print(f"Erro: O ficheiro '{nome_arquivo}' não foi encontrado. Corre o script de scraping primeiro!")

for id_fav, titulo_favorito in zip(dados_usuario["ids_favoritos"], dados_usuario["titulos_favoritos"]):
    
    idx_favorito = id_para_linha_matriz.get(id_fav)
    
    if idx_favorito is None:
        continue 
        
    vetor_filme = matriz_embeddings[idx_favorito].reshape(1, -1)
    
    distancias, indices = knn.kneighbors(vetor_filme, n_neighbors=10)
    
    print(f"\nComo você gostou de '{titulo_favorito}':")
    
    for i in range(1, len(indices[0])):
        idx_vizinho = indices[0][i]
        distancia_cosseno = distancias[0][i]
        
        filme_recomendado = todos_filmes[idx_vizinho]
        match = round((1 - distancia_cosseno) * 100, 2)
        
        print(f"  -> {filme_recomendado['title']} ({match}% de match)")