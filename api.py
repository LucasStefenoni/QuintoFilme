import json
import os
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pickle
import gc
from sklearn.neighbors import NearestNeighbors

from transformers import AutoTokenizer
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tmdb import buscar_filme_por_id_no_tmdb, pegar_recomendacoes_por_id

app = FastAPI(title="API QuintoFilme")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MODEL_PATH = "model.onnx"
if not os.path.exists(MODEL_PATH):
    url_onnx = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    response_model = requests.get(url_onnx, stream=True)
    if response_model.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response_model.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise RuntimeError("Falha ao baixar o modelo ONNX do Hugging Face Hub.")
ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

def gerar_embedding_leve(texto: str):
    if not texto:
        texto = ""
    inputs = tokenizer(texto, padding=True, truncation=True, return_tensors="np")
    onnx_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    outputs = ort_session.run(None, onnx_inputs)
    return np.mean(outputs[0], axis=1)[0]


with open("embeddings_fixos.pkl", "rb") as f:
    embeddings_fixos = np.array(pickle.load(f)).astype(np.float32)

with open("catalogo_fixo.json", "r", encoding="utf-8") as f:
    filmes_catalogo = json.load(f)

knn = NearestNeighbors(n_neighbors=6, metric="cosine")
knn.fit(embeddings_fixos)


def pegar_favoritos_letterboxd(username: str):
    url = f"https://letterboxd.com/{username}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    section = soup.find('section', id='featured-films')
    if not section:
        return []
        
    favoritos = []
    for li in section.find_all('li', class_='poster-container'):
        img = li.find('img')
        film_id = li.find('div', class_='really-lazy-load')
        
        titulo_filme = img['alt'] if img else "Filme Desconhecido"
        id_tmdb = film_id['data-film-id'] if film_id else None
        
        favoritos.append({
            "titulo": titulo_filme,
            "id_tmdb": id_tmdb
        })
    return favoritos


@app.get("/")
def home():
    return {"status": "QuintoFilme API rodando perfeitamente!"}


@app.get("/recomendar/{username}")
def recomendar(username: str):
    try:
        favoritos = pegar_favoritos_letterboxd(username)
        
        if not favoritos:
            raise HTTPException(status_code=404, detail="Perfil não encontrado ou sem favoritos públicos.")
            
        resultados_finais = []
        
        for filme_fav in favoritos:
            dados_tmdb = buscar_filme_por_id_no_tmdb(filme_fav["id_tmdb"])
            sinopse_filme = dados_tmdb.get("overview", "")
            
            vetor_usuario = gerar_embedding_leve(sinopse_filme).reshape(1, -1)
            
            distancias, indices = knn.kneighbors(vetor_usuario)
            
            lista_recomendacoes = []
            for i in range(len(indices[0])):
                idx_catalogo = indices[0][i]
                distancia = distancias[0][i]
                
                porcentagem_match = int((1 - distancia) * 100)
                
                filme_recomendado = filmes_catalogo[idx_catalogo]
                
                lista_recomendacoes.append({
                    "titulo": filme_recomendado["title"], 
                    "match": porcentagem_match
                })
            
            resultados_finais.append({
                "filme_favorito": filme_fav["titulo"],
                "recomendacoes": lista_recomendacoes[:5]
            })
            
        gc.collect()
        
        return {"usuario": username, "resultados": resultados_finais}
        
    except HTTPException as http_err:
        gc.collect()
        raise http_err
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")
