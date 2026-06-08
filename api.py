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

knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(embeddings_fixos)

def pegar_id_tmdb_direto(slug_filme):
    url = f"https://letterboxd.com/film/{slug_filme}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
        
    soup = BeautifulSoup(response.text, "html.parser")
    corpo = soup.find("body")
    if corpo and corpo.has_attr("data-tmdb-id"):
        return int(corpo["data-tmdb-id"])
        
    link_tmdb = soup.find("a", href=re.compile(r"themoviedb\.org/movie/"))
    if link_tmdb:
        match = re.search(r"/movie/(\d+)", link_tmdb["href"])
        if match:
            return int(match.group(1))
            
    return None

def pegar_favoritos_letterboxd(url): 
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
        
    soup = BeautifulSoup(response.text, "html.parser")
    secao_favoritos = soup.find("section", class_="section", id="favourites")
    
    if not secao_favoritos:
        return []
        
    componentes_filmes = secao_favoritos.find_all("div", class_="react-component")
    filmes_favoritos = []

    for comp in componentes_filmes:
        titulo_filme = comp.get("data-item-name")
        slug_filme = comp.get("data-item-slug")

        if titulo_filme and slug_filme:
            filmes_favoritos.append({"title": titulo_filme, "slug": slug_filme})

    return filmes_favoritos

@app.get("/")
def home():
    return {"status": "QuintoFilme API rodando perfeitamente!"}

@app.get("/recomendar/{username}")
def recomendar(username: str):
    try:
        url_usuario = f"https://letterboxd.com/{username}/"
        favoritos = pegar_favoritos_letterboxd(url_usuario)
        
        if not favoritos:
            raise HTTPException(status_code=404, detail="Perfil não encontrado ou sem favoritos públicos.")
            
        resultados_finais = []
        
        for fav in favoritos:
            id_tmdb = pegar_id_tmdb_direto(fav['slug'])
            if not id_tmdb:
                continue
                
            dados_tmdb = buscar_filme_por_id_no_tmdb(id_tmdb)
            sinopse_filme = dados_tmdb.get("overview", "")
            
            vetor_usuario = gerar_embedding_leve(sinopse_filme).reshape(1, -1)
            
            distancias, indices = knn.kneighbors(vetor_usuario, n_neighbors=6)
            
            lista_recomendacoes = []
            for i in range(len(indices[0])):
                idx_catalogo = indices[0][i]
                distancia = distancias[0][i]
                
                porcentagem_match = int((1 - float(distancia)) * 100)
                filme_recomendado = filmes_catalogo[idx_catalogo]
                
                lista_recomendacoes.append({
                    "titulo": filme_recomendado["title"],
                    "match": porcentagem_match
                })
            
            resultados_finais.append({
                "filme_favorito": fav["title"],
                "recomendacoes": lista_recomendacoes[1:6]
            })
            
        gc.collect()
        return {"usuario": username, "resultados": resultados_finais}
        
    except HTTPException as http_err:
        gc.collect()
        raise http_err
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")
