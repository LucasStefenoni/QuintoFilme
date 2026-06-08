from tmdb import buscar_filme_por_id_no_tmdb, pegar_recomendacoes_por_id
import json
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API QuintoFilme")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def salvar(filme, ids_salvos, lista_destino):
    if filme["id"] not in ids_salvos:
        lista_destino.append(filme)
        ids_salvos.add(filme["id"])

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
        print(f"Erro ao acessar o perfil: {response.status_code}")
        return []
        
    soup = BeautifulSoup(response.text, "html.parser")
    
    secao_favoritos = soup.find("section", class_="section", id="favourites")
    
    if not secao_favoritos:
        print("Nenhum filme favorito encontrado ou o perfil é privado.")
        return []
        
    componentes_filmes = secao_favoritos.find_all(
        "div", class_="react-component"
    )

    filmes_favoritos = []

    for comp in componentes_filmes:
        titulo_filme = comp.get("data-item-name")
        slug_filme = comp.get("data-item-slug")

        if titulo_filme and slug_filme:
            filmes_favoritos.append(
                {"title": titulo_filme, "slug": slug_filme}
            )

    return filmes_favoritos

def salvar_dados_usuario(url, ids, titulos):
    return  {
        "url_letterboxd": url,
        "ids_favoritos": ids,
        "titulos_favoritos": titulos
    }

def ler_json(arquivo):
    try:
        with open(arquivo, "r", encoding="utf-8") as f:
            catalogo = json.load(f)
        return catalogo
    except FileNotFoundError:
         print(f"Erro ao abrir {arquivo}")
         return {}
    
def ler_pickle(arquivo):
    try:
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Erro: O arquivo {arquivo} não foi encontrado!")
        exit()

model = SentenceTransformer("all-MiniLM-L6-v2")    
catalogo_fixo = ler_json("catalogo_fixo.json")
ids_salvos = {filme['id'] for filme in catalogo_fixo}
embeddings_fixos = ler_pickle("embeddings_fixos.pkl")

@app.get("/recomendar/{username}")
def recomendar_filmes(username: str):
    try:
        url_usuario = f"https://letterboxd.com/{username}/"
        favoritos = pegar_favoritos_letterboxd(url_usuario)
        if not favoritos:
            raise HTTPException(status_code=404, detail="Nenhum favorito encontrado para este usuário.")
        ids_favoritos = []
        titulos_favoritos = []

        print("Dados do usuário pronto")

        for fav in favoritos:
            id_tmdb = pegar_id_tmdb_direto(fav['slug'])
            if id_tmdb:
                ids_favoritos.append(id_tmdb)
                titulos_favoritos.append(fav['title'])

        dados_usuario = salvar_dados_usuario(url_usuario, ids_favoritos, titulos_favoritos)
        filmes = []

        ids_salvos_sessao = {filme['id'] for filme in catalogo_fixo}

        for id_filme in ids_favoritos:
            filme_completo = buscar_filme_por_id_no_tmdb(id_filme) 
            if filme_completo:
                salvar(filme_completo, ids_salvos_sessao, filmes)

            recomendados_tmdb = pegar_recomendacoes_por_id(id_filme)

            if recomendados_tmdb:
                for filme in recomendados_tmdb:
                    salvar(filme, ids_salvos_sessao, filmes)

        todos_filmes = catalogo_fixo + filmes
        sinopses_variaveis = [filme["overview"] for filme in filmes]

        if sinopses_variaveis:
            embeddings_variaveis = model.encode(sinopses_variaveis, show_progress_bar=False)
        else:
            embeddings_variaveis = np.empty((0, 384))

        matriz_embeddings = np.vstack([embeddings_fixos, embeddings_variaveis])
        knn = NearestNeighbors(n_neighbors=10, metric="cosine")
        knn.fit(matriz_embeddings)

        print("KNN Treinado")

        id_para_linha_matriz = {filme["id"]: idx for idx, filme in enumerate(todos_filmes)}

        resposta_api = {
            "usuario": username,
            "resultados": []
        }

        for id_fav, titulo_favorito in zip(dados_usuario["ids_favoritos"], dados_usuario["titulos_favoritos"]):
            idx_favorito = id_para_linha_matriz.get(id_fav)
            if idx_favorito is None:
                continue 
                
            vetor_filme = matriz_embeddings[idx_favorito].reshape(1, -1)
            distancias, indices = knn.kneighbors(vetor_filme, n_neighbors=10)
            
            recomendacoes_deste_filme = []
            for i in range(1, len(indices[0])):
                idx_vizinho = indices[0][i]
                distancia_cosseno = distancias[0][i]
                
                filme_recomendado = todos_filmes[idx_vizinho]
                distancia_cosseno_float = float(distancia_cosseno)
                match = round((1 - distancia_cosseno_float) * 100)   
                
                recomendacoes_deste_filme.append({
                    "titulo": filme_recomendado['title'],
                    "match": match,
                    "id_tmdb": filme_recomendado['id']
                })
                
            resposta_api["resultados"].append({
                "filme_favorito": titulo_favorito,
                "recomendacoes": recomendacoes_deste_filme
            })

        return resposta_api

    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))