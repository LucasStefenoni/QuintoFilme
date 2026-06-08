import requests
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para o sistema
load_dotenv()

# Puxa o token do ambiente de forma segura
TOKEN_TMDB = os.getenv("TMDB_TOKEN")

global headers
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TOKEN_TMDB}"
}
        

def pesquisar_pagina_filmes(resp):
    if resp.status_code == 200:
        dados_brutos = resp.json()
        print(dados_brutos)
        lista_filmes = dados_brutos['results']

        filmes_pagina = []
        for filme in lista_filmes:
            info = {
                "id": filme["id"],
                "title": filme["title"],
                "overview": filme[
                    "overview"
                ],
                "genres": filme["genre_ids"],
                "popularity": filme["popularity"],
            }
            filmes_pagina.append(info)

        return filmes_pagina
    
    else:
        print(f"Erro na API: {resp.status_code}")
        return None
    
def salvar_filmes_categoria(categoria, quantidade_paginas):
    URL = f'https://api.themoviedb.org/3/movie/{categoria}'
    filmes_da_categoria = []
    for i in range(1, quantidade_paginas + 1):
        params = {"language": "en-US", "page": i}
        response = requests.get(URL, headers=headers, params=params)
        filmes_da_categoria.extend(pesquisar_pagina_filmes(response))
    return filmes_da_categoria

def descobrir_id_filme(slug):
    termo_busca = slug.replace("-", " ")
    
    URL = "https://api.themoviedb.org/3/search/movie/"
    params = { "query": termo_busca, "language": "en-US"}
    
    response = requests.get(URL, headers=headers, params=params)
    
    if response.status_code == 200:
        dados = response.json()
        resultados = dados.get("results", [])
        
        if resultados:
            filme = resultados[0]
            
            return {
                "id": filme["id"],
                "title": filme["title"],
                "overview": filme[
                    "overview"
                ],
                "genres": filme["genre_ids"],
                "popularity": filme["popularity"],
            }
            
    print(f"Não foi possível encontrar o ID para: {slug}")
    return None

def buscar_filme_por_id_no_tmdb(id_filme):
    URL = f"https://api.themoviedb.org/3/movie/{id_filme}"
    
    params = {
        "language": "en-US"
    }
    
    response = requests.get(URL, headers=headers, params=params)
    
    if response.status_code == 200:
        filme = response.json()
        
        return {
            "id": filme["id"],
            "title": filme["title"],
            "overview": filme.get("overview", ""),
            "genres": [g["id"] for g in filme.get("genres", [])],
            "popularity": filme.get("popularity", 0),
        }
    else:
        print(f"Erro ao buscar o ID {id_filme} no TMDB (Status: {response.status_code})")
        return None
    
def pegar_recomendacoes_por_id(id_filme):
    URL = f"https://api.themoviedb.org/3/movie/{id_filme}/recommendations"
    
    params = {
        "language": "en-US",
        "page": 1
    }
    
    response = requests.get(URL, headers=headers, params=params)
    
    if response.status_code == 200:
        return pesquisar_pagina_filmes(response)
    else:
        print(f"Erro ao buscar recomendações para o ID {id_filme} (Status: {response.status_code})")
        return []