from tmdb import salvar_filmes_categoria
import json

if __name__ == "__main__":
    QTDPAGINAS = 150
    categorias = ["popular", "top_rated"]
    filmes = []
    ids_salvos = set()

    for categoria in categorias:
        filmes_da_categoria = salvar_filmes_categoria(categoria, QTDPAGINAS)
        for filme in filmes_da_categoria:
            if filme["id"] not in ids_salvos:
                ids_salvos.add(filme["id"])
                filmes.append(filme)

    with open("catalogo_fixo.json", "w", encoding="utf-8") as f:
        json.dump(filmes, f, ensure_ascii=False, indent=4)

    print(f"Pronto! Arquivo gerado com sucesso. Total de {len(filmes)} filmes salvos.")