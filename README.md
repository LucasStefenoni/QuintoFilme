# QuintoFilme

Link do projeto: https://quinto-filme.vercel.app/

---

## Descrição do Projeto
O QuintoFilme extrai os filmes favoritos da sua conta do Letterboxd e, por meio do algoritmo de classificação KNN, localiza quais são os possíveis filmes que você pode gostar, encontrando o seu "quinto filme" favorito.

---

## Tecnologias e Ferramentas
* **Linguagem:** Python
* **Framework Web:** FastAPI
* **Machine Learning / IA:** Scikit-learn & Transformers
* **Web Scraping:** BeautifulSoup
* **Consumo de Dados:** TMDB API
* **Hospedagem & Nuvem:** Render (Back-end) & Vercel (Front-end)

---

## Arquitetura e Passo a Passo
1. **Catálogo Fixo:** A aplicação possui um catálogo estático contendo os filmes mais famosos e os títulos que estão em alta no momento na API do TMDB, estruturado em arquivos formato JSON e `.pkl` (gerados previamente via Transformers).
2. **Catálogo Variável:** O sistema realiza a raspagem dos filmes favoritos no perfil do Letterboxd do usuário e cria um catálogo dinâmico, complementado com recomendações obtidas diretamente do TMDB.
3. **Processamento de Embeddings:** É realizado o cálculo dos embeddings do catálogo variável através de Onnx (por ser mais leve), juntando os vetores com matriz do catálogo fixo.
4. **Análise de Proximidade (KNN):** O algoritmo KNN (*K-Nearest Neighbors*) do Scikit-learn é executado para calcular a distância de cosseno entre as sinopses (embeddings).
5. **Seleção de Recomendações:** O modelo seleciona os 5 elementos mais próximos de cada filme, ignorando ele próprio (o que seria 100% de match).

---

## Diretrizes de Uso e Limitações

### Observações Importantes
> * **Identificação do Usuário:** É necessário informar o nome de usuário exato utilizado na estrutura da URL do seu perfil do Letterboxd, e não o nome de exibição textual que aparece no topo da página.
> * **Disponibilidade da API:** Um mecanismo de *cronjob* configurado externamente realiza requisições periódicas para evitar que a instância da API entre em modo de hibernação (*sleep*).

### Limitações do Sistema
Devido à utilização do plano de acesso pessoal da API do TMDB e à infraestrutura da camada gratuita do Render, a aplicação possui restrições de concorrência. Caso ocorram falhas ou timeouts no processamento, o comportamento pode estar associado ao limite de requisições simultâneas suportadas pelo ambiente.

---

## Demonstração da Interface

<img width="835" height="418" alt="image" src="https://github.com/user-attachments/assets/49ecd5b2-2253-4c93-abd0-a8c8f20ac9c6" />

<img width="813" height="439" alt="image" src="https://github.com/user-attachments/assets/629fa007-0b5b-4745-9205-9a7680c1a934" />

<img width="1324" height="609" alt="image" src="https://github.com/user-attachments/assets/ddc8132e-761c-470a-8e08-8d4f29e8291b" />
