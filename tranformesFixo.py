import json
import pickle
from sentence_transformers import SentenceTransformer

print("A carregar o catálogo fixo...")
with open("catalogo_fixo.json", "r", encoding="utf-8") as f:
    catalogo_fixo = json.load(f)

sinopses_fixas = [filme["overview"] for filme in catalogo_fixo]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_fixos = model.encode(sinopses_fixas, show_progress_bar=True)

ARQUIVO_CACHE = "embeddings_fixos.pkl"
with open(ARQUIVO_CACHE, "wb") as f:
    pickle.dump(embeddings_fixos, f)
