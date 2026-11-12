"""Configurações lidas de variáveis de ambiente (arquivo .env).

O cliente de LLM usa a API compatível com OpenAI. Assim, dá para trocar de
provedor só mudando LLM_BASE_URL e LLM_MODEL:

- Groq (padrão, gratuito):  https://api.groq.com/openai/v1
- Gemini:                   https://generativelanguage.googleapis.com/v1beta/openai/
- Ollama (local):           http://localhost:11434/v1
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv é opcional
    pass

RAIZ = Path(__file__).resolve().parents[2]
PASTA_DADOS = RAIZ / "data"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# "tfidf" (leve, sem dependências pesadas) ou "embeddings" (sentence-transformers)
RETRIEVER = os.getenv("RETRIEVER", "tfidf")
MODELO_EMBEDDINGS = os.getenv(
    "MODELO_EMBEDDINGS", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Cliente "logado" na conversa. O agente só enxerga os pedidos desse cliente.
CLIENTE_ID = os.getenv("CLIENTE_ID", "c001")

# Limite de voltas do agente (chamada de ferramenta -> resposta) por mensagem.
MAX_PASSOS = int(os.getenv("MAX_PASSOS", "6"))
