"""RAG: busca dos trechos de política mais relevantes para uma pergunta.

1. Chunking: cada seção "## ..." de cada arquivo .md vira um trecho. As políticas
   são curtas e organizadas por tema, então uma seção é a unidade natural de
   significado (nem pequena demais, nem misturando assuntos).
2. Indexação: duas opções intercambiáveis
   - TfidfRetriever: palavras em comum (leve, roda em qualquer lugar)
   - EmbeddingRetriever: similaridade semântica com sentence-transformers
     (entende sinônimos, ex.: "estorno" ~ "reembolso")
3. Busca: retorna os k trechos com maior similaridade de cosseno.

O script avaliacao/avaliar_retriever.py mede o recall@k das duas opções.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MODELO_EMBEDDINGS, PASTA_DADOS, RETRIEVER


@dataclass
class Trecho:
    fonte: str  # ex.: "reembolso.md > Item faltando ou item errado"
    texto: str
    score: float = 0.0


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para "reembolso" e "Reembolso" baterem."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def carregar_trechos(pasta: Path = PASTA_DADOS / "politicas") -> list[Trecho]:
    trechos = []
    for arquivo in sorted(pasta.glob("*.md")):
        conteudo = arquivo.read_text(encoding="utf-8")
        titulo_doc = conteudo.splitlines()[0].lstrip("# ").strip()
        # divide nas seções de nível 2 ("## Título")
        for secao in re.split(r"^## ", conteudo, flags=re.MULTILINE)[1:]:
            titulo, _, corpo = secao.partition("\n")
            trechos.append(
                Trecho(
                    fonte=f"{arquivo.name} > {titulo.strip()}",
                    # o título do documento entra no texto para dar contexto ao trecho
                    texto=f"{titulo_doc} - {titulo.strip()}: {corpo.strip()}",
                )
            )
    return trechos


class TfidfRetriever:
    """TF-IDF em dois modos:
    - "palavras": compara palavras inteiras (e pares de palavras)
    - "caracteres": compara pedaços de 3 a 5 letras, então "preparando" e "preparo"
      ou "traço" e "traços" ainda se parecem (um stemming "de graça")
    """

    def __init__(self, trechos: list[Trecho], modo: str = "caracteres"):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.trechos = trechos
        if modo == "palavras":
            self.vetorizador = TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)
        else:
            self.vetorizador = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                               ngram_range=(3, 5), sublinear_tf=True)
        self.matriz = self.vetorizador.fit_transform([t.texto for t in trechos])

    def buscar(self, pergunta: str, k: int = 3) -> list[Trecho]:
        consulta = self.vetorizador.transform([pergunta])
        scores = (self.matriz @ consulta.T).toarray().ravel()  # vetores já normalizados -> cosseno
        return _top_k(self.trechos, scores, k)


class EmbeddingRetriever:
    def __init__(self, trechos: list[Trecho], modelo: str = MODELO_EMBEDDINGS):
        from sentence_transformers import SentenceTransformer

        self.trechos = trechos
        self.modelo = SentenceTransformer(modelo)
        self.matriz = self.modelo.encode([t.texto for t in trechos], normalize_embeddings=True)

    def buscar(self, pergunta: str, k: int = 3) -> list[Trecho]:
        consulta = self.modelo.encode([pergunta], normalize_embeddings=True)[0]
        scores = self.matriz @ consulta
        return _top_k(self.trechos, scores, k)


def _top_k(trechos: list[Trecho], scores: np.ndarray, k: int) -> list[Trecho]:
    ordem = np.argsort(-scores)[:k]
    return [Trecho(trechos[i].fonte, trechos[i].texto, round(float(scores[i]), 3)) for i in ordem]


def criar_retriever(tipo: str = RETRIEVER):
    trechos = carregar_trechos()
    if tipo == "embeddings":
        return EmbeddingRetriever(trechos)
    return TfidfRetriever(trechos)
