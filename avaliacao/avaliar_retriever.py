"""Mede a qualidade da busca do RAG, sem gastar nenhuma chamada de LLM.

Para cada pergunta rotulada, verifica em que posição aparece o trecho correto.
Métricas:
- recall@1: o trecho certo veio em 1º lugar
- recall@3: o trecho certo está entre os 3 primeiros (o agente recebe 3 trechos)
- MRR: média de 1/posição do trecho certo (1,0 = sempre em 1º)

Uso:  python avaliacao/avaliar_retriever.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agente_delivery.retriever import EmbeddingRetriever, TfidfRetriever, carregar_trechos  # noqa: E402

PERGUNTAS = Path(__file__).with_name("perguntas_rag.jsonl")


def avaliar(retriever, perguntas):
    r1 = r3 = mrr = 0
    erros = []
    for item in perguntas:
        fontes = [t.fonte for t in retriever.buscar(item["pergunta"], k=len(retriever.trechos))]
        posicao = fontes.index(item["fonte_esperada"]) + 1
        r1 += posicao == 1
        r3 += posicao <= 3
        mrr += 1 / posicao
        if posicao > 3:
            erros.append((item["pergunta"], item["fonte_esperada"], fontes[0]))
    n = len(perguntas)
    return {"recall@1": r1 / n, "recall@3": r3 / n, "MRR": mrr / n}, erros


def main():
    perguntas = [json.loads(linha) for linha in PERGUNTAS.read_text(encoding="utf-8").splitlines() if linha.strip()]
    trechos = carregar_trechos()
    variantes = {
        "TF-IDF palavras": lambda: TfidfRetriever(trechos, modo="palavras"),
        "TF-IDF caracteres": lambda: TfidfRetriever(trechos, modo="caracteres"),
        "Embeddings (MiniLM multilíngue)": lambda: EmbeddingRetriever(trechos),
    }

    print(f"{len(perguntas)} perguntas, {len(trechos)} trechos de política\n")
    print(f"| Retriever | recall@1 | recall@3 | MRR |\n|---|---|---|---|")
    todos_erros = {}
    for nome, construir in variantes.items():
        try:
            retriever = construir()
        except ImportError:
            print(f"| {nome} | (instale sentence-transformers) | | |")
            continue
        m, erros = avaliar(retriever, perguntas)
        todos_erros[nome] = erros
        print(f"| {nome} | {m['recall@1']:.0%} | {m['recall@3']:.0%} | {m['MRR']:.2f} |")

    for nome, erros in todos_erros.items():
        if erros:
            print(f"\nFalhas fora do top 3 — {nome}:")
            for pergunta, esperado, obtido in erros:
                print(f'  - "{pergunta}"\n    esperado: {esperado}\n    veio:     {obtido}')


if __name__ == "__main__":
    main()
