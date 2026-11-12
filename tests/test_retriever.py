import json
from pathlib import Path

from avaliar_retriever import avaliar


def test_pergunta_direta_acha_a_secao_certa(retriever):
    assert retriever.buscar("em quanto tempo cai o estorno no pix", k=1)[0].fonte == "pagamentos.md > Pix"


def test_recall_minimo_nao_piora(retriever):
    """Trava de regressão: se uma mudança piorar a busca, o teste avisa."""
    arquivo = Path(__file__).resolve().parents[1] / "avaliacao" / "perguntas_rag.jsonl"
    perguntas = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]
    metricas, _ = avaliar(retriever, perguntas)
    assert metricas["recall@3"] >= 0.80
