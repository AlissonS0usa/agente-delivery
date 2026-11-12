import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "avaliacao"))

from agente_delivery.ferramentas import Ferramentas  # noqa: E402
from agente_delivery.retriever import criar_retriever  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return criar_retriever("tfidf")


@pytest.fixture
def ferramentas(retriever):
    return Ferramentas("c001", retriever)


class LLMFalso:
    """Imita o cliente da OpenAI devolvendo respostas roteirizadas, sem rede.

    Cada item do roteiro é um texto (resposta final) ou uma lista de
    (nome_da_ferramenta, argumentos_json) para simular tool calls.
    """

    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.chamadas = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.chamadas.append(kwargs)
        passo = self.roteiro.pop(0) if self.roteiro else "fim"
        if isinstance(passo, str):
            msg = SimpleNamespace(content=passo, tool_calls=None)
        else:
            msg = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=nome, arguments=args))
                    for i, (nome, args) in enumerate(passo)
                ],
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(total_tokens=100))


@pytest.fixture
def llm_falso():
    return LLMFalso
