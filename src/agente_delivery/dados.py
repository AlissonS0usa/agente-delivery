"""Carrega a base simulada: cardápio, pedidos e o "relógio" do sistema.

O horário atual é fixo (campo "agora" em pedidos.json). Isso torna as regras de
atraso e de prazo de reclamação reproduzíveis: a avaliação dá sempre o mesmo
resultado, não importa o dia em que você rode.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .config import PASTA_DADOS


@dataclass
class Base:
    agora: datetime
    restaurantes: dict
    pratos: dict
    pedidos: dict
    clientes: dict = field(default_factory=dict)

    def total_pedido(self, pedido: dict) -> float:
        itens = sum(self.pratos[i["prato_id"]]["preco"] * i["qtd"] for i in pedido["itens"])
        taxa = self.restaurantes[pedido["restaurante_id"]]["taxa_entrega"]
        return round(itens + taxa, 2)


def _data(valor):
    return datetime.fromisoformat(valor) if valor else None


@lru_cache(maxsize=None)
def carregar_base(pasta: Path = PASTA_DADOS) -> Base:
    cardapio = json.loads((pasta / "cardapio.json").read_text(encoding="utf-8"))
    bruto = json.loads((pasta / "pedidos.json").read_text(encoding="utf-8"))

    pedidos = {}
    for p in bruto["pedidos"]:
        p = dict(p)
        for campo in ("criado_em", "previsao_entrega", "entregue_em"):
            p[campo] = _data(p[campo])
        pedidos[p["id"]] = p

    return Base(
        agora=_data(bruto["agora"]),
        restaurantes={r["id"]: r for r in cardapio["restaurantes"]},
        pratos={p["id"]: p for p in cardapio["pratos"]},
        pedidos=pedidos,
        clientes={c["id"]: c for c in bruto["clientes"]},
    )
