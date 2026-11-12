"""Ferramentas (tools) que o agente pode chamar.

Princípio central: **as regras de negócio ficam no código, não no prompt.**
O LLM decide QUAL ferramenta chamar e com quais argumentos; quem decide se um
reembolso é aprovado, quanto vale ou se um pedido pode ser cancelado é este
arquivo. Assim, mesmo que o modelo seja convencido a "dar um reembolso extra"
(ex.: prompt injection), a ferramenta recusa.

Outras proteções implementadas aqui:
- Privacidade: cada instância só enxerga os pedidos do cliente logado. Um pedido
  de outra pessoa recebe a mesma resposta de um pedido inexistente, para não
  revelar que ele existe.
- Idempotência: o mesmo reembolso não é concedido duas vezes na conversa.
- Confirmação: cancelar exige confirmação explícita do cliente.
"""

from __future__ import annotations

import copy
from datetime import timedelta

from .dados import Base, carregar_base
from .retriever import normalizar

PRAZO_ESTORNO = {
    "cartao_credito": "em até 2 faturas do cartão de crédito",
    "cartao_debito": "em até 7 dias úteis na conta",
    "pix": "em até 2 dias úteis na conta de origem do Pix",
    "vale_refeicao": "em até 5 dias úteis no saldo do vale-refeição",
}
JANELA_RECLAMACAO = timedelta(hours=24)
NAO_ENCONTRADO = {"erro": "Pedido não encontrado na sua conta. Confira o número do pedido."}
MOTIVOS_REEMBOLSO = ["item_faltando", "item_errado", "atraso", "qualidade", "nao_recebido"]


class Ferramentas:
    def __init__(self, cliente_id: str, retriever, base: Base | None = None):
        self.base = base or carregar_base()
        self.cliente_id = cliente_id
        self.retriever = retriever
        # cópia por conversa: um cancelamento numa conversa não afeta as outras
        self.pedidos = copy.deepcopy(self.base.pedidos)
        self.reembolsos_concedidos: set[tuple] = set()
        self.atendimentos: list[dict] = []

    # ------------------------------------------------------------------ helpers
    def _pedido_do_cliente(self, pedido_id) -> dict | None:
        pedido = self.pedidos.get(str(pedido_id).strip().lstrip("#"))
        if pedido is None or pedido["cliente_id"] != self.cliente_id:
            return None
        return pedido

    def _atraso_minutos(self, pedido: dict) -> int | None:
        if pedido["status"] == "cancelado":
            return None
        referencia = pedido["entregue_em"] or self.base.agora
        minutos = (referencia - pedido["previsao_entrega"]).total_seconds() / 60
        return max(0, int(minutos))

    def _compensacao_atraso(self, pedido: dict) -> dict | None:
        """Calcula no código o direito a compensação por atraso.

        Antes, o modelo lia a política e tirava a conclusão sozinho, e errava quando
        o RAG não trazia o trecho certo. Agora o fato já vem pronto na consulta do pedido.
        """
        atraso = self._atraso_minutos(pedido)
        if not atraso:
            return None
        if (pedido["id"], "atraso") in self.reembolsos_concedidos:
            return {"situacao": "compensação por atraso já concedida nesta conversa"}
        if atraso <= 30:
            return {"tem_direito": False, "motivo": "atrasos de até 30 minutos não geram compensação"}
        direito = "cupom de R$ 10,00" + (" + reembolso da taxa de entrega" if atraso > 60 else "")
        return {"tem_direito": True, "direito": direito,
                "como_conceder": "chame solicitar_reembolso com motivo 'atraso' se o cliente quiser"}

    def _itens_com_nome(self, pedido: dict) -> list[dict]:
        return [{"prato_id": i["prato_id"], "nome": self.base.pratos[i["prato_id"]]["nome"]} for i in pedido["itens"]]

    def _protocolo(self, motivo: str, pedido_id=None) -> dict:
        protocolo = f"ATD-{2001 + len(self.atendimentos)}"
        self.atendimentos.append({"protocolo": protocolo, "motivo": motivo, "pedido_id": pedido_id})
        return {
            "encaminhado_para_humano": True,
            "protocolo": protocolo,
            "mensagem": "Um atendente humano vai continuar o atendimento pelo chat do app.",
        }

    # -------------------------------------------------------------- ferramentas
    def consultar_pedido(self, pedido_id: str | None = None) -> dict:
        if not pedido_id:
            meus = [p for p in self.pedidos.values() if p["cliente_id"] == self.cliente_id]
            meus.sort(key=lambda p: p["criado_em"], reverse=True)
            return {
                "pedidos": [
                    {
                        "pedido_id": p["id"],
                        "restaurante": self.base.restaurantes[p["restaurante_id"]]["nome"],
                        "status": p["status"],
                        "criado_em": p["criado_em"].strftime("%d/%m %H:%M"),
                    }
                    for p in meus
                ]
            }

        pedido = self._pedido_do_cliente(pedido_id)
        if pedido is None:
            return NAO_ENCONTRADO

        itens = []
        for item in pedido["itens"]:
            prato = self.base.pratos[item["prato_id"]]
            itens.append({"prato_id": prato["id"], "nome": prato["nome"], "qtd": item["qtd"], "preco_unitario": prato["preco"]})

        fmt = lambda d: d.strftime("%d/%m %H:%M") if d else None  # noqa: E731
        restaurante = self.base.restaurantes[pedido["restaurante_id"]]
        return {
            "pedido_id": pedido["id"],
            "restaurante": restaurante["nome"],
            "status": pedido["status"],
            "itens": itens,
            "taxa_entrega": restaurante["taxa_entrega"],
            "total": self.base.total_pedido(pedido),
            "pagamento": pedido["pagamento"],
            "criado_em": fmt(pedido["criado_em"]),
            "previsao_entrega": fmt(pedido["previsao_entrega"]),
            "entregue_em": fmt(pedido["entregue_em"]),
            "atraso_minutos": self._atraso_minutos(pedido),
            "compensacao_por_atraso": self._compensacao_atraso(pedido),
            "agora": fmt(self.base.agora),
        }

    def buscar_cardapio(
        self,
        consulta: str | None = None,
        restaurante: str | None = None,
        sem_alergenicos: list[str] | None = None,
        apenas_veganos: bool = False,
        preco_max: float | None = None,
    ) -> dict:
        termos = normalizar(consulta).split() if consulta else []
        excluir = {normalizar(a).replace(" ", "_") for a in (sem_alergenicos or [])}
        resultados = []
        for prato in self.base.pratos.values():
            rest = self.base.restaurantes[prato["restaurante_id"]]
            texto = normalizar(f'{prato["nome"]} {prato["descricao"]} {rest["nome"]} {rest["categoria"]}')
            if termos and not any(t in texto for t in termos):
                continue
            if restaurante and normalizar(restaurante) not in normalizar(rest["nome"]):
                continue
            if excluir & set(prato["alergenicos"]):
                continue
            if apenas_veganos and not prato["vegano"]:
                continue
            if preco_max is not None and prato["preco"] > preco_max:
                continue
            resultados.append(
                {
                    "prato_id": prato["id"],
                    "nome": prato["nome"],
                    "restaurante": rest["nome"],
                    "preco": prato["preco"],
                    "alergenicos_declarados": prato["alergenicos"],
                    "vegano": prato["vegano"],
                }
            )
        return {
            "resultados": resultados[:8],
            "aviso": "Alergênicos declarados pelo restaurante. Não há garantia contra contaminação cruzada.",
        }

    def buscar_politicas(self, pergunta: str) -> dict:
        trechos = self.retriever.buscar(pergunta, k=3)
        return {"trechos": [{"fonte": t.fonte, "texto": t.texto} for t in trechos]}

    def solicitar_reembolso(self, pedido_id: str, motivo: str, prato_ids: list[str] | None = None) -> dict:
        pedido = self._pedido_do_cliente(pedido_id)
        if pedido is None:
            return NAO_ENCONTRADO
        if motivo not in MOTIVOS_REEMBOLSO:
            return {"erro": f"Motivo inválido. Use um destes: {', '.join(MOTIVOS_REEMBOLSO)}."}
        if pedido["status"] == "cancelado":
            return {"aprovado": False, "motivo": "Pedido cancelado: o reembolso integral já foi feito automaticamente."}

        prazo = PRAZO_ESTORNO[pedido["pagamento"]]

        if motivo == "atraso":
            if (motivo_chave := (pedido["id"], "atraso")) in self.reembolsos_concedidos:
                return {"aprovado": False, "motivo": "A compensação por atraso deste pedido já foi concedida."}
            atraso = self._atraso_minutos(pedido)
            if atraso <= 30:
                return {"aprovado": False, "atraso_minutos": atraso,
                        "motivo": "Atrasos de até 30 minutos não geram compensação."}
            self.reembolsos_concedidos.add(motivo_chave)
            resposta = {"aprovado": True, "atraso_minutos": atraso, "cupom": 10.00,
                        "detalhe": "Cupom de R$ 10,00 creditado na hora, válido por 30 dias."}
            if atraso > 60:
                taxa = self.base.restaurantes[pedido["restaurante_id"]]["taxa_entrega"]
                resposta.update(reembolso_taxa_entrega=taxa, prazo_estorno=prazo)
            return resposta

        if motivo == "nao_recebido":
            if pedido["status"] != "entregue":
                return {"aprovado": False, "motivo": f'O pedido ainda está com status "{pedido["status"]}".'}
            return self._protocolo("pedido marcado como entregue mas não recebido", pedido["id"])

        # item_faltando, item_errado e qualidade exigem pedido entregue há menos de 24 h
        if pedido["status"] != "entregue":
            return {"aprovado": False, "motivo": "Esse tipo de reclamação só vale para pedidos já entregues."}
        if self.base.agora - pedido["entregue_em"] > JANELA_RECLAMACAO:
            return {"aprovado": False,
                    "motivo": "A reclamação foi feita mais de 24 horas depois da entrega. "
                              "O cliente pode pedir análise excepcional a um atendente humano."}

        if motivo == "qualidade":
            return self._protocolo("problema de qualidade da comida", pedido["id"])

        # item_faltando / item_errado: reembolso do valor dos itens informados
        itens_do_pedido = {i["prato_id"]: i["qtd"] for i in pedido["itens"]}
        # Os erros trazem os itens COM NOME: só com os códigos, o modelo não percebia que
        # o "refrigerante" reclamado pelo cliente nunca fez parte do pedido.
        aviso = ("Se o item citado pelo cliente não está nesta lista, ele não fazia parte do pedido: "
                 "diga isso ao cliente em vez de pedir mais detalhes.")
        if not prato_ids:
            return {"erro": "Informe quais itens (prato_id) faltaram ou vieram errados.",
                    "itens_do_pedido": self._itens_com_nome(pedido), "aviso": aviso}
        invalidos = [p for p in prato_ids if p not in itens_do_pedido]
        if invalidos:
            return {"erro": f"Itens que não fazem parte do pedido: {', '.join(invalidos)}.",
                    "itens_do_pedido": self._itens_com_nome(pedido), "aviso": aviso}

        valor, ja_reembolsados = 0.0, []
        for prato_id in prato_ids:
            chave = (pedido["id"], "item", prato_id)
            if chave in self.reembolsos_concedidos:
                ja_reembolsados.append(prato_id)
                continue
            self.reembolsos_concedidos.add(chave)
            valor += self.base.pratos[prato_id]["preco"] * itens_do_pedido[prato_id]
        if valor == 0:
            return {"aprovado": False, "motivo": "Esses itens já foram reembolsados nesta conversa."}
        return {"aprovado": True, "valor_reembolso": round(valor, 2), "prazo_estorno": prazo,
                "itens_ja_reembolsados_antes": ja_reembolsados}

    def cancelar_pedido(self, pedido_id: str, confirmado_pelo_cliente: bool = False) -> dict:
        pedido = self._pedido_do_cliente(pedido_id)
        if pedido is None:
            return NAO_ENCONTRADO
        status = pedido["status"]
        if status == "cancelado":
            return {"cancelado": False, "motivo": "Este pedido já está cancelado."}
        if status in ("a_caminho", "entregue"):
            return {"cancelado": False, "motivo": f'Pedidos com status "{status}" não podem ser cancelados.'}
        if status == "preparando":
            return {"cancelado": False, **self._protocolo("cancelamento de pedido em preparo", pedido["id"]),
                    "motivo": "O pedido já está em preparo; o cancelamento depende do restaurante."}
        # status == "confirmado"
        if not confirmado_pelo_cliente:
            return {"cancelado": False,
                    "motivo": "Peça ao cliente uma confirmação explícita antes de cancelar."}
        pedido["status"] = "cancelado"
        return {"cancelado": True, "reembolso_integral": self.base.total_pedido(pedido),
                "prazo_estorno": PRAZO_ESTORNO[pedido["pagamento"]]}

    def transferir_para_humano(self, motivo: str, pedido_id: str | None = None) -> dict:
        return self._protocolo(motivo, pedido_id)

    # ------------------------------------------------------------------ registro
    def executar(self, nome: str, argumentos: dict) -> dict:
        funcao = getattr(self, nome, None)
        if nome not in NOMES_FERRAMENTAS or funcao is None:
            return {"erro": f"Ferramenta desconhecida: {nome}"}
        try:
            return funcao(**argumentos)
        except TypeError as e:  # argumentos errados vindos do modelo
            return {"erro": f"Argumentos inválidos para {nome}: {e}"}


def _funcao(nome, descricao, propriedades, obrigatorios=()):
    # Campos opcionais também aceitam null: muitos modelos mandam {"pedido_id": null}
    # em vez de omitir o campo, e provedores como o Groq validam o esquema e recusam
    # a chamada inteira se o tipo não bater.
    for campo, esquema in propriedades.items():
        if campo not in obrigatorios:
            esquema["type"] = [esquema["type"], "null"]
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {"type": "object", "properties": propriedades, "required": list(obrigatorios)},
        },
    }


ESQUEMA_FERRAMENTAS = [
    _funcao(
        "consultar_pedido",
        "Consulta um pedido do cliente (status, itens, valores, previsão e atraso). "
        "Sem pedido_id, lista os pedidos recentes do cliente.",
        {"pedido_id": {"type": "string", "description": "Número do pedido, ex.: 1001"}},
    ),
    _funcao(
        "buscar_cardapio",
        "Busca pratos no cardápio, com filtros de alergênicos, veganos e preço.",
        {
            "consulta": {"type": "string", "description": "Texto livre, ex.: 'hambúrguer', 'sushi'"},
            "restaurante": {"type": "string"},
            "sem_alergenicos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alergênicos a evitar: gluten, lactose, ovo, peixe, frutos_do_mar, "
                               "amendoim, soja, gergelim, oleaginosas",
            },
            "apenas_veganos": {"type": "boolean"},
            "preco_max": {"type": "number"},
        },
    ),
    _funcao(
        "buscar_politicas",
        "Busca trechos das políticas da empresa (reembolso, atrasos, cancelamento, estornos, "
        "alergênicos, atendimento). Use antes de explicar qualquer regra.",
        {"pergunta": {"type": "string"}},
        ["pergunta"],
    ),
    _funcao(
        "solicitar_reembolso",
        "Registra uma reclamação. O sistema decide se aprova seguindo a política. "
        "Para item_faltando ou item_errado, informe os prato_ids afetados.",
        {
            "pedido_id": {"type": "string"},
            "motivo": {"type": "string", "enum": MOTIVOS_REEMBOLSO},
            "prato_ids": {"type": "array", "items": {"type": "string"}},
        },
        ["pedido_id", "motivo"],
    ),
    _funcao(
        "cancelar_pedido",
        "Cancela um pedido. Só use confirmado_pelo_cliente=true depois que o cliente disser "
        "explicitamente que quer cancelar.",
        {"pedido_id": {"type": "string"}, "confirmado_pelo_cliente": {"type": "boolean"}},
        ["pedido_id"],
    ),
    _funcao(
        "transferir_para_humano",
        "Encaminha o atendimento para um atendente humano.",
        {"motivo": {"type": "string"}, "pedido_id": {"type": "string"}},
        ["motivo"],
    ),
]
NOMES_FERRAMENTAS = {f["function"]["name"] for f in ESQUEMA_FERRAMENTAS}