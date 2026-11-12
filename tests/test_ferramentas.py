"""Testes das regras de negócio. Rodam sem LLM e sem internet."""

from agente_delivery.ferramentas import Ferramentas


def test_total_do_pedido_inclui_taxa(ferramentas):
    assert ferramentas.consultar_pedido("1002")["total"] == 71.89  # 42,90 + 22,00 + 6,99


def test_lista_so_pedidos_do_cliente(ferramentas):
    ids = {p["pedido_id"] for p in ferramentas.consultar_pedido()["pedidos"]}
    assert "1008" not in ids  # 1008 é do cliente c002
    assert {"1001", "1002", "1004"} <= ids


def test_pedido_de_outro_cliente_parece_inexistente(ferramentas):
    assert ferramentas.consultar_pedido("1008") == ferramentas.consultar_pedido("9999")
    assert "erro" in ferramentas.solicitar_reembolso("1008", "item_faltando", ["p201"])
    assert "erro" in ferramentas.cancelar_pedido("1008", True)


def test_atraso_menor_que_30_min_nao_compensa(ferramentas):
    r = ferramentas.solicitar_reembolso("1001", "atraso")  # 10 min de atraso
    assert r["aprovado"] is False and r["atraso_minutos"] == 10


def test_atraso_maior_que_30_min_da_cupom_uma_vez(ferramentas):
    primeiro = ferramentas.solicitar_reembolso("1006", "atraso")  # 50 min
    segundo = ferramentas.solicitar_reembolso("1006", "atraso")
    assert primeiro["aprovado"] and primeiro["cupom"] == 10.0
    assert "reembolso_taxa_entrega" not in primeiro
    assert segundo["aprovado"] is False


def test_item_faltando_reembolsa_valor_do_item_uma_vez(ferramentas):
    r = ferramentas.solicitar_reembolso("1002", "item_faltando", ["p104"])
    assert r["aprovado"] and r["valor_reembolso"] == 22.0
    assert ferramentas.solicitar_reembolso("1002", "item_faltando", ["p104"])["aprovado"] is False


def test_item_que_nao_esta_no_pedido_e_recusado(ferramentas):
    assert "erro" in ferramentas.solicitar_reembolso("1002", "item_faltando", ["p999"])


def test_reclamacao_depois_de_24h_e_negada(ferramentas):
    r = ferramentas.solicitar_reembolso("1003", "item_errado", ["p201"])
    assert r["aprovado"] is False and "24 horas" in r["motivo"]


def test_qualidade_e_nao_recebido_vao_para_humano(ferramentas):
    assert ferramentas.solicitar_reembolso("1009", "qualidade")["encaminhado_para_humano"]
    assert ferramentas.solicitar_reembolso("1002", "nao_recebido")["encaminhado_para_humano"]


def test_cancelamento_exige_confirmacao(ferramentas):
    assert ferramentas.cancelar_pedido("1004")["cancelado"] is False
    r = ferramentas.cancelar_pedido("1004", confirmado_pelo_cliente=True)
    assert r["cancelado"] and r["reembolso_integral"] == 60.89
    assert ferramentas.consultar_pedido("1004")["status"] == "cancelado"


def test_cancelamento_nao_vaza_entre_conversas(retriever):
    a, b = Ferramentas("c001", retriever), Ferramentas("c001", retriever)
    a.cancelar_pedido("1004", True)
    assert b.consultar_pedido("1004")["status"] == "confirmado"


def test_pedido_em_preparo_vai_para_humano(ferramentas):
    r = ferramentas.cancelar_pedido("1005", True)
    assert r["cancelado"] is False and r["encaminhado_para_humano"]


def test_pedido_a_caminho_nao_cancela(ferramentas):
    assert ferramentas.cancelar_pedido("1001", True)["cancelado"] is False


def test_filtro_de_alergenicos_aceita_acento(ferramentas):
    nomes = {p["nome"] for p in ferramentas.buscar_cardapio(restaurante="Burger Brasa", sem_alergenicos=["glúten"])["resultados"]}
    assert nomes == {"Batata frita grande", "Milkshake de chocolate"}


def test_ferramenta_desconhecida_ou_argumento_errado(ferramentas):
    assert "erro" in ferramentas.executar("apagar_banco", {})
    assert "erro" in ferramentas.executar("consultar_pedido", {"numero": "1001"})


def test_campos_opcionais_aceitam_null_no_esquema():
    """Regressão: o Groq recusava {"pedido_id": null} porque o esquema só aceitava string."""
    from agente_delivery.ferramentas import ESQUEMA_FERRAMENTAS

    for f in ESQUEMA_FERRAMENTAS:
        params = f["function"]["parameters"]
        for campo, esquema in params["properties"].items():
            if campo not in params["required"]:
                assert "null" in esquema["type"], f'{f["function"]["name"]}.{campo}'
            else:
                assert "null" not in esquema["type"], f'{f["function"]["name"]}.{campo}'


def test_ferramentas_funcionam_com_argumentos_null(ferramentas):
    assert "pedidos" in ferramentas.executar("consultar_pedido", {"pedido_id": None})
    r = ferramentas.executar("buscar_cardapio", {"consulta": "burger", "restaurante": None, "sem_alergenicos": None,
                                                 "apenas_veganos": None, "preco_max": None})
    assert r["resultados"]
    assert ferramentas.executar("cancelar_pedido", {"pedido_id": "1004", "confirmado_pelo_cliente": None})["cancelado"] is False


def test_consulta_ja_traz_direito_a_compensacao(ferramentas):
    """Regressão da rodada 2: o modelo lia a política e negava o cupom de um atraso de 50 min."""
    assert ferramentas.consultar_pedido("1006")["compensacao_por_atraso"]["tem_direito"] is True  # 50 min
    assert ferramentas.consultar_pedido("1001")["compensacao_por_atraso"]["tem_direito"] is False  # 10 min
    ferramentas.solicitar_reembolso("1006", "atraso")
    assert "já concedida" in ferramentas.consultar_pedido("1006")["compensacao_por_atraso"]["situacao"]


def test_compensacao_acima_de_60_min_inclui_taxa(ferramentas):
    pedido = ferramentas.pedidos["1006"]
    pedido["previsao_entrega"] = ferramentas.base.agora.replace(hour=18, minute=50)  # 70 min de atraso
    assert "taxa de entrega" in ferramentas.consultar_pedido("1006")["compensacao_por_atraso"]["direito"]


def test_erro_de_item_mostra_nomes_dos_itens(ferramentas):
    """Regressão da rodada 2: só com códigos, o modelo não via que não havia refrigerante no pedido."""
    r = ferramentas.solicitar_reembolso("1002", "item_faltando")
    assert {"prato_id": "p104", "nome": "Tiramisù"} in r["itens_do_pedido"]
    assert "não fazia parte do pedido" in r["aviso"]