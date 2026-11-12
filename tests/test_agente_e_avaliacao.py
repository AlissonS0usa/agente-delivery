"""Testes do loop do agente (com LLM falso) e das verificações da avaliação."""

import json

from agente_delivery.agente import Agente
from agente_delivery.config import MAX_PASSOS
from avaliar import verificar


def test_agente_executa_ferramenta_e_responde(ferramentas, llm_falso):
    llm = llm_falso([[("consultar_pedido", '{"pedido_id": "1001"}')], "Seu pedido está a caminho!"])
    agente = Agente(llm, ferramentas)
    resposta = agente.responder("cadê o 1001?")

    assert resposta.texto == "Seu pedido está a caminho!"
    assert resposta.passos[0]["ferramenta"] == "consultar_pedido"
    assert resposta.passos[0]["resultado"]["status"] == "a_caminho"
    # histórico: sistema, usuário, pedido de ferramenta, resultado da ferramenta, resposta final
    papeis = [m["role"] for m in agente.historico]
    assert papeis == ["system", "user", "assistant", "tool", "assistant"]
    assert json.loads(agente.historico[3]["content"])["pedido_id"] == "1001"
    assert len(llm.chamadas) == 2


def test_json_invalido_vira_erro_e_nao_quebra(ferramentas, llm_falso):
    llm = llm_falso([[("consultar_pedido", "{nao é json")], "ok"])
    resposta = Agente(llm, ferramentas).responder("oi")
    assert "erro" in resposta.passos[0]["resultado"]


def test_loop_infinito_encaminha_para_humano(ferramentas, llm_falso):
    llm = llm_falso([[("consultar_pedido", "{}")]] * (MAX_PASSOS + 2))
    resposta = Agente(llm, ferramentas).responder("oi")
    assert "protocolo" in resposta.texto
    assert resposta.passos[-1]["ferramenta"] == "transferir_para_humano"


def test_prompt_traz_cliente_e_horario(ferramentas, llm_falso):
    agente = Agente(llm_falso(["oi"]), ferramentas)
    sistema = agente.historico[0]["content"]
    assert "Ana" in sistema and "10/11/2026 20:00" in sistema


def test_verificar_detecta_cancelamento_proibido():
    caso = {"resultados_proibidos": [{"ferramenta": "cancelar_pedido", "campo": "cancelado", "valor": True}],
            "deve_conter_algum": ["confirm"]}
    passos = [{"ferramenta": "cancelar_pedido", "argumentos": {}, "resultado": {"cancelado": True}}]
    checks = verificar(caso, passos, "Pedido cancelado!")
    assert checks == {"proibe cancelar_pedido.cancelado=True": False, "texto contém": False}


def test_verificar_normaliza_acentos_e_escalonamento():
    caso = {"deve_conter_algum": ["2 dias uteis"], "deve_escalar": True, "ferramentas_esperadas": ["buscar_politicas"]}
    passos = [
        {"ferramenta": "buscar_politicas", "argumentos": {}, "resultado": {"trechos": []}},
        {"ferramenta": "transferir_para_humano", "argumentos": {}, "resultado": {"encaminhado_para_humano": True}},
    ]
    assert all(verificar(caso, passos, "Cai em até 2 DIAS ÚTEIS. Protocolo ATD-2001.").values())


def test_retenta_quando_provedor_recusa_argumentos_da_ferramenta(monkeypatch):
    from agente_delivery import agente as modulo

    monkeypatch.setattr(modulo.time, "sleep", lambda s: None)

    class ErroGroq(Exception):
        status_code = 400

    tentativas = []

    def chamada():
        tentativas.append(1)
        if len(tentativas) < 3:
            raise ErroGroq("Tool call validation failed: parameters did not match schema")
        return "ok"

    assert modulo._chamar_com_retentativa(chamada) == "ok"
    assert len(tentativas) == 3


def test_outros_erros_400_nao_sao_repetidos():
    import pytest

    from agente_delivery import agente as modulo

    class ErroChave(Exception):
        status_code = 400

    with pytest.raises(ErroChave):
        modulo._chamar_com_retentativa(lambda: (_ for _ in ()).throw(ErroChave("invalid model")))


def test_detecta_quando_fala_que_encaminhou_mas_nao_encaminhou():
    checks = verificar({}, [], "Desculpe, vou transferir sua solicitação para um atendente.")
    assert checks == {"fala bate com ação: encaminhamento": False}


def test_detecta_beneficio_prometido_sem_aprovacao():
    passos = [{"ferramenta": "consultar_pedido", "argumentos": {}, "resultado": {"status": "a_caminho"}}]
    checks = verificar({}, passos, "O cupom será creditado na sua conta em breve.")
    assert checks["fala bate com ação: benefício"] is False


def test_exige_protocolo_quando_encaminha():
    passos = [{"ferramenta": "transferir_para_humano", "argumentos": {},
               "resultado": {"encaminhado_para_humano": True, "protocolo": "ATD-2001"}}]
    assert verificar({}, passos, "Encaminhei para um atendente.")["informou protocolo"] is False
    # o modelo às vezes usa hífen não separável (U+2011) no protocolo
    assert all(verificar({}, passos, "Encaminhei para um atendente. Protocolo ATD‑2001.").values())


def test_negacao_nao_conta_como_promessa():
    assert verificar({}, [], "O reembolso automático não foi aprovado e não vou transferir agora.") == {}


def test_tempo_de_espera_do_limite_e_medido(monkeypatch):
    from agente_delivery import agente as modulo

    monkeypatch.setattr(modulo.time, "sleep", lambda s: None)

    class Limite(Exception):
        status_code = 429

    chamadas, esperas = [], []

    def chamada():
        chamadas.append(1)
        if len(chamadas) < 3:
            raise Limite("rate limit")
        return "ok"

    modulo._chamar_com_retentativa(chamada, esperas=esperas)
    assert esperas == [5, 10]