"""Interface web (Streamlit).

Rodar localmente:  streamlit run app.py
Publicar: Streamlit Community Cloud, com LLM_API_KEY nos "Secrets" do app.
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

# no Streamlit Cloud, a chave vem de st.secrets; copia para o ambiente antes de importar a config
try:
    for chave in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if chave in st.secrets and not os.getenv(chave):
            os.environ[chave] = st.secrets[chave]
except Exception:  # noqa: BLE001 - sem secrets.toml: usa o .env
    pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agente_delivery import criar_agente  # noqa: E402
from agente_delivery.config import LLM_MODEL  # noqa: E402
from agente_delivery.dados import carregar_base  # noqa: E402

LIMITE_MENSAGENS = 20  # protege a cota gratuita da API numa demo pública

EXEMPLOS = [
    "Cadê meu pedido 1001?",
    "Meu pedido 1006 está muito atrasado, quero compensação",
    "No pedido 1002 não veio o tiramisù",
    "Quero cancelar o pedido 1004",
    "Tenho alergia a amendoim, o que posso pedir no Verde Vivo?",
    "Me mostra o pedido 1008",
]

st.set_page_config(page_title="Agente de atendimento — Rango", page_icon="🛵")


@st.cache_resource
def _retriever():
    from agente_delivery.retriever import criar_retriever

    return criar_retriever()


def nova_conversa():
    st.session_state.agente = criar_agente(retriever=_retriever())
    st.session_state.mensagens = []


if "agente" not in st.session_state:
    nova_conversa()

with st.sidebar:
    st.header("🛵 Rango (app fictício)")
    st.caption(f"Você está logada como **Ana** (c001). Modelo: `{LLM_MODEL}`")
    st.button("🔄 Nova conversa", on_click=nova_conversa, use_container_width=True)

    base = carregar_base()
    st.subheader("Pedidos da Ana")
    for p in sorted(base.pedidos.values(), key=lambda p: p["criado_em"], reverse=True):
        if p["cliente_id"] == "c001":
            st.markdown(f'`{p["id"]}` {base.restaurantes[p["restaurante_id"]]["nome"]} — *{p["status"]}*')
    st.caption(f'Horário simulado: {base.agora:%d/%m %H:%M}. O pedido 1008 é de outro cliente.')

    st.subheader("Experimente")
    for exemplo in EXEMPLOS:
        if st.button(exemplo, use_container_width=True):
            st.session_state.pendente = exemplo

st.title("Agente de atendimento para delivery")
st.caption("Tool calling + RAG sobre políticas + regras de negócio no código. Clique em "
           "**Por dentro do agente** para ver cada ferramenta chamada.")

for m in st.session_state.mensagens:
    with st.chat_message(m["papel"]):
        st.markdown(m["texto"])
        if m.get("passos") is not None:
            with st.expander(f"🔍 Por dentro do agente — {len(m['passos'])} ferramenta(s), "
                             f"{m['latencia']}s, {m['tokens']} tokens"):
                for passo in m["passos"]:
                    st.markdown(f"**{passo['ferramenta']}** `{json.dumps(passo['argumentos'], ensure_ascii=False)}`")
                    st.json(passo["resultado"], expanded=False)

entrada = st.chat_input("Escreva sua mensagem") or st.session_state.pop("pendente", None)
if entrada:
    usadas = sum(1 for m in st.session_state.mensagens if m["papel"] == "user")
    if usadas >= LIMITE_MENSAGENS:
        st.warning("Limite de mensagens desta demo atingido. Clique em Nova conversa.")
    else:
        st.session_state.mensagens.append({"papel": "user", "texto": entrada})
        with st.spinner("Pensando..."):
            try:
                r = st.session_state.agente.responder(entrada)
                st.session_state.mensagens.append(
                    {"papel": "assistant", "texto": r.texto, "passos": r.passos,
                     "latencia": r.latencia_s, "tokens": r.tokens})
            except Exception as erro:  # noqa: BLE001
                st.session_state.mensagens.append(
                    {"papel": "assistant", "texto": f"⚠️ Erro ao chamar o modelo: {erro}"})
        st.rerun()
