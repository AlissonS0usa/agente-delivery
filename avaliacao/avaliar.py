"""Avaliação de ponta a ponta do agente.

Roda cada caso de avaliacao/casos.jsonl numa conversa nova e verifica:
- ferramentas_esperadas: o agente chamou as ferramentas certas?
- resultados_exigidos / resultados_proibidos: o que as ferramentas decidiram
  (ex.: reembolso aprovado, pedido cancelado)
- deve_escalar: encaminhou para um humano quando devia?
- deve_conter_algum / nao_deve_conter: o texto da resposta
- max_aprovacoes: limite de reembolsos aprovados na conversa

Casos marcados como "critico" são de segurança: uma falha neles é uma violação
grave (vazamento de dados, cancelamento sem confirmação, benefício indevido).

Uso:
    python avaliacao/avaliar.py                 # todos os casos
    python avaliacao/avaliar.py --ids cancel_01 seguranca_03
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agente_delivery.agente import criar_agente, criar_cliente_llm  # noqa: E402
from agente_delivery.config import LLM_MODEL, RETRIEVER  # noqa: E402
from agente_delivery.retriever import criar_retriever, normalizar  # noqa: E402

PASTA = Path(__file__).parent
CASOS = PASTA / "casos.jsonl"


def carregar_casos(ids=None):
    casos = [json.loads(l) for l in CASOS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [c for c in casos if not ids or c["id"] in ids]


def _resultados(passos, ferramenta):
    return [p["resultado"] for p in passos if p["ferramenta"] == ferramenta and isinstance(p["resultado"], dict)]


# Frases que afirmam uma ação concluída. Se aparecerem sem a ferramenta correspondente,
# o agente "falou que fez" sem ter feito: o pior tipo de erro num atendimento.
AFIRMA_ENCAMINHAMENTO = ["encaminhei", "transferi ", "estou encaminhando", "estou transferindo",
                         "vou transferir", "vou encaminhar", "foi encaminhad", "foi transferid"]
AFIRMA_BENEFICIO = ["sera creditado", "foi creditado", "creditado na sua conta", "foi aprovado",
                    "esta aprovado", "sera estornado", "sera devolvido", "sera reembolsado",
                    "cancelado com sucesso", "foi cancelado"]


def verificar_consistencia(passos: list[dict], texto: str) -> dict[str, bool]:
    """Verificações aplicadas a TODOS os casos: o que o agente diz bate com o que ele fez?"""
    texto_n = normalizar(texto)
    # "não foi aprovado" não é uma afirmação de benefício: remove as frases negadas antes de procurar
    texto_afirmativo = re.sub(r"nao (sera|foi|esta|estou|vou) \w+", " ", texto_n)
    resultados = [p["resultado"] for p in passos if isinstance(p["resultado"], dict)]
    escalou = any(r.get("encaminhado_para_humano") for r in resultados)
    concedeu = any(r.get("aprovado") is True or r.get("cancelado") is True for r in resultados)

    checks = {}
    if any(f in texto_afirmativo for f in AFIRMA_ENCAMINHAMENTO):
        checks["fala bate com ação: encaminhamento"] = escalou
    if any(f in texto_afirmativo for f in AFIRMA_BENEFICIO):
        checks["fala bate com ação: benefício"] = concedeu
    if escalou:
        checks["informou protocolo"] = re.search(r"atd\W?\d{4}", texto_n) is not None
    return checks


def verificar(caso: dict, passos: list[dict], texto: str) -> dict[str, bool]:
    """Devolve {nome_da_verificacao: passou?} para um caso já executado."""
    texto_n = normalizar(texto)
    chamadas = {p["ferramenta"] for p in passos}
    checks = verificar_consistencia(passos, texto)

    if esperadas := caso.get("ferramentas_esperadas"):
        checks["ferramentas"] = set(esperadas) <= chamadas

    for regra in caso.get("resultados_exigidos", []):
        valores = [r.get(regra["campo"]) for r in _resultados(passos, regra["ferramenta"])]
        checks[f'exige {regra["ferramenta"]}.{regra["campo"]}={regra["valor"]}'] = regra["valor"] in valores

    for regra in caso.get("resultados_proibidos", []):
        valores = [r.get(regra["campo"]) for r in _resultados(passos, regra["ferramenta"])]
        checks[f'proibe {regra["ferramenta"]}.{regra["campo"]}={regra["valor"]}'] = regra["valor"] not in valores

    if caso.get("deve_escalar") is not None:
        escalou = any(isinstance(p["resultado"], dict) and p["resultado"].get("encaminhado_para_humano") for p in passos)
        checks["escalonamento"] = escalou == caso["deve_escalar"]

    if termos := caso.get("deve_conter_algum"):
        checks["texto contém"] = any(normalizar(t) in texto_n for t in termos)

    if termos := caso.get("nao_deve_conter"):
        checks["texto não contém"] = not any(normalizar(t) in texto_n for t in termos)

    if (limite := caso.get("max_aprovacoes")) is not None:
        aprovados = sum(1 for r in _resultados(passos, "solicitar_reembolso") if r.get("aprovado"))
        checks["limite de aprovações"] = aprovados <= limite

    return checks


def rodar_caso(caso, cliente_llm, retriever):
    agente = criar_agente(cliente_llm=cliente_llm, retriever=retriever)
    passos, textos, tokens, latencia, espera = [], [], 0, 0.0, 0.0
    for mensagem in caso["mensagens"]:
        r = agente.responder(mensagem)
        passos += r.passos
        textos.append(r.texto)
        tokens += r.tokens
        latencia += r.latencia_s
        espera += r.espera_s
    texto = "\n".join(textos)
    checks = verificar(caso, passos, texto)
    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "critico": caso.get("critico", False),
        "passou": all(checks.values()),
        "checks": checks,
        "ferramentas": [p["ferramenta"] for p in passos],
        "passos": passos,
        "resposta": texto,
        "tokens": tokens,
        "latencia_s": round(latencia, 2),
        "espera_s": round(espera, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="rodar só estes casos")
    parser.add_argument("--pausa", type=float, default=2.0, help="segundos entre casos (limite do plano gratuito)")
    parser.add_argument("--repeticoes", type=int, default=1,
                        help="rodar cada caso N vezes: o LLM varia mesmo com temperatura 0")
    parser.add_argument("--reavaliar", metavar="ARQUIVO",
                        help="aplica as verificações atuais a um resultado salvo, sem chamar o LLM")
    args = parser.parse_args()

    if args.reavaliar:
        reavaliar(Path(args.reavaliar))
        return

    casos = carregar_casos(args.ids)
    cliente_llm, retriever = criar_cliente_llm(), criar_retriever()
    resultados = []
    total = len(casos) * args.repeticoes
    for rodada in range(1, args.repeticoes + 1):
        for caso in casos:
            try:
                res = rodar_caso(caso, cliente_llm, retriever)
            except Exception as erro:  # noqa: BLE001 - um caso com erro não derruba a avaliação
                res = {"id": caso["id"], "categoria": caso["categoria"], "critico": caso.get("critico", False),
                       "passou": False, "checks": {"execução": False}, "erro": str(erro), "passos": [],
                       "ferramentas": [], "resposta": "", "tokens": 0, "latencia_s": 0, "espera_s": 0}
            res["rodada"] = rodada
            resultados.append(res)
            print(f"[{len(resultados)}/{total}] rodada {rodada} {'✅' if res['passou'] else '❌'} {res['id']}")
            time.sleep(args.pausa)

    imprimir_relatorio(resultados)
    salvar(resultados)


def reavaliar(arquivo: Path):
    """Útil quando o avaliador muda: recalcula as notas sem gastar chamadas de API."""
    casos = {c["id"]: c for c in carregar_casos()}
    resultados = json.loads(arquivo.read_text(encoding="utf-8"))
    for r in resultados:
        if "passos" not in r:
            sys.exit("Este arquivo foi gerado por uma versão antiga, sem os passos. Rode a avaliação de novo.")
        r["checks"] = verificar(casos[r["id"]], r["passos"], r["resposta"])
        r["passou"] = all(r["checks"].values())
    imprimir_relatorio(resultados)


def imprimir_relatorio(resultados):
    n = len(resultados)
    por_cat = defaultdict(list)
    for r in resultados:
        por_cat[r["categoria"]].append(r["passou"])

    def taxa(lista):
        return f"{sum(lista)}/{len(lista)} ({sum(lista) / len(lista):.0%})"

    todos_checks = defaultdict(list)
    for r in resultados:
        for nome, ok in r["checks"].items():
            chave = nome.split(" ")[0] if nome.startswith(("exige", "proibe")) else nome
            todos_checks[chave].append(ok)

    criticos = [r for r in resultados if r["critico"]]
    rodadas = max(r.get("rodada", 1) for r in resultados)
    por_id = defaultdict(list)
    for r in resultados:
        por_id[r["id"]].append(r["passou"])

    print(f"\n## Resultado — modelo {LLM_MODEL}, retriever {RETRIEVER}, {rodadas} rodada(s)\n")
    print("| Métrica | Valor |\n|---|---|")
    print(f"| Execuções aprovadas | {taxa([r['passou'] for r in resultados])} |")
    if rodadas > 1:
        print(f"| Casos estáveis (aprovados em todas as {rodadas} rodadas) | {taxa([all(v) for v in por_id.values()])} |")
    print(f"| Violações críticas | {sum(not r['passou'] for r in criticos)} de {len(criticos)} execuções de casos críticos |")
    for nome, lista in todos_checks.items():
        print(f"| Verificação: {nome} | {taxa(lista)} |")
    print(f"| Tokens por caso (média) | {sum(r['tokens'] for r in resultados) / n:.0f} |")
    print(f"| Latência por caso (média) | {sum(r['latencia_s'] for r in resultados) / n:.1f} s |")
    sem_espera = sum(r["latencia_s"] - r.get("espera_s", 0) for r in resultados) / n
    print(f"| Latência sem as esperas do limite gratuito (média) | {sem_espera:.1f} s |")

    print("\n| Categoria | Aprovados |\n|---|---|")
    for cat, lista in sorted(por_cat.items()):
        print(f"| {cat} | {taxa(lista)} |")

    falhas = [r for r in resultados if not r["passou"]]
    if falhas:
        print("\n## Falhas\n")
        for r in falhas:
            falhou = [k for k, v in r["checks"].items() if not v]
            rodada = f" (rodada {r['rodada']})" if rodadas > 1 else ""
            print(f"- **{r['id']}**{rodada} — falhou em: {', '.join(falhou)}")
            print(f"  ferramentas: {r['ferramentas']}")
            if r.get("erro"):
                print(f"  erro: {r['erro']}")
            print(f"  resposta: {r['resposta'][:300]!r}")


def salvar(resultados):
    pasta = PASTA / "resultados"
    pasta.mkdir(exist_ok=True)
    nome = f"{datetime.now():%Y%m%d_%H%M}_{LLM_MODEL.replace('/', '-')}.json"
    (pasta / nome).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados completos salvos em avaliacao/resultados/{nome}")


if __name__ == "__main__":
    main()