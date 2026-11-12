"""Converse com o agente pelo terminal.

Uso:  python cli.py           (digite "sair" para encerrar)
      python cli.py --debug   (mostra as ferramentas chamadas)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agente_delivery import criar_agente  # noqa: E402


def main():
    debug = "--debug" in sys.argv
    agente = criar_agente()
    print("Assistente Rango — você está logada como Ana (cliente c001). Digite 'sair' para encerrar.\n")
    while True:
        mensagem = input("Você: ").strip()
        if mensagem.lower() in {"sair", "exit", "quit"}:
            break
        if not mensagem:
            continue
        resposta = agente.responder(mensagem)
        if debug:
            for passo in resposta.passos:
                print(f"  🔧 {passo['ferramenta']}({json.dumps(passo['argumentos'], ensure_ascii=False)})")
                print(f"     → {json.dumps(passo['resultado'], ensure_ascii=False, default=str)[:300]}")
            print(f"  ⏱ {resposta.latencia_s}s · {resposta.tokens} tokens")
        print(f"Rango: {resposta.texto}\n")


if __name__ == "__main__":
    main()
