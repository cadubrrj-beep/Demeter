#!/usr/bin/env python3
"""
ranking_b3.py - Ranking de ações da B3 pela Fórmula Mágica (Greenblatt).

Fluxo:
  1. Baixa a tabela de indicadores do Fundamentus (ou lê um HTML salvo).
  2. Filtra ações sem liquidez, com indicadores inválidos ou de setores excluídos.
  3. Ranqueia por Earnings Yield (EBIT/EV) e por ROIC; soma as duas posições.
  4. Quanto MENOR a soma, melhor a ação (barata e de boa qualidade).
  5. Salva ranking.csv e ranking.json.

Uso:
  python ranking_b3.py                  # baixa do Fundamentus
  python ranking_b3.py --html pag.html  # usa um HTML salvo (para testes)
  python ranking_b3.py --top 30         # mostra 30 no terminal

Aviso: material informativo, não é recomendação de investimento.
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- configuração
URL = "https://www.fundamentus.com.br/resultado.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (projeto pessoal ranking-b3)"}

# Volume médio diário (R$, últimos 2 meses) mínimo. Evita ações ilíquidas.
MIN_LIQUIDEZ = 1_000_000

# Greenblatt recomenda excluir financeiras (bancos, seguradoras): EV/EBIT não
# faz sentido para elas. Lista inicial (4 primeiras letras do ticker) - ajuste.
EXCLUIR_BASES = {
    "ITUB", "BBDC", "BBAS", "SANB", "BPAC", "BRSR", "BPAN", "BMGB", "ABCB",
    "BAZA", "BNBR", "BEES", "BIDI", "B3SA", "BBSE", "CXSE", "IRBR", "PSSA",
    "SULA", "WIZC", "BRBI", "BRAP",
}

TOP_PADRAO = 20


# ---------------------------------------------------------------------- coleta
def baixar_html() -> str:
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=(10, 15))
    except requests.exceptions.ConnectTimeout:
        sys.exit(
            "Não consegui nem conectar ao Fundamentus (ConnectTimeout).\n"
            "Provável bloqueio de rede/firewall. Teste no terminal:\n"
            "  curl -I https://www.fundamentus.com.br/resultado.php"
        )
    except requests.exceptions.ReadTimeout:
        sys.exit(
            "Conectou, mas o servidor não respondeu a tempo (ReadTimeout).\n"
            "Pode ser bloqueio anti-bot do site. Alternativa: abra a página\n"
            "no navegador, salve como HTML (Ctrl+S) e rode:\n"
            "  python ranking_b3.py --html caminho/da/pagina.html"
        )
    except requests.exceptions.RequestException as e:
        sys.exit(f"Falha na conexão: {e}")

    resp.raise_for_status()
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("latin-1")


def ler_html_arquivo(caminho: str) -> str:
    bruto = Path(caminho).read_bytes()
    try:
        return bruto.decode("utf-8")
    except UnicodeDecodeError:
        return bruto.decode("latin-1")


# --------------------------------------------------------------------- parsing
def normalizar(nome) -> str:
    """'EV/EBIT' -> 'evebit', 'Liq.2meses' -> 'liq2meses', 'Cotação' -> 'cotacao'."""
    s = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def para_numero(valor) -> float:
    """Converte '1.234,56' e '12,34%' (formato BR) em float. Inválido vira NaN."""
    if pd.isna(valor):
        return float("nan")
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip().replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def carregar_dados(html: str) -> pd.DataFrame:
    tabelas = pd.read_html(StringIO(html), decimal=",", thousands=".")
    if not tabelas:
        sys.exit("Nenhuma tabela encontrada no HTML.")
    df = tabelas[0].copy()
    df.columns = [normalizar(c) for c in df.columns]

    obrigatorias = {"papel", "evebit", "roic", "liq2meses"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        sys.exit(
            f"Colunas esperadas não encontradas: {sorted(faltando)}.\n"
            f"Colunas recebidas: {list(df.columns)}\n"
            "O Fundamentus pode ter mudado o layout - ajuste os nomes no script."
        )

    for col in df.columns:
        if col != "papel":
            df[col] = df[col].map(para_numero)
    df["papel"] = df["papel"].astype(str).str.strip()
    return df


# ------------------------------------------------------------------------ score
def calcular_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base"] = df["papel"].str[:4]

    df = df[~df["base"].isin(EXCLUIR_BASES)]
    df = df[df["liq2meses"] >= MIN_LIQUIDEZ]
    df = df[(df["evebit"] > 0) & (df["roic"] > 0)]

    # P/VP negativo = patrimônio líquido negativo (empresa endividada além
    # do que possui). Earnings Yield e ROIC ficam distorcidos nesses casos
    # (ex.: BHIA3), então é mais seguro excluir.
    if "pvp" in df.columns:
        df = df[df["pvp"] > 0]

    # Uma classe por empresa (PETR3 e PETR4): fica a mais líquida.
    df = df.sort_values("liq2meses", ascending=False).drop_duplicates("base")

    # Earnings Yield em %: quanto maior, mais barata a empresa.
    df["earnings_yield"] = 100 / df["evebit"]

    df["rank_ey"] = df["earnings_yield"].rank(ascending=False, method="min")
    df["rank_roic"] = df["roic"].rank(ascending=False, method="min")
    df["score"] = df["rank_ey"] + df["rank_roic"]

    df = df.sort_values(["score", "liq2meses"], ascending=[True, False])
    df = df.reset_index(drop=True)
    df.insert(0, "posicao", df.index + 1)
    return df


# ------------------------------------------------------------------------ saída
def montar_saida(df: pd.DataFrame) -> pd.DataFrame:
    cotacao = next((c for c in df.columns if c.startswith("cota")), None)

    # Apenas informativo - não entra no score. Valor em R$ pago em
    # dividendos por ação nos últimos 12 meses (mesma janela do Div.Yield).
    if cotacao and "divyield" in df.columns:
        df["div_por_acao"] = (df[cotacao] * df["divyield"] / 100).round(2)

    desejadas = [
        "posicao", "papel", cotacao, "earnings_yield", "roic",
        "pl", "pvp", "divyield", "div_por_acao",
        "liq2meses", "rank_ey", "rank_roic", "score",
    ]
    colunas = [c for c in desejadas if c and c in df.columns]
    saida = df[colunas].copy()
    if cotacao:
        saida = saida.rename(columns={cotacao: "cotacao"})
    return saida.round(2)


def salvar(saida: pd.DataFrame, pasta: Path) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    saida.to_csv(pasta / "ranking.csv", index=False)

    payload = {
        "atualizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterio": "Formula Magica (Greenblatt): EBIT/EV + ROIC. Menor score = melhor.",
        "total": len(saida),
        "acoes": json.loads(saida.to_json(orient="records")),
    }
    (pasta / "ranking.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Ranking B3 - Fórmula Mágica")
    ap.add_argument("--html", help="usar um HTML salvo em vez de baixar")
    ap.add_argument("--saida", default=".", help="pasta de saída (padrão: atual)")
    ap.add_argument("--top", type=int, default=TOP_PADRAO, help="quantas mostrar")
    args = ap.parse_args()

    html = ler_html_arquivo(args.html) if args.html else baixar_html()
    bruto = carregar_dados(html)
    ranking = calcular_ranking(bruto)
    if ranking.empty:
        sys.exit("Nenhuma ação passou pelos filtros. Revise MIN_LIQUIDEZ.")

    saida = montar_saida(ranking)
    salvar(saida, Path(args.saida))

    print(f"\n{len(bruto)} papéis lidos -> {len(saida)} após filtros.\n")
    print(saida.head(args.top).to_string(index=False))
    print(f"\nArquivos salvos em: {Path(args.saida).resolve()} (ranking.csv, ranking.json)")
    print("Material informativo, não é recomendação de investimento.\n")


if __name__ == "__main__":
    main()