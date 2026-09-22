#!/usr/bin/env python3
"""
Analisa os CSVs gerados pelo benchmark_omp.py.

Tipos de análise:
  strong -> tempo, Speedup e Eficiência por número de threads
  chunks -> tempo, F_LB e relação F_LB x tempo por schedule/chunk
  weak   -> tempo por resolução, anotando as threads usadas

Exemplos:
  python3 scripts/analisar_resultados.py \
      --tipo strong \
      --input resultados/brutos/strong_scaling.csv \
      --output-dir resultados/analise/strong

  python3 scripts/analisar_resultados.py \
      --tipo chunks \
      --input resultados/brutos/chunks.csv \
      --output-dir resultados/analise/chunks

  python3 scripts/analisar_resultados.py \
      --tipo weak \
      --input resultados/brutos/weak_scaling.csv \
      --output-dir resultados/analise/weak
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED = {
    "caso", "resolucao", "max_iter", "threads_utilizadas",
    "schedule_utilizado", "chunk_utilizado", "repeticao",
    "tempo_serial_s", "tempo_calc_s", "speedup",
    "diferenca_percentual", "pixels_diferentes", "percentual_erro",
    "diferenca_maxima", "tempo_io_s", "tempo_total_s",
    "t_min_s", "t_max_s", "f_lb"
}

GROUP = [
    "caso", "resolucao", "max_iter", "threads_utilizadas",
    "schedule_utilizado", "chunk_utilizado"
]

SUMMARY = [
    *GROUP,
    "n_repeticoes", "tempo_serial_s", "tempo_calc_media_s",
    "tempo_calc_mediana_s", "tempo_calc_desvio_s", "tempo_calc_min_s",
    "tempo_calc_max_s", "tempo_io_media_s", "tempo_total_media_s",
    "speedup_media", "speedup_mediana", "eficiencia_percentual",
    "f_lb_media", "f_lb_mediana", "f_lb_desvio",
    "diferenca_percentual_media", "pixels_diferentes_max",
    "percentual_erro_max", "diferenca_maxima_max", "corretude_ok"
]


# ---------------------------------------------------------------------------
# Leitura e agregação
# ---------------------------------------------------------------------------

def ler_csv(paths: list[Path]) -> list[dict]:
    registros = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"CSV nao encontrado: {path}")

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"'{path}' nao possui cabecalho.")

            faltantes = REQUIRED - set(reader.fieldnames)
            if faltantes:
                raise ValueError(
                    f"'{path}' nao possui as colunas: "
                    f"{', '.join(sorted(faltantes))}"
                )

            for numero, row in enumerate(reader, start=2):
                try:
                    registros.append({
                        "caso": row["caso"],
                        "resolucao": int(row["resolucao"]),
                        "max_iter": int(row["max_iter"]),
                        "threads_utilizadas": int(row["threads_utilizadas"]),
                        "schedule_utilizado": row["schedule_utilizado"],
                        "chunk_utilizado": int(row["chunk_utilizado"]),
                        "repeticao": int(row["repeticao"]),
                        "tempo_serial_s": float(row["tempo_serial_s"]),
                        "tempo_calc_s": float(row["tempo_calc_s"]),
                        "speedup": float(row["speedup"]),
                        "diferenca_percentual": float(row["diferenca_percentual"]),
                        "pixels_diferentes": int(row["pixels_diferentes"]),
                        "percentual_erro": float(row["percentual_erro"]),
                        "diferenca_maxima": int(row["diferenca_maxima"]),
                        "tempo_io_s": float(row["tempo_io_s"]),
                        "tempo_total_s": float(row["tempo_total_s"]),
                        "t_min_s": float(row["t_min_s"]),
                        "t_max_s": float(row["t_max_s"]),
                        "f_lb": float(row["f_lb"]),
                    })
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Linha {numero} invalida em '{path}': {exc}"
                    ) from exc

    if not registros:
        raise ValueError("Nenhum resultado encontrado.")
    return registros


def media(valores):
    return statistics.fmean(valores)


def desvio(valores):
    return statistics.stdev(valores) if len(valores) > 1 else 0.0


def resumir(registros: list[dict]) -> list[dict]:
    grupos = defaultdict(list)
    for r in registros:
        grupos[tuple(r[c] for c in GROUP)].append(r)

    resumo = []
    for chave, rows in sorted(grupos.items()):
        caso, resolucao, max_iter, threads, schedule, chunk = chave
        tempos = [r["tempo_calc_s"] for r in rows]
        speedups = [r["speedup"] for r in rows]
        flbs = [r["f_lb"] for r in rows]

        erro_max = max(r["percentual_erro"] for r in rows)
        diff_max = max(r["diferenca_maxima"] for r in rows)
        speedup = media(speedups)

        resumo.append({
            "caso": caso,
            "resolucao": resolucao,
            "max_iter": max_iter,
            "threads_utilizadas": threads,
            "schedule_utilizado": schedule,
            "chunk_utilizado": chunk,
            "n_repeticoes": len(rows),
            "tempo_serial_s": media(r["tempo_serial_s"] for r in rows),
            "tempo_calc_media_s": media(tempos),
            "tempo_calc_mediana_s": statistics.median(tempos),
            "tempo_calc_desvio_s": desvio(tempos),
            "tempo_calc_min_s": min(tempos),
            "tempo_calc_max_s": max(tempos),
            "tempo_io_media_s": media(r["tempo_io_s"] for r in rows),
            "tempo_total_media_s": media(r["tempo_total_s"] for r in rows),
            "speedup_media": speedup,
            "speedup_mediana": statistics.median(speedups),
            "eficiencia_percentual": speedup / threads * 100.0,
            "f_lb_media": media(flbs),
            "f_lb_mediana": statistics.median(flbs),
            "f_lb_desvio": desvio(flbs),
            "diferenca_percentual_media": media(
                r["diferenca_percentual"] for r in rows
            ),
            "pixels_diferentes_max": max(
                r["pixels_diferentes"] for r in rows
            ),
            "percentual_erro_max": erro_max,
            "diferenca_maxima_max": diff_max,
            "corretude_ok": "SIM" if erro_max <= 0.01 and diff_max <= 1 else "NAO",
        })
    return resumo


def escrever_csv(path: Path, rows: list[dict], columns: list[str] = SUMMARY):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {}
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    out[col] = "" if math.isnan(value) else f"{value:.10f}"
                else:
                    out[col] = value
            writer.writerow(out)


# ---------------------------------------------------------------------------
# Títulos
# ---------------------------------------------------------------------------

def nome_caso(caso: str) -> str:
    return {
        "padrao": "Vista completa",
        "cavalos": "Vale dos cavalos-marinhos",
    }.get(caso, caso)


def titulo_base(rows: list[dict]) -> str:
    r = rows[0]
    return (
        f"{nome_caso(r['caso'])} — "
        f"{r['resolucao']}×{r['resolucao']} — "
        f"MAX_ITER={r['max_iter']}"
    )


def salvar_figura(path: Path):
    fig = plt.gcf()
    ax = fig.gca()

    # --- Ajusta largura conforme o título ---
    titulo = ax.get_title()
    if titulo:
        linhas = titulo.split("\n")
        maior_linha = max(len(l) for l in linhas)
        # ~0.12 polegadas por caractere + margem mínima
        largura_minima = max(6.4, maior_linha * 0.12)
        altura_atual = fig.get_size_inches()[1]
        fig.set_size_inches(largura_minima, altura_atual)

    # --- Ajusta altura conforme número de linhas do título ---
    if titulo and "\n" in titulo:
        n_linhas = titulo.count("\n") + 1
        altura_minima = 4.8 + (n_linhas - 1) * 0.4
        largura_atual = fig.get_size_inches()[0]
        fig.set_size_inches(largura_atual, altura_minima)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")  # <- chave!
    plt.close(fig)


# ---------------------------------------------------------------------------
# Gráficos de Strong Scaling
# ---------------------------------------------------------------------------

def grafico_tempo_strong(rows, path):
    rows = sorted(rows, key=lambda r: r["threads_utilizadas"])
    x = [r["threads_utilizadas"] for r in rows]
    y = [r["tempo_calc_media_s"] for r in rows]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Número de threads")
    plt.ylabel("Tempo médio de cálculo (s)")
    plt.title(f"Strong Scaling — Tempo\n{titulo_base(rows)}")
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


def grafico_speedup(rows, path):
    rows = sorted(rows, key=lambda r: r["threads_utilizadas"])
    x = [r["threads_utilizadas"] for r in rows]
    y = [r["speedup_media"] for r in rows]

    plt.figure()
    plt.plot(x, y, marker="o", label="Obtido")
    plt.plot(x, x, "--", label="Ideal: S(p)=p")
    plt.xlabel("Número de threads")
    plt.ylabel("Speedup")
    plt.title(f"Strong Scaling — Speedup\n{titulo_base(rows)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


def grafico_eficiencia(rows, path):
    rows = sorted(rows, key=lambda r: r["threads_utilizadas"])
    x = [r["threads_utilizadas"] for r in rows]
    y = [r["eficiencia_percentual"] for r in rows]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.axhline(100, linestyle="--", label="Ideal: 100%")
    plt.xlabel("Número de threads")
    plt.ylabel("Eficiência (%)")
    plt.title(f"Strong Scaling — Eficiência\n{titulo_base(rows)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


# ---------------------------------------------------------------------------
# Gráficos de chunks / balanceamento
# ---------------------------------------------------------------------------

def grafico_tempo_chunk(rows, path):
    plt.figure()
    for schedule in sorted({r["schedule_utilizado"] for r in rows}):
        dados = sorted(
            [r for r in rows if r["schedule_utilizado"] == schedule],
            key=lambda r: r["chunk_utilizado"]
        )
        x = [r["chunk_utilizado"] for r in dados]
        y = [r["tempo_calc_media_s"] for r in dados]
        plt.plot(x, y, marker="o", label=schedule)

    threads = rows[0]["threads_utilizadas"]
    plt.xlabel("Tamanho do chunk")
    plt.ylabel("Tempo médio de cálculo (s)")
    plt.title(
        f"Impacto do chunk no desempenho\n"
        f"{titulo_base(rows)} — {threads} threads"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


def grafico_flb_chunk(rows, path):
    plt.figure()
    for schedule in sorted({r["schedule_utilizado"] for r in rows}):
        dados = sorted(
            [r for r in rows if r["schedule_utilizado"] == schedule],
            key=lambda r: r["chunk_utilizado"]
        )
        x = [r["chunk_utilizado"] for r in dados]
        y = [r["f_lb_media"] for r in dados]
        plt.plot(x, y, marker="o", label=schedule)

    threads = rows[0]["threads_utilizadas"]
    plt.xlabel("Tamanho do chunk")
    plt.ylabel("Fator de balanceamento de carga (F_LB)")
    plt.title(
        f"Balanceamento de carga por chunk\n"
        f"{titulo_base(rows)} — {threads} threads"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


def grafico_flb_tempo(rows, path):
    plt.figure()
    for schedule in sorted({r["schedule_utilizado"] for r in rows}):
        dados = [r for r in rows if r["schedule_utilizado"] == schedule]
        plt.scatter(
            [r["f_lb_media"] for r in dados],
            [r["tempo_calc_media_s"] for r in dados],
            label=schedule
        )
        for r in dados:
            plt.annotate(
                str(r["chunk_utilizado"]),
                (r["f_lb_media"], r["tempo_calc_media_s"]),
                xytext=(5, 5),
                textcoords="offset points"
            )

    threads = rows[0]["threads_utilizadas"]
    plt.xlabel("Fator de balanceamento de carga (F_LB)")
    plt.ylabel("Tempo médio de cálculo (s)")
    plt.title(
        f"Balanceamento × desempenho\n"
        f"{titulo_base(rows)} — {threads} threads"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


# ---------------------------------------------------------------------------
# Weak Scaling
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Weak Scaling
# ---------------------------------------------------------------------------

def grafico_weak(rows, path):
    # Mapeia todas as threads disponíveis por resolução
    por_resolucao = defaultdict(list)
    for r in rows:
        por_resolucao[r["resolucao"]].append(r)

    # Mapeamento esperado do Weak Scaling (Resolução proporcional às threads)
    # Ex: 4096 -> 1 thread, 8192 -> 4 threads, 16384 -> 12 threads
    resolucoes_ordenadas = sorted(por_resolucao.keys())
    
    pontos = []
    for i, resolucao in enumerate(resolucoes_ordenadas):
        dados_res = por_resolucao[resolucao]
        
        # Pega a thread correspondente na mesma proporção do índice, 
        # ou se houver apenas uma linha para essa resolução, pega ela.
        threads_disponiveis = sorted(list({r["threads_utilizadas"] for r in dados_res}))
        
        if len(threads_disponiveis) == 1:
            thread_alvo = threads_disponiveis[0]
        else:
            # Se houver várias threads para a mesma resolução (ex: 1, 4, 12),
            # escolhe a thread correspondente ao nível do weak scaling (índice i)
            # Se o índice passar do limite, pega a maior thread disponível.
            idx_thread = min(i, len(threads_disponiveis) - 1)
            thread_alvo = threads_disponiveis[idx_thread]

        # Filtra a linha exata para essa resolução e essa thread
        linha_alvo = [
            r for r in dados_res 
            if r["threads_utilizadas"] == thread_alvo
        ]
        
        if linha_alvo:
            tempo = media(r["tempo_calc_media_s"] for r in linha_alvo)
            pontos.append((resolucao, thread_alvo, tempo))

    if len(pontos) < 2:
        raise ValueError("Sao necessarias pelo menos duas resolucoes validas no weak scaling.")

    x = [p[0] for p in pontos]
    y = [p[2] for p in pontos]

    plt.figure()
    plt.plot(x, y, marker="o")
    for resolucao, threads, tempo in pontos:
        plt.annotate(
            f"{threads} threads",
            (resolucao, tempo),
            xytext=(5, 5),
            textcoords="offset points"
        )

    caso = nome_caso(rows[0]["caso"])
    plt.xlabel("Resolução")
    plt.ylabel("Tempo médio de cálculo (s)")
    plt.title(f"Weak Scaling — {caso}")
    plt.grid(True, alpha=0.3)
    salvar_figura(path)


# ---------------------------------------------------------------------------
# Execução das análises
# ---------------------------------------------------------------------------

def analisar_strong(resumo, out):
    grupos = defaultdict(list)
    for r in resumo:
        grupos[(
            r["caso"], r["resolucao"],
            r["schedule_utilizado"], r["chunk_utilizado"]
        )].append(r)

    for i, rows in enumerate(grupos.values(), 1):
        sufixo = "" if len(grupos) == 1 else f"_{i}"
        grafico_tempo_strong(rows, out / f"01_tempo_strong{sufixo}.png")
        grafico_speedup(rows, out / f"02_speedup{sufixo}.png")
        grafico_eficiencia(rows, out / f"03_eficiencia{sufixo}.png")


def analisar_chunks(resumo, out):
    grupos = defaultdict(list)
    for r in resumo:
        grupos[(
            r["caso"], r["resolucao"], r["threads_utilizadas"]
        )].append(r)

    for i, rows in enumerate(grupos.values(), 1):
        sufixo = "" if len(grupos) == 1 else f"_{i}"
        grafico_tempo_chunk(rows, out / f"04_tempo_por_chunk{sufixo}.png")
        grafico_flb_chunk(rows, out / f"05_f_lb_por_chunk{sufixo}.png")
        grafico_flb_tempo(rows, out / f"06_f_lb_vs_tempo{sufixo}.png")


def analisar_weak(resumo, out):
    grafico_weak(resumo, out / "07_tempo_weak_scaling.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analisa resultados do benchmark OpenMP."
    )
    parser.add_argument(
        "--tipo", required=True,
        choices=["strong", "chunks", "weak"],
        help="Tipo de experimento a analisar."
    )
    parser.add_argument(
        "--input", nargs="+", type=Path, required=True,
        help="CSV(s) gerado(s) pelo benchmark_omp.py."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("resultados/analise"),
        help="Diretorio onde serao salvos CSVs e graficos."
    )
    args = parser.parse_args()

    try:
        registros = ler_csv(args.input)
        resumo = resumir(registros)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        graficos = args.output_dir / "graficos"
        graficos.mkdir(parents=True, exist_ok=True)

        escrever_csv(args.output_dir / "resultados_agregados.csv", resumo)
        escrever_csv(
            args.output_dir / "corretude.csv",
            resumo,
            [
                "caso", "resolucao", "max_iter", "threads_utilizadas",
                "schedule_utilizado", "chunk_utilizado", "n_repeticoes",
                "pixels_diferentes_max", "percentual_erro_max",
                "diferenca_maxima_max", "corretude_ok"
            ]
        )

        if args.tipo == "strong":
            analisar_strong(resumo, graficos)
        elif args.tipo == "chunks":
            analisar_chunks(resumo, graficos)
        else:
            analisar_weak(resumo, graficos)

    except (OSError, ValueError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print()
    print("=" * 100)
    print("ANALISE CONCLUIDA")
    print("=" * 100)

    for r in resumo:
        print(
            f'{r["caso"]:8s} | {r["resolucao"]:5d}² | '
            f'{r["threads_utilizadas"]:3d} t | '
            f'{r["schedule_utilizado"]:7s},{r["chunk_utilizado"]:<4d} | '
            f'T={r["tempo_calc_media_s"]:.6f}s | '
            f'S={r["speedup_media"]:.3f} | '
            f'E={r["eficiencia_percentual"]:.2f}% | '
            f'F_LB={r["f_lb_media"]:.6f} | '
            f'Corr={r["corretude_ok"]}'
        )

    print("=" * 100)
    print(f"CSV agregado: {args.output_dir / 'resultados_agregados.csv'}")
    print(f"Corretude:    {args.output_dir / 'corretude.csv'}")
    print(f"Graficos:     {graficos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
