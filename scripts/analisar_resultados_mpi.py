#!/usr/bin/env python3
"""Gera os nove gráficos comparativos dos benchmarks OpenMP e MPI.

Exemplo:
  python analisar_resultados_combinado.py \
    --omp-csv resultados/brutos/omp.csv \
    --mpi-csv resultados/brutos/mpi.csv \
    --output-dir resultados/analise_combinada

Os CSVs brutos são agregados por configuração antes dos gráficos. Para os
comparativos OpenMP/MPI, speedup e eficiência usam o tempo de cálculo, não o
I/O. No MPI, tempo paralelo total significa TEMPO_PARALELO_TOTAL_S, sem I/O.
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

OMP_REQUIRED = {
    "caso", "resolucao", "max_iter", "schedule_utilizado", "threads_utilizadas", 
    "chunk_utilizado", "repeticao", "tempo_serial_s", "tempo_calc_s", 
    "speedup", "f_lb",
}
MPI_REQUIRED = {
    "caso", "resolucao", "max_iter", "estrategia", "processos", 
    "chunk", "repeticao", "tempo_serial_mediana_s", "tempo_calculo_max_s", 
    "f_lb", "tempo_paralelo_total_s",
    "speedup_calculo",
}


def ler_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"'{path}' não possui cabeçalho.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"'{path}' não possui as colunas necessárias: {', '.join(sorted(missing))}")
        return list(reader)


def numero(row: dict, key: str, tipo=float):
    try:
        return tipo(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Valor inválido na coluna '{key}': {row.get(key)!r}") from exc


def normalizar(omp_rows: list[dict], mpi_rows: list[dict]) -> list[dict]:
    rows = []
    for raw in omp_rows:
        rows.append({
            "paradigma": "OpenMP", 
            "caso": raw["caso"],
            "resolucao": numero(raw, "resolucao", int), 
            "max_iter": numero(raw, "max_iter", int),
            "workers": numero(raw, "threads_utilizadas", int),
            "estrategia": raw["schedule_utilizado"].strip().lower(),
            "chunk": numero(raw, "chunk_utilizado", int),
            "repeticao": numero(raw, "repeticao", int),
            "tempo_serial": numero(raw, "tempo_serial_s"),
            "tempo_calc": numero(raw, "tempo_calc_s"),
            "tempo_paralelo": numero(raw, "tempo_calc_s"),
            "tempo_total": numero(raw, "tempo_calc_s"),
            "speedup": numero(raw, "speedup"), 
            "f_lb": numero(raw, "f_lb"),
        })
    for raw in mpi_rows:
        estrategia = raw["estrategia"].strip().lower()
        rows.append({
            "paradigma": "MPI", 
            "caso": raw["caso"],
            "resolucao": numero(raw, "resolucao", int), 
            "max_iter": numero(raw, "max_iter", int),
            "workers": numero(raw, "processos", int), 
            "estrategia": estrategia,
            "chunk": numero(raw, "chunk", int), 
            "repeticao": numero(raw, "repeticao", int),
            "tempo_serial": numero(raw, "tempo_serial_mediana_s"),
            "tempo_calc": numero(raw, "tempo_calculo_max_s"),
            "tempo_paralelo": numero(raw, "tempo_paralelo_total_s"),
            "tempo_total": numero(raw, "tempo_paralelo_total_s"),
            "speedup": numero(raw, "speedup_calculo"), 
            "f_lb": numero(raw, "f_lb"),
        })
    if not rows:
        raise ValueError("Os CSVs não contêm resultados.")
    return rows


def media(vals):
    vals = list(vals)
    return statistics.fmean(vals) if vals else math.nan


def agregar(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    keys = ("paradigma", "caso", "resolucao", "max_iter", "workers", "estrategia", "chunk")
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, group in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        item = dict(zip(keys, key))
        for metric in ("tempo_serial", "tempo_calc", "tempo_total", "tempo_paralelo", "speedup", "f_lb"):
            item[metric + "_media"] = media(r[metric] for r in group)
        item["eficiencia"] = item["speedup_media"] / item["workers"] * 100 if item["workers"] else math.nan
        item["n_repeticoes"] = len(group)
        out.append(item)
    return out


def salvar(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def nome_caso(caso: str) -> str:
    return {"padrao": "Vista completa", "cavalos": "Vale dos cavalos-marinhos"}.get(caso, caso)


def linha(label: str) -> str:
    return label.replace("_", " ").title()


def escolher_config(rows: list[dict], caso: str | None = None, resolucao: int | None = None):
    result = rows
    if caso is not None:
        result = [r for r in result if r["caso"] == caso]
    if resolucao is not None:
        result = [r for r in result if r["resolucao"] == resolucao]
    return result


def plot_mpi_tempo(mpi: list[dict], out: Path):
    # Um arquivo por caso/resolução para não misturar tamanhos de problema.
    configs = sorted({(r["caso"], r["resolucao"], r["max_iter"]) for r in mpi})
    for caso, res, max_iter in configs:
        rows = [r for r in mpi if (r["caso"], r["resolucao"], r["max_iter"]) == (caso, res, max_iter)]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for strategy in ("bloco", "blocov", "intercaladov"):
            sr = [r for r in rows if r["estrategia"] == strategy and (strategy != "intercaladov" or r["chunk"] == 1)]
            by_worker = defaultdict(list)
            for r in sr:
                by_worker[r["workers"]].append(r)
            if not by_worker:
                continue
            xs = sorted(by_worker)
            ax.plot(xs, [media(x["tempo_calc_media"] for x in by_worker[w]) for w in xs], marker="o", label=f"{strategy} — cálculo")
            ax.plot(xs,[media(x["tempo_paralelo_media"] for x in by_worker[w]) for w in xs], marker="s", linestyle="--", label=f"{strategy} — paralelo total")
        ax.set(xlabel="Número de processos MPI", ylabel="Tempo médio (s)", title=f"MPI — cálculo e tempo paralelo total\n{nome_caso(caso)} — {res}×{res} — MAX_ITER={max_iter}")
        ax.grid(True, alpha=.3); ax.legend(ncol=2)
        salvar(fig, out / f"01_mpi_calculo_total_{caso}_{res}.png")

def tempo_desempenho(r):
    if r["paradigma"] == "MPI":
        return r["tempo_paralelo_media"]
    return r["tempo_calc_media"]

def plot_strong(combined: list[dict], out: Path):
    configs = sorted({(r["caso"], r["resolucao"], r["max_iter"]) for r in combined})
    schedule_order = [("OpenMP", "static"), ("OpenMP", "dynamic"), ("OpenMP", "guided"),
                      ("MPI", "bloco"), ("MPI", "blocov"), ("MPI", "intercaladov")]
    for caso, res, max_iter in configs:
        rows = [r for r in combined if (r["caso"], r["resolucao"], r["max_iter"]) == (caso, res, max_iter)]
        series = []
        for paradigm, strategy in schedule_order:
            sr = [r for r in rows if r["paradigma"] == paradigm and r["estrategia"] == strategy
                  and ((r["paradigma"] == "OpenMP" and r["chunk"] == 1)
                       or (r["paradigma"] == "MPI" and (strategy != "intercaladov" or r["chunk"] == 1)))]
            by_worker = {r["workers"]: r for r in sr}
            if by_worker:
                series.append((f"{paradigm} {strategy}", sorted(by_worker.items())))
        for metric, ylabel, filename, ideal in [
            ("tempo_calc_media", "Tempo médio de execução paralela (s)", "02_strong_tempo", False),
            ("speedup_media", "Speedup de cálculo", "03_strong_speedup", True),
            ("eficiencia", "Eficiência (%)", "04_strong_eficiencia", False),
        ]:
            fig, ax = plt.subplots(figsize=(9.5, 6))
            for label, points in series:
                if metric == "tempo_calc_media":
                    valores = [tempo_desempenho(r) for _, r in points]
                else:
                    valores = [r[metric] for _, r in points]

                ax.plot([w for w, _ in points], valores, marker="o", label=label)
            if ideal and series:
                xs = sorted({w for _, points in series for w, _ in points})
                ax.plot(xs, xs, "k--", alpha=.55, label="Ideal S(p)=p")
            if metric == "eficiencia":
                ax.axhline(100, color="gray", linestyle="--", alpha=.6, label="Ideal 100%")
            ax.set(xlabel="Threads OpenMP / processos MPI", ylabel=ylabel,
                   title=f"Strong scaling — {ylabel.split('(')[0].strip()}\n{nome_caso(caso)} — {res}×{res} — MAX_ITER={max_iter}")
            ax.grid(True, alpha=.3); ax.legend(ncol=2, fontsize=8)
            salvar(fig, out / f"{filename}_{caso}_{res}.png")


def plot_flb_cavalos(mpi: list[dict], out: Path):
    rows = [r for r in mpi if r["caso"] == "cavalos" and r["estrategia"] in {"bloco", "blocov", "intercaladov"}
            and (r["estrategia"] != "intercaladov" or r["chunk"] == 1)]
    if not rows:
        print("AVISO: gráfico 05 ignorado; não há dados MPI do caso 'cavalos'.", file=sys.stderr)
        return
    configs = sorted({(r["resolucao"], r["max_iter"]) for r in rows})
    for res, max_iter in configs:
        subset = [r for r in rows if r["resolucao"] == res and r["max_iter"] == max_iter]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for strategy in ("bloco", "blocov", "intercaladov"):
            sr = [r for r in subset if r["estrategia"] == strategy]
            by_worker = defaultdict(list)
            for r in sr: by_worker[r["workers"]].append(r["f_lb_media"])
            if by_worker:
                xs = sorted(by_worker)
                ax.plot(xs, [media(by_worker[w]) for w in xs], marker="o", label=strategy)
        ax.set(xlabel="Número de processos MPI", ylabel="Fator de balanceamento de carga (F_LB)",
               title=f"F_LB por estratégia MPI — {nome_caso('cavalos')}\n{res}×{res} — MAX_ITER={max_iter}")
        ax.grid(True, alpha=.3); ax.legend()
        salvar(fig, out / f"05_flb_cavalos_{res}.png")


def plot_chunks(combined: list[dict], out: Path):
    # Uma família de três gráficos por combinação caso/resolução/número de workers.
    configs = sorted({(r["caso"], r["resolucao"], r["max_iter"], r["workers"])
                      for r in combined if (r["paradigma"] == "OpenMP" and r["estrategia"] in {"static", "dynamic", "guided"})
                      or (r["paradigma"] == "MPI" and r["estrategia"] == "intercaladov")})
    for caso, res, max_iter, workers in configs:
        rows = [r for r in combined if (r["caso"], r["resolucao"], r["max_iter"], r["workers"]) == (caso, res, max_iter, workers)
                and ((r["paradigma"] == "OpenMP" and r["estrategia"] in {"static", "dynamic", "guided"})
                     or (r["paradigma"] == "MPI" and r["estrategia"] == "intercaladov"))]
        series = []
        for paradigm, strategy in [("OpenMP", "static"), ("OpenMP", "dynamic"), ("OpenMP", "guided"), ("MPI", "intercaladov")]:
            sr = sorted([r for r in rows if r["paradigma"] == paradigm and r["estrategia"] == strategy], key=lambda r: r["chunk"])
            if sr:
                series.append((f"{paradigm} {strategy}", sr))
        suffix = f"{caso}_{res}_{workers}workers"
        for metric, ylabel, filename in [
            ("tempo_calc_media", "Tempo médio de execução paralela (s)", "06_chunks_tempo"),
            ("f_lb_media", "Fator de balanceamento de carga (F_LB)", "07_chunks_flb")]:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for label, sr in series:
                by_chunk = defaultdict(list)
                for r in sr:
                    valor = tempo_desempenho(r) if metric == "tempo_calc_media" else r[metric]
                    by_chunk[r["chunk"]].append(valor)
                xs = sorted(by_chunk)
                ax.plot(xs, [media(by_chunk[x]) for x in xs], marker="o", label=label)
            ax.set(xlabel="Tamanho do chunk", ylabel=ylabel,
                   title=f"{ylabel.split('(')[0].strip()} por chunk\n{nome_caso(caso)} — {res}×{res} — {workers} workers")
            ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
            salvar(fig, out / f"{filename}_{suffix}.png")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, sr in series:
            ax.scatter([r["f_lb_media"] for r in sr], [tempo_desempenho(r) for r in sr], label=label)
            for r in sr:
                ax.annotate(str(r["chunk"]),(r["f_lb_media"], tempo_desempenho(r)), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Fator de balanceamento de carga (F_LB)", ylabel="Tempo médio de execução paralela (s)",
            title=f"F_LB × tempo de execução paralela\n{nome_caso(caso)} — {res}×{res} — {workers} workers")
        ax.grid(True, alpha=.3); ax.legend(fontsize=8, ncol=2)
        salvar(fig, out / f"08_chunks_flb_tempo_{suffix}.png")


def plot_weak(combined: list[dict], out: Path, caso_alvo: str = "padrao"):
    rows = [r for r in combined if r["caso"] == caso_alvo and
            ((r["paradigma"] == "OpenMP" and r["chunk"] == 1 and r["estrategia"] in {"static", "dynamic", "guided"})
             or (r["paradigma"] == "MPI" and (r["estrategia"] != "intercaladov" or r["chunk"] == 1)))]
    if not rows:
        print(f"AVISO: gráfico 09 ignorado; não há dados de weak scaling para caso={caso_alvo!r}.", file=sys.stderr)
        return

    series_defs = [("OpenMP", "static"), ("OpenMP", "dynamic"), ("OpenMP", "guided"),
                   ("MPI", "bloco"), ("MPI", "blocov"), ("MPI", "intercaladov")]
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for paradigm, strategy in series_defs:
        sr = [r for r in rows if r["paradigma"] == paradigm and r["estrategia"] == strategy]
        resolutions = sorted({r["resolucao"] for r in sr})
        points = []

        for i, res in enumerate(resolutions):
            res_rows = [r for r in sr if r["resolucao"] == res]
            worker_values = sorted({r["workers"] for r in res_rows})
            if not worker_values:
                continue
            target_worker = worker_values[min(i, len(worker_values) - 1)]
            selected = [r for r in res_rows if r["workers"] == target_worker]
            points.append((res, media(tempo_desempenho(r) for r in selected), target_worker))

        if len(points) >= 2:
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=f"{paradigm} {strategy}")
            for res, t, workers in points:
                ax.annotate(f"{workers}", (res, t), xytext=(4, 4), textcoords="offset points", fontsize=7)
            plotted += 1

    if plotted == 0:
        print("AVISO: gráfico 09 ignorado; são necessárias pelo menos duas resoluções por estratégia.", file=sys.stderr)
        plt.close(fig)
        return

    ax.set(xlabel="Resolução (largura × altura)", ylabel="Tempo médio de execução paralela (s)",
           title=f"Weak scaling — {nome_caso(caso_alvo)}\nNúmeros junto aos pontos indicam threads/processos")
    ax.grid(True, alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    salvar(fig, out / "09_weak_scaling_seis_estrategias.png")


def main():
    p = argparse.ArgumentParser(description="Gera os gráficos combinados dos benchmarks OpenMP e MPI.")
    p.add_argument("--omp-csv", type=Path, required=True, help="CSV bruto do benchmark OpenMP")
    p.add_argument("--mpi-csv", type=Path, required=True, help="CSV bruto do benchmark MPI")
    p.add_argument("--output-dir", type=Path, default=Path("resultados/analise_combinada"))
    p.add_argument("--weak-case", choices=["padrao", "cavalos"], default="padrao")
    args = p.parse_args()
    try:
        omp_raw = ler_csv(args.omp_csv, OMP_REQUIRED)
        mpi_raw = ler_csv(args.mpi_csv, MPI_REQUIRED)
        all_raw = normalizar(omp_raw, mpi_raw)
        combined = agregar(all_raw)
        out = args.output_dir.expanduser().resolve() / "graficos"
        out.mkdir(parents=True, exist_ok=True)
        # Primeiro gráfico compara o tempo de cálculo com o tempo paralelo total do MPI.
        plot_mpi_tempo([r for r in combined if r["paradigma"] == "MPI"], out)
        plot_strong(combined, out)
        plot_flb_cavalos([r for r in combined if r["paradigma"] == "MPI"], out)
        plot_chunks(combined, out)
        plot_weak(combined, out, args.weak_case)
        # Exporta uma tabela normalizada/agregada para conferência e reutilização.
        columns = ["paradigma", "caso", "resolucao", "max_iter", "workers", "estrategia", "chunk",
           "n_repeticoes", "tempo_serial_media", "tempo_calc_media", "tempo_paralelo_media",
           "tempo_total_media", "speedup_media", "eficiencia", "f_lb_media"]
        csv_out = args.output_dir.expanduser().resolve() / "resultados_agregados_combinados.csv"
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in combined:
                writer.writerow({k: (f"{r[k]:.10f}" if isinstance(r.get(k), float) and math.isfinite(r[k]) else r.get(k, "")) for k in columns})
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
    print(f"Análise concluída. Gráficos: {out}")
    print(f"CSV agregado: {csv_out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
