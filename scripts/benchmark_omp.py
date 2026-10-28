#!/usr/bin/env python3
"""Benchmark OpenMP usando referencias seriais persistentes.

Execute a partir da raiz do projeto. Antes, gere as referencias com gerar_referencias.py.

Exemplo:
    python3 benchmark_omp.py \
        --exe ./openmp/mandelbrot_omp --cwd ./openmp \
        --references-dir ./referencias \
        --threads 1 2 4 8 --schedules static --chunks 1 \
        --cases padrao --resolutions 4096 --repeats 5 \
        --output resultados/brutos/omp.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from referencias_mandelbrot import (
    carregar_manifesto,
    comparar_binarios,
    nome_arquivo_binario,
    obter_referencia,
)

RE_TEMPO_CALC = re.compile(r"Tempo de calculo:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_TEMPO_IO = re.compile(r"Tempo de I/O:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_TEMPO_TOTAL = re.compile(r"Tempo total:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_THREADS = re.compile(r"Threads utilizadas:\s*(\d+)")
RE_SCHEDULE = re.compile(r"Schedule:\s*([A-Za-z]+)")
RE_CHUNK = re.compile(r"Chunk:\s*(\d+|padrao)")
RE_TMIN = re.compile(r"T_min:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_TMAX = re.compile(r"T_max:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_FLB = re.compile(r"Fator F_LB:\s*([0-9]+(?:\.[0-9]+)?)")
RE_MAX_ITER = re.compile(r"MAX_ITER:\s*(\d+)")

# As primeiras colunas mantem compatibilidade com analisar_resultados_omp.py.
CSV_FIELDS = [
    "data_hora", "executavel", "caso", "resolucao", "max_iter",
    "threads_solicitadas", "threads_utilizadas", "schedule_configurado",
    "schedule_utilizado", "chunk_configurado", "chunk_utilizado", "repeticao",
    "tempo_serial_s", "tempo_serial_min_s", "tempo_serial_max_s",
    "speedup", "diferenca_percentual", "pixels_diferentes", "percentual_erro",
    "diferenca_maxima", "correto_exato", "referencia_binario", "tempo_calc_s",
    "tempo_io_s", "tempo_total_s", "t_min_s", "t_max_s", "f_lb", "returncode",
]


def parse_float(pattern, texto, campo):
    match = pattern.search(texto)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{campo}' na saida OpenMP.\n{texto}")
    return float(match.group(1))


def parse_int(pattern, texto, campo):
    match = pattern.search(texto)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{campo}' na saida OpenMP.\n{texto}")
    return int(match.group(1))


def parse_text(pattern, texto, campo):
    match = pattern.search(texto)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{campo}' na saida OpenMP.\n{texto}")
    return match.group(1)


def parse_chunk(texto):
    match = RE_CHUNK.search(texto)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar 'Chunk' na saida OpenMP.\n{texto}")
    return 0 if match.group(1) == "padrao" else int(match.group(1))


def executar_openmp(executavel, caso, resolucao, threads, schedule, chunk, repeticao, cwd, timeout):
    ambiente = os.environ.copy()
    ambiente["OMP_NUM_THREADS"] = str(threads)
    ambiente["OMP_DYNAMIC"] = "FALSE"
    ambiente["OMP_SCHEDULE"] = f"{schedule},{chunk}"

    comando = [str(executavel), caso, str(resolucao)]
    inicio = datetime.now().astimezone().isoformat(timespec="seconds")
    processo = subprocess.run(
        comando, cwd=cwd, env=ambiente, capture_output=True, text=True, timeout=timeout
    )
    saida, stderr = processo.stdout, processo.stderr
    if processo.returncode != 0:
        raise RuntimeError(
            "A execucao OpenMP terminou com erro.\n"
            f"Comando: {' '.join(comando)}\nCodigo: {processo.returncode}\n"
            f"STDOUT:\n{saida}\nSTDERR:\n{stderr}"
        )

    max_iter = parse_int(RE_MAX_ITER, saida, "MAX_ITER")
    resultado = {
        "data_hora": inicio,
        "executavel": str(executavel),
        "caso": caso,
        "resolucao": resolucao,
        "max_iter": max_iter,
        "threads_solicitadas": threads,
        "threads_utilizadas": parse_int(RE_THREADS, saida, "Threads utilizadas"),
        "schedule_configurado": schedule,
        "schedule_utilizado": parse_text(RE_SCHEDULE, saida, "Schedule"),
        "chunk_configurado": chunk,
        "chunk_utilizado": parse_chunk(saida),
        "repeticao": repeticao,
        "tempo_calc_s": parse_float(RE_TEMPO_CALC, saida, "Tempo de calculo"),
        "tempo_io_s": parse_float(RE_TEMPO_IO, saida, "Tempo de I/O"),
        "tempo_total_s": parse_float(RE_TEMPO_TOTAL, saida, "Tempo total"),
        "t_min_s": parse_float(RE_TMIN, saida, "T_min"),
        "t_max_s": parse_float(RE_TMAX, saida, "T_max"),
        "f_lb": parse_float(RE_FLB, saida, "Fator F_LB"),
        "returncode": processo.returncode,
    }
    return resultado, saida, stderr


def quantidade_execucoes(casos, resolucoes, threads, schedules, chunks, repeticoes):
    return len(casos) * len(resolucoes) * len(threads) * len(schedules) * len(chunks) * repeticoes


def validar_csv_existente(path: Path):
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            cabecalho = next(csv.reader(f), [])
        if cabecalho != CSV_FIELDS:
            raise ValueError(
                f"O CSV existente {path} tem um cabecalho de outra versao. "
                "Use outro --output ou mova/remova o CSV antigo antes de continuar."
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenMP usando referencia serial compartilhada.")
    parser.add_argument("--exe", type=Path, default=Path("./openmp/mandelbrot_omp"), help="Executavel OpenMP.")
    parser.add_argument("--cwd", type=Path, default=Path("./openmp"), help="Diretorio de trabalho do executavel.")
    parser.add_argument("--references-dir", type=Path, default=Path("./referencias"), help="Diretorio gerado por gerar_referencias.py.")
    parser.add_argument("--cases", nargs="+", choices=["padrao", "cavalos"], default=["padrao"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[4096])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--schedules", nargs="+", choices=["static", "dynamic", "guided"], default=["static"])
    parser.add_argument("--chunks", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("resultados/brutos/omp.csv"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0 or args.timeout <= 0:
        parser.error("--repeats e --timeout devem ser positivos.")
    if any(x <= 0 for x in args.resolutions + args.threads + args.chunks):
        parser.error("Resolucao, threads e chunks devem ser positivos.")

    executavel = args.exe.expanduser().resolve()
    cwd = args.cwd.expanduser().resolve()
    references_dir = args.references_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    logs_dir = args.logs_dir.expanduser().resolve() if args.logs_dir else None

    if not executavel.is_file():
        parser.error(f"Executavel OpenMP nao encontrado: {executavel}")
    if not cwd.is_dir():
        parser.error(f"Diretorio de trabalho OpenMP nao encontrado: {cwd}")

    total = quantidade_execucoes(args.cases, args.resolutions, args.threads, args.schedules, args.chunks, args.repeats)
    print("=============== BENCHMARK OPENMP ===============")
    print(f"Executavel: {executavel}")
    print(f"Diretorio: {cwd}")
    print(f"Referencias: {references_dir}")
    print(f"Casos: {args.cases}; resolucoes: {args.resolutions}")
    print(f"Threads: {args.threads}; schedules: {args.schedules}; chunks: {args.chunks}")
    print(f"Repeticoes: {args.repeats}; total de execucoes: {total}")
    print("=================================================")

    if args.dry_run:
        i = 0
        for caso in args.cases:
            for resolucao in args.resolutions:
                for threads in args.threads:
                    for schedule in args.schedules:
                        for chunk in args.chunks:
                            for repeticao in range(1, args.repeats + 1):
                                i += 1
                                print(f"{i:4d}. {caso} {resolucao}x{resolucao}, {threads} threads, {schedule},{chunk}, rep={repeticao}")
        return 0

    try:
        manifesto = carregar_manifesto(references_dir)
        referencias = {}
        for caso in args.cases:
            for resolucao in args.resolutions:
                chave = (caso, resolucao)
                if chave not in referencias:
                    referencias[chave] = obter_referencia(references_dir, manifesto, caso, resolucao)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERRO ao carregar referencias: {exc}", file=sys.stderr)
        return 1

    try:
        validar_csv_existente(output)
    except ValueError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
    escrever_cabecalho = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=CSV_FIELDS)
        if escrever_cabecalho:
            writer.writeheader()

        numero = 0
        for caso in args.cases:
            for resolucao in args.resolutions:
                referencia = referencias[(caso, resolucao)]
                print(
                    f"\nReferencia serial: {caso} {resolucao}x{resolucao} | "
                    f"mediana={referencia['tempo_serial_mediana_s']:.6f} s | "
                    f"MAX_ITER={referencia['max_iter']}"
                )

                for threads in args.threads:
                    for schedule in args.schedules:
                        for chunk in args.chunks:
                            for repeticao in range(1, args.repeats + 1):
                                numero += 1
                                print(
                                    f"\n[{numero}/{total}] {caso} {resolucao}x{resolucao} | "
                                    f"{threads} threads | {schedule},{chunk} | rep={repeticao}/{args.repeats}"
                                )
                                arquivo_saida = cwd / "saida" / nome_arquivo_binario(caso, resolucao)
                                arquivo_saida.unlink(missing_ok=True)
                                try:
                                    resultado, stdout, stderr = executar_openmp(
                                        executavel, caso, resolucao, threads, schedule, chunk,
                                        repeticao, cwd, args.timeout
                                    )
                                    if resultado["max_iter"] != referencia["max_iter"]:
                                        raise ValueError(
                                            f"MAX_ITER do OpenMP ({resultado['max_iter']}) difere da referencia "
                                            f"({referencia['max_iter']})."
                                        )
                                    diferentes, percentual, max_diferenca = comparar_binarios(
                                        referencia["binario"], arquivo_saida
                                    )
                                    tempo_serial = referencia["tempo_serial_mediana_s"]
                                    resultado.update({
                                        "tempo_serial_s": tempo_serial,
                                        "tempo_serial_min_s": referencia["tempo_serial_min_s"],
                                        "tempo_serial_max_s": referencia["tempo_serial_max_s"],
                                        "speedup": tempo_serial / resultado["tempo_calc_s"] if resultado["tempo_calc_s"] > 0 else "",
                                        "diferenca_percentual": ((resultado["tempo_calc_s"] - tempo_serial) / tempo_serial * 100.0) if tempo_serial > 0 else "",
                                        "pixels_diferentes": diferentes,
                                        "percentual_erro": percentual,
                                        "diferenca_maxima": max_diferenca,
                                        "correto_exato": "SIM" if diferentes == 0 else "NAO",
                                        "referencia_binario": referencia["arquivo_binario"],
                                    })
                                except (OSError, RuntimeError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                                    print(f"ERRO na execucao {numero}: {exc}", file=sys.stderr)
                                    return 1

                                writer.writerow(resultado)
                                arquivo_csv.flush()
                                print(f"  Tempo de calculo: {resultado['tempo_calc_s']:.6f} s")
                                print(f"  Serial (mediana): {resultado['tempo_serial_s']:.6f} s")
                                print(f"  Speedup: {resultado['speedup']:.4f}")
                                print(f"  F_LB: {resultado['f_lb']:.6f}")
                                print(f"  Corretude exata: {resultado['correto_exato']} ({diferentes} pixels diferentes)")

                                if logs_dir:
                                    nome_base = f"{numero:04d}_{caso}_{resolucao}_{threads}t_{schedule}_{chunk}_rep{repeticao}"
                                    (logs_dir / f"{nome_base}.out").write_text(stdout, encoding="utf-8")
                                    (logs_dir / f"{nome_base}.err").write_text(stderr, encoding="utf-8")

    print(f"\nBenchmark OpenMP concluido. CSV: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
