#!/usr/bin/env python3
"""
Automatiza os experimentos da versao OpenMP de Mandelbrot.

O script:
- define OMP_NUM_THREADS e OMP_SCHEDULE;
- executa o programa varias vezes;
- captura os tempos e o F_LB impressos pelo mandelbrot_omp;
- grava uma linha por execucao em CSV.

Exemplo simples:
    python3 benchmark_omp.py \
        --threads 1 2 4 8 \
        --schedules static \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 \
        --repeats 5

Strong scaling:
    python3 benchmark_omp.py \
        --threads 1 2 4 8 \
        --schedules static \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/strong_scaling.csv

Comparacao de schedules/chunks no caso de desbalanceamento:
    python3 benchmark_omp.py \
        --threads 8 \
        --schedules static dynamic guided \
        --chunks 1 16 64 256 \
        --cases cavalos \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/scheduling_cavalos.csv

Weak scaling (ajuste as threads conforme o hardware):
    python3 benchmark_omp.py \
        --threads 1 4 16 \
        --schedules static \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 8192 16384 \
        --repeats 5 \
        --output resultados/brutos/weak_scaling.csv

Use --dry-run para conferir a quantidade de execucoes antes de iniciar.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RE_TEMPO_CALC = re.compile(
    r"Tempo de calculo:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)

RE_TEMPO_IO = re.compile(
    r"Tempo de I/O:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)

RE_TEMPO_TOTAL = re.compile(
    r"Tempo total:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)

RE_THREADS = re.compile(
    r"Threads utilizadas:\s*(\d+)"
)

RE_SCHEDULE = re.compile(
    r"Schedule:\s*([A-Za-z]+)"
)

RE_CHUNK = re.compile(
    r"Chunk:\s*(\d+|padrao)"
)

RE_TMIN = re.compile(
    r"T_min:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)

RE_TMAX = re.compile(
    r"T_max:\s*([0-9]+(?:\.[0-9]+)?)\s*s"
)

RE_FLB = re.compile(
    r"Fator F_LB:\s*([0-9]+(?:\.[0-9]+)?)"
)

RE_MAX_ITER = re.compile(
    r"MAX_ITER:\s*(\d+)"
)


CSV_FIELDS = [
    "data_hora",
    "executavel",
    "caso",
    "resolucao",
    "max_iter",
    "threads_solicitadas",
    "threads_utilizadas",
    "schedule_configurado",
    "schedule_utilizado",
    "chunk_configurado",
    "chunk_utilizado",
    "repeticao",
    "tempo_calc_s",
    "tempo_io_s",
    "tempo_total_s",
    "t_min_s",
    "t_max_s",
    "f_lb",
    "returncode",
]


def parse_float(pattern, text, field):
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{field}' na saida.")
    return float(match.group(1))


def parse_int(pattern, text, field):
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{field}' na saida.")
    return int(match.group(1))


def parse_text(pattern, text, field):
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Nao foi possivel encontrar '{field}' na saida.")
    return match.group(1)


def parse_chunk(text):
    match = RE_CHUNK.search(text)

    if not match:
        raise ValueError("Nao foi possivel encontrar 'Chunk' na saida.")

    valor = match.group(1)

    if valor == "padrao":
        return 0

    return int(valor)


def executar(
    executavel,
    caso,
    resolucao,
    threads,
    schedule,
    chunk,
    repeticao,
    cwd,
    timeout,
):
    ambiente = os.environ.copy()

    # Evita que a biblioteca OpenMP altere dinamicamente o numero
    # de threads solicitado durante o benchmark.
    ambiente["OMP_NUM_THREADS"] = str(threads)
    ambiente["OMP_DYNAMIC"] = "FALSE"

    ambiente["OMP_SCHEDULE"] = f"{schedule},{chunk}"

    comando = [
        str(executavel),
        caso,
        str(resolucao),
    ]

    inicio = datetime.now()

    processo = subprocess.run(
        comando,
        cwd=cwd,
        env=ambiente,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    saida = processo.stdout
    stderr = processo.stderr

    if processo.returncode != 0:
        raise RuntimeError(
            "O programa terminou com erro.\n"
            f"Comando: {' '.join(comando)}\n"
            f"Codigo de saida: {processo.returncode}\n"
            f"STDOUT:\n{saida}\n"
            f"STDERR:\n{stderr}"
        )

    resultado = {
        "data_hora": inicio.isoformat(timespec="seconds"),
        "executavel": str(executavel),
        "caso": caso,
        "resolucao": resolucao,
        "max_iter": parse_int(RE_MAX_ITER, saida, "MAX_ITER"),
        "threads_solicitadas": threads,
        "threads_utilizadas": parse_int(
            RE_THREADS,
            saida,
            "Threads utilizadas",
        ),
        "schedule_configurado": schedule,
        "schedule_utilizado": parse_text(
            RE_SCHEDULE,
            saida,
            "Schedule",
        ),
        "chunk_configurado": chunk,
        "chunk_utilizado": parse_chunk(saida),
        "repeticao": repeticao,
        "tempo_calc_s": parse_float(
            RE_TEMPO_CALC,
            saida,
            "Tempo de calculo",
        ),
        "tempo_io_s": parse_float(
            RE_TEMPO_IO,
            saida,
            "Tempo de I/O",
        ),
        "tempo_total_s": parse_float(
            RE_TEMPO_TOTAL,
            saida,
            "Tempo total",
        ),
        "t_min_s": parse_float(
            RE_TMIN,
            saida,
            "T_min",
        ),
        "t_max_s": parse_float(
            RE_TMAX,
            saida,
            "T_max",
        ),
        "f_lb": parse_float(
            RE_FLB,
            saida,
            "Fator F_LB",
        ),
        "returncode": processo.returncode,
    }

    return resultado, saida, stderr


def quantidade_execucoes(cases, resolutions, threads, schedules, chunks, repeats):
    return (
        len(cases)
        * len(resolutions)
        * len(threads)
        * len(schedules)
        * len(chunks)
        * repeats
    )


def main():
    parser = argparse.ArgumentParser(
        description="Automatiza benchmarks OpenMP do Mandelbrot."
    )

    parser.add_argument(
        "--exe",
        type=Path,
        default=Path("./mandelbrot_omp"),
        help="Caminho para o executavel OpenMP.",
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["padrao", "cavalos"],
        default=["padrao"],
        help="Casos a executar.",
    )

    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[4096],
        help="Resolucao(s) quadrada(s).",
    )

    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Numero de threads a testar.",
    )

    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=["static", "dynamic", "guided"],
        default=["static"],
        help="Politicas de escalonamento.",
    )

    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=[1],
        help="Tamanhos de chunk.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Quantidade de repeticoes por configuracao.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados/brutos/omp.csv"),
        help="Arquivo CSV de saida.",
    )

    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path("."),
        help="Diretorio de trabalho do executavel.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Tempo maximo de cada execucao, em segundos.",
    )

    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Diretorio opcional para salvar stdout/stderr de cada execucao.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra as configuracoes, sem executar.",
    )

    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats deve ser > 0.")

    if any(value <= 0 for value in args.resolutions):
        parser.error("Todas as resolucoes devem ser > 0.")

    if any(value <= 0 for value in args.threads):
        parser.error("Todas as quantidades de threads devem ser > 0.")

    if any(value <= 0 for value in args.chunks):
        parser.error("Todos os chunks devem ser > 0.")

    if args.timeout <= 0:
        parser.error("--timeout deve ser > 0.")

    executavel = args.exe
    if not executavel.is_absolute():
        executavel = (Path.cwd() / executavel).resolve()

    cwd = args.cwd
    if not cwd.is_absolute():
        cwd = (Path.cwd() / cwd).resolve()

    if not executavel.is_file():
        parser.error(f"Executavel nao encontrado: {executavel}")

    if not cwd.is_dir():
        parser.error(f"Diretorio de trabalho nao encontrado: {cwd}")

    total = quantidade_execucoes(
        args.cases,
        args.resolutions,
        args.threads,
        args.schedules,
        args.chunks,
        args.repeats,
    )

    print("========================================")
    print("        BENCHMARK OPENMP")
    print("========================================")
    print(f"Executavel:   {executavel}")
    print(f"Diretorio:    {cwd}")
    print(f"Casos:        {', '.join(args.cases)}")
    print(f"Resolucao:    {args.resolutions}")
    print(f"Threads:      {args.threads}")
    print(f"Schedules:    {', '.join(args.schedules)}")
    print(f"Chunks:       {args.chunks}")
    print(f"Repeticoes:   {args.repeats}")
    print(f"Total de execucoes: {total}")
    print("========================================")

    if args.dry_run:
        print("\nConfiguracoes:")
        numero = 0

        for caso in args.cases:
            for resolucao in args.resolutions:
                for threads in args.threads:
                    for schedule in args.schedules:
                        for chunk in args.chunks:
                            for repeticao in range(1, args.repeats + 1):
                                numero += 1
                                print(
                                    f"{numero:4d}. "
                                    f"{caso:8s} "
                                    f"{resolucao:5d} "
                                    f"{threads:3d} threads "
                                    f"{schedule:7s},{chunk:<4d} "
                                    f"rep={repeticao}"
                                )

        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.logs_dir is not None:
        args.logs_dir.mkdir(parents=True, exist_ok=True)

    escrever_cabecalho = not args.output.exists() or args.output.stat().st_size == 0

    with args.output.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as arquivo_csv:

        writer = csv.DictWriter(
            arquivo_csv,
            fieldnames=CSV_FIELDS,
        )

        if escrever_cabecalho:
            writer.writeheader()

        numero = 0

        for caso in args.cases:
            for resolucao in args.resolutions:
                for threads in args.threads:
                    for schedule in args.schedules:
                        for chunk in args.chunks:
                            for repeticao in range(1, args.repeats + 1):

                                numero += 1

                                print(
                                    f"\n[{numero}/{total}] "
                                    f"{caso} | "
                                    f"{resolucao}x{resolucao} | "
                                    f"{threads} threads | "
                                    f"{schedule},{chunk} | "
                                    f"repeticao {repeticao}"
                                )

                                try:
                                    resultado, stdout, stderr = executar(
                                        executavel=executavel,
                                        caso=caso,
                                        resolucao=resolucao,
                                        threads=threads,
                                        schedule=schedule,
                                        chunk=chunk,
                                        repeticao=repeticao,
                                        cwd=cwd,
                                        timeout=args.timeout,
                                    )
                                except (
                                    OSError,
                                    subprocess.TimeoutExpired,
                                    RuntimeError,
                                    ValueError,
                                ) as exc:
                                    print(
                                        f"ERRO na execucao {numero}: {exc}",
                                        file=sys.stderr,
                                    )
                                    return 1

                                writer.writerow(resultado)
                                arquivo_csv.flush()

                                print(
                                    f"  Tempo calculo: "
                                    f"{resultado['tempo_calc_s']:.6f} s"
                                )
                                print(
                                    f"  Tempo I/O:     "
                                    f"{resultado['tempo_io_s']:.6f} s"
                                )
                                print(
                                    f"  F_LB:          "
                                    f"{resultado['f_lb']:.6f}"
                                )

                                if args.logs_dir is not None:
                                    nome_base = (
                                        f"{numero:04d}_"
                                        f"{caso}_"
                                        f"{resolucao}_"
                                        f"{threads}t_"
                                        f"{schedule}_{chunk}_"
                                        f"rep{repeticao}"
                                    )

                                    (args.logs_dir / f"{nome_base}.out").write_text(
                                        stdout,
                                        encoding="utf-8",
                                    )

                                    (args.logs_dir / f"{nome_base}.err").write_text(
                                        stderr,
                                        encoding="utf-8",
                                    )

    print("\n========================================")
    print("Benchmark concluido.")
    print(f"Resultados: {args.output}")
    print("========================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
