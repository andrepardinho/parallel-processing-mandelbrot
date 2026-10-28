#!/usr/bin/env python3
"""Benchmark das estrategias MPI usando referencias seriais compartilhadas.

Antes de executar, gere as referencias uma unica vez para cada combinação caso x resolução:
    python3 gerar_referencias.py 
    --cases padrao cavalos \
    --resolutions 4096 8192\
    --repeats 5

Exemplo de benchmark MPI:
    python3 benchmark_mpi.py \
        --mpi-exe ./mpi/mandelbrot_mpi --mpi-cwd ./mpi \
        --references-dir ./referencias \
        --processes 1 2 4 8 --strategies blocov intercaladov \
        --chunks 1 4 16 64 --cases padrao --resolutions 4096 \
        --repeats 5 --output resultados/brutos/mpi.csv

Comparação de strategies MPI de tempo de calculo e tempo paralelo total
(inclui o tempo de reorganização, mas não o tempo de I/O) 
    python3 benchmark_mpi.py \
        --processes 1 2 4 8 \
        --strategies bloco blocov intercaladov \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/mpi_strategies_time.csv


Strong scaling
(segue os mesmos argumentos que o anterior, mas deverá ser comparado com o strong scaling do OMP):
    python3 benchmark_mpi.py \
        --processes 1 2 4 8 \
        --strategies bloco blocov intercaladov \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/strong_scaling.csv

Comparacao de strategies do MPI com chunk tamanho 1 no caso de desbalanceamento:
    python3 benchmark_mpi.py \
        --processes 8 \
        --strategies bloco blocov intercaladov \
        --chunks 1 \
        --cases cavalos \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/strategies_cavalos_chunk_1.csv

Comparacao de strategies (apenas intercaladov) por chunks no caso de desbalanceamento
(deverá ser comparado com o teste de chunks de OMP):
    python3 benchmark_mpi.py \
        --processes 8 \
        --strategies intercaladov \
        --chunks 1 16 64 256 \
        --cases cavalos \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/scheduling_cavalos.csv

Weak scaling
(deverá ser comparado com o weak scaling do OMP):
    python3 benchmark_mpi.py \
        --processes 2 \
        --strategies bloco blocov intercaladov \
        --chunks 1 \
        --cases padrao \
        --resolutions 4096 \
        --repeats 5 \
        --output resultados/brutos/weak_scaling.csv

e

    python3 benchmark_mpi.py \
        --processes 8 \
        --strategies bloco blocov intercaladov \
        --chunks 1 \
        --cases padrao \
        --resolutions 8192 \
        --repeats 5 \
        --output resultados/brutos/weak_scaling.csv

O argumento --chunks so gera combinacoes adicionais para intercaladov.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from referencias_mandelbrot import (
    carregar_manifesto,
    comparar_binarios,
    nome_arquivo_binario,
    obter_referencia,
)

RE_MAX_ITER = re.compile(r"MAX_ITER:\s*(\d+)")
RE_ESTRATEGIA = re.compile(r"ESTRATEGIA\s*=\s*([A-Za-z]+)")
RE_PROCESSOS = re.compile(r"PROCESSOS\s*=\s*(\d+)")
RE_CHUNK = re.compile(r"CHUNK\s*=\s*(\d+)")
RE_TCALC_MIN = re.compile(r"TEMPO_CALCULO_MIN_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TCALC_MAX = re.compile(r"TEMPO_CALCULO_MAX_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_FLB = re.compile(r"F_LB\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TCOLETA = re.compile(r"TEMPO_COLETA_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TREORG = re.compile(r"TEMPO_REORGANIZACAO_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TPARALELO = re.compile(r"TEMPO_PARALELO_TOTAL_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TIO = re.compile(r"TEMPO_IO_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_TTOTAL_IO = re.compile(r"TEMPO_TOTAL_COM_IO_S\s*=\s*([0-9]+(?:\.[0-9]+)?)")
RE_STATUS_SAIDA = re.compile(r"STATUS_SAIDA\s*=\s*(OK|ERRO)")

CSV_FIELDS = [
    "data_hora", "executavel_mpi", "caso", "resolucao", "max_iter",
    "estrategia", "processos", "chunk", "repeticao",
    "tempo_serial_min_s", "tempo_serial_mediana_s", "tempo_serial_max_s",
    "referencia_binario", "tempo_calculo_min_s", "tempo_calculo_max_s", "f_lb",
    "tempo_coleta_s", "tempo_reorganizacao_s", "tempo_paralelo_total_s",
    "tempo_io_s", "tempo_total_com_io_s", "tempo_wall_processo_s",
    "speedup_calculo", "speedup_paralelo_total",
    "pixels_diferentes", "percentual_diferenca", "diferenca_maxima",
    "correto_exato", "status_saida", "returncode",
]


def parse_float(pattern, text, field):
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Nao foi possivel ler '{field}' da saida MPI.\nSaida:\n{text}")
    return float(match.group(1))


def parse_int(pattern, text, field):
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Nao foi possivel ler '{field}' da saida MPI.\nSaida:\n{text}")
    return int(match.group(1))


def executar_mpi(mpirun, executavel, caso, resolucao, estrategia, processos,
                 chunk, cwd, timeout):
    comando = [mpirun, "-np", str(processos), str(executavel), caso, str(resolucao), estrategia]
    if estrategia == "intercaladov":
        comando.append(str(chunk))

    arquivo_bin = cwd / "saida" / nome_arquivo_binario(caso, resolucao)
    arquivo_bin.unlink(missing_ok=True)

    inicio_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    inicio = time.perf_counter()
    processo = subprocess.run(
        comando, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    tempo_wall = time.perf_counter() - inicio
    saida = processo.stdout
    stderr = processo.stderr

    if processo.returncode != 0:
        raise RuntimeError(
            "A execucao MPI terminou com erro.\n"
            f"Comando: {' '.join(comando)}\n"
            f"Codigo: {processo.returncode}\nSTDOUT:\n{saida}\nSTDERR:\n{stderr}"
        )

    status = re.search(RE_STATUS_SAIDA, saida)
    if not status or status.group(1) != "OK":
        raise RuntimeError(f"MPI nao confirmou STATUS_SAIDA=OK.\nSTDOUT:\n{saida}\nSTDERR:\n{stderr}")

    dados = {
        "data_hora": inicio_iso,
        "executavel_mpi": str(executavel),
        "caso": caso,
        "resolucao": resolucao,
        "max_iter": parse_int(RE_MAX_ITER, saida, "MAX_ITER"),
    }
    # Le os identificadores impressos pelo proprio programa.
    match_estrategia = RE_ESTRATEGIA.search(saida)
    if not match_estrategia:
        raise ValueError(f"Nao foi possivel ler a estrategia MPI.\nSaida:\n{saida}")
    match_processos = RE_PROCESSOS.search(saida)
    if not match_processos:
        raise ValueError(f"Nao foi possivel ler o numero de processos.\nSaida:\n{saida}")

    estrategia_real = match_estrategia.group(1)
    processos_reais = int(match_processos.group(1))
    chunk_real = parse_int(RE_CHUNK, saida, "CHUNK")
    if estrategia_real != estrategia or processos_reais != processos:
        raise ValueError(
            f"Configuracao retornada pelo MPI ({estrategia_real}, {processos_reais}) "
            f"difere da solicitada ({estrategia}, {processos})."
        )

    dados.update({
        "estrategia": estrategia_real,
        "processos": processos_reais,
        "chunk": chunk_real,
        "tempo_calculo_min_s": parse_float(RE_TCALC_MIN, saida, "TEMPO_CALCULO_MIN_S"),
        "tempo_calculo_max_s": parse_float(RE_TCALC_MAX, saida, "TEMPO_CALCULO_MAX_S"),
        "f_lb": parse_float(RE_FLB, saida, "F_LB"),
        "tempo_coleta_s": parse_float(RE_TCOLETA, saida, "TEMPO_COLETA_S"),
        "tempo_reorganizacao_s": parse_float(RE_TREORG, saida, "TEMPO_REORGANIZACAO_S"),
        "tempo_paralelo_total_s": parse_float(RE_TPARALELO, saida, "TEMPO_PARALELO_TOTAL_S"),
        "tempo_io_s": parse_float(RE_TIO, saida, "TEMPO_IO_S"),
        "tempo_total_com_io_s": parse_float(RE_TTOTAL_IO, saida, "TEMPO_TOTAL_COM_IO_S"),
        "tempo_wall_processo_s": tempo_wall,
        "returncode": processo.returncode,
    })
    return dados, saida, stderr, arquivo_bin


def montar_configuracoes(casos, resolucoes, processos, estrategias, chunks, repeticoes):
    configuracoes = []
    ignoradas = []
    for caso in casos:
        for resolucao in resolucoes:
            for nproc in processos:
                for estrategia in estrategias:
                    if estrategia == "bloco" and resolucao % nproc != 0:
                        ignoradas.append(
                            f"{caso} {resolucao}x{resolucao}, P={nproc}, bloco: "
                            "altura nao divisivel pelo numero de processos"
                        )
                        continue
                    chunks_estrategia = chunks if estrategia == "intercaladov" else [0]
                    for chunk in chunks_estrategia:
                        for repeticao in range(1, repeticoes + 1):
                            configuracoes.append((caso, resolucao, nproc, estrategia, chunk, repeticao))
    return configuracoes, ignoradas


def validar_csv_existente(path: Path):
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as arquivo:
            cabecalho = next(csv.reader(arquivo), [])
        if cabecalho != CSV_FIELDS:
            raise ValueError(
                f"O CSV existente {path} tem cabecalho de outra versao. "
                "Use outro --output ou mova/remova o CSV antigo."
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark das estrategias MPI usando referencia serial persistente."
    )
    parser.add_argument("--mpi-exe", type=Path, default=Path("./mpi/mandelbrot_mpi"),
                        help="Caminho do executavel MPI.")
    parser.add_argument("--mpirun", default="mpirun", help="Launcher MPI (mpirun ou mpiexec).")
    parser.add_argument("--mpi-cwd", type=Path, default=Path("./mpi"),
                        help="Diretorio de execucao do MPI.")
    parser.add_argument("--references-dir", type=Path, default=Path("./referencias"),
                        help="Diretorio criado por gerar_referencias.py.")
    parser.add_argument("--cases", nargs="+", choices=["padrao", "cavalos"], default=["padrao"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[4096])
    parser.add_argument("--processes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--strategies", nargs="+", choices=["bloco", "blocov", "intercaladov"],
                        default=["blocov", "intercaladov"])
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--repeats", type=int, default=5, help="Repeticoes por configuracao MPI.")
    parser.add_argument("--output", type=Path, default=Path("resultados/brutos/mpi.csv"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Lista configuracoes sem executar.")
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats deve ser maior que zero.")
    if args.timeout <= 0:
        parser.error("--timeout deve ser maior que zero.")
    if any(v <= 0 for v in args.resolutions + args.processes + args.chunks):
        parser.error("Resolucao, processos e chunks devem ser positivos.")

    mpi_exe = args.mpi_exe.expanduser().resolve()
    mpi_cwd = args.mpi_cwd.expanduser().resolve()
    references_dir = args.references_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    logs_dir = args.logs_dir.expanduser().resolve() if args.logs_dir else None

    if not mpi_exe.is_file():
        parser.error(f"Executavel MPI nao encontrado: {mpi_exe}")
    if not mpi_cwd.is_dir():
        parser.error(f"Diretorio MPI nao encontrado: {mpi_cwd}")
    mpirun_encontrado = shutil.which(args.mpirun)
    if mpirun_encontrado:
        caminho_launcher = Path(mpirun_encontrado)
        mpirun_command = str(caminho_launcher.resolve()) if not caminho_launcher.is_absolute() else str(caminho_launcher)
    else:
        launcher_local = Path(args.mpirun).expanduser().resolve()
        if not launcher_local.is_file():
            parser.error(f"Launcher MPI nao encontrado: {args.mpirun}")
        mpirun_command = str(launcher_local)

    configuracoes, ignoradas = montar_configuracoes(
        args.cases, args.resolutions, args.processes, args.strategies, args.chunks, args.repeats
    )
    total_execucoes = len(configuracoes)

    print("================ BENCHMARK MPI ================")
    print(f"Executavel MPI: {mpi_exe}")
    print(f"Launcher:      {mpirun_command}")
    print(f"Diretorio MPI: {mpi_cwd}")
    print(f"Referencias:   {references_dir}")
    print(f"Casos:         {args.cases}")
    print(f"Resolucao:     {args.resolutions}")
    print(f"Processos:     {args.processes}")
    print(f"Estrategias:   {args.strategies}")
    print(f"Chunks:        {args.chunks}")
    print(f"Repeticoes MPI por configuracao: {args.repeats}")
    print(f"Configuracoes MPI: {total_execucoes}")
    if ignoradas:
        print("Combinacoes ignoradas:")
        for item in ignoradas:
            print(f"  - {item}")
    print("================================================")

    if args.dry_run:
        for idx, (caso, resolucao, nproc, estrategia, chunk, repeticao) in enumerate(configuracoes, 1):
            chunk_texto = f" chunk={chunk}" if estrategia == "intercaladov" else ""
            print(f"{idx:4d}. {caso} {resolucao}x{resolucao} P={nproc} {estrategia}{chunk_texto} rep={repeticao}")
        return 0

    try:
        manifesto = carregar_manifesto(references_dir)
        referencias = {}
        for caso in args.cases:
            for resolucao in args.resolutions:
                chave = (caso, resolucao)
                if chave not in referencias:
                    referencias[chave] = obter_referencia(
                        references_dir, manifesto, caso, resolucao, verificar_hash=True
                    )
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERRO ao carregar referencias seriais: {exc}", file=sys.stderr)
        return 1

    try:
        validar_csv_existente(output)
    except ValueError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
    novo_csv = not output.exists() or output.stat().st_size == 0

    with output.open("a", newline="", encoding="utf-8") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=CSV_FIELDS)
        if novo_csv:
            writer.writeheader()

        for idx, (caso, resolucao, nproc, estrategia, chunk, repeticao) in enumerate(configuracoes, 1):
            referencia = referencias[(caso, resolucao)]
            print(
                f"\n[{idx}/{total_execucoes}] MPI {caso} {resolucao}x{resolucao} | "
                f"P={nproc} | {estrategia}"
                + (f" chunk={chunk}" if estrategia == "intercaladov" else "")
                + f" | repeticao {repeticao}/{args.repeats}"
            )
            try:
                dados, stdout, stderr, arquivo_mpi = executar_mpi(
                    mpirun_command, mpi_exe, caso, resolucao, estrategia, nproc,
                    chunk, mpi_cwd, args.timeout
                )
                if dados["max_iter"] != referencia["max_iter"]:
                    raise ValueError(
                        f"MAX_ITER MPI ({dados['max_iter']}) difere da referencia "
                        f"({referencia['max_iter']})."
                    )

                diferentes, percentual, max_diferenca = comparar_binarios(
                    referencia["binario"], arquivo_mpi
                )
                tempo_serial_mediana = referencia["tempo_serial_mediana_s"]
                dados.update({
                    "repeticao": repeticao,
                    "tempo_serial_min_s": referencia["tempo_serial_min_s"],
                    "tempo_serial_mediana_s": tempo_serial_mediana,
                    "tempo_serial_max_s": referencia["tempo_serial_max_s"],
                    "referencia_binario": referencia["arquivo_binario"],
                    "speedup_calculo": (
                        tempo_serial_mediana / dados["tempo_calculo_max_s"]
                        if dados["tempo_calculo_max_s"] > 0 else ""
                    ),
                    "speedup_paralelo_total": (
                        tempo_serial_mediana / dados["tempo_paralelo_total_s"]
                        if dados["tempo_paralelo_total_s"] > 0 else ""
                    ),
                    "pixels_diferentes": diferentes,
                    "percentual_diferenca": percentual,
                    "diferenca_maxima": max_diferenca,
                    "correto_exato": "SIM" if diferentes == 0 else "NAO",
                    "status_saida": "OK",
                })
                writer.writerow(dados)
                arquivo_csv.flush()

                print(f"  Serial (mediana): {tempo_serial_mediana:.6f} s")
                print(f"  Calculo maximo:   {dados['tempo_calculo_max_s']:.6f} s")
                print(f"  Coleta:           {dados['tempo_coleta_s']:.6f} s")
                print(f"  Reorganizacao:    {dados['tempo_reorganizacao_s']:.6f} s")
                print(f"  Total paralelo:   {dados['tempo_paralelo_total_s']:.6f} s")
                print(f"  F_LB:             {dados['f_lb']:.6f}")
                print(f"  Speedup total:    {dados['speedup_paralelo_total']:.4f}")
                print(
                    f"  Corretude exata: {dados['correto_exato']} "
                    f"({diferentes} pixels diferentes; max diferenca={max_diferenca})"
                )

                if logs_dir is not None:
                    stem = f"{caso}_{resolucao}_P{nproc}_{estrategia}_chunk{chunk}_rep{repeticao}"
                    (logs_dir / f"{stem}.out").write_text(stdout, encoding="utf-8")
                    (logs_dir / f"{stem}.err").write_text(stderr, encoding="utf-8")

            except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                print(f"Erro nesta execucao: {exc}", file=sys.stderr)
                return 1

    print(f"\nBenchmark MPI concluido. CSV: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
