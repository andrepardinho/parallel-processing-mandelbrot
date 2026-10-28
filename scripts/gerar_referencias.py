#!/usr/bin/env python3
"""Gera uma referencia serial persistente para cada caso e resolucao.

Exemplo, executado a partir da raiz do projeto:
    python3 gerar_referencias.py \
        --serial-exe ./serial/mandelbrot_serial \
        --serial-cwd ./serial \
        --cases padrao cavalos \
        --resolutions 4096 8192 \
        --repeats 5

As referencias sao guardadas em referencias/serial.csv e referencias/binarios/.
Referencias validas ja existentes sao reutilizadas. Use --force para recalcula-las.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import statistics
import sys
from datetime import datetime
from pathlib import Path

from referencias_mandelbrot import (
    MANIFEST_FIELDS,
    MANIFEST_NAME,
    carregar_manifesto,
    comparar_binarios,
    nome_arquivo_binario,
    obter_referencia,
    sha256_arquivo,
)

RE_TEMPO_CALC = re.compile(r"Tempo de calculo:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
RE_MAX_ITER = re.compile(r"MAX_ITER\s*[:=]\s*(\d+)")


def escrever_manifesto(diretorio_referencias: Path, linhas: dict[tuple[str, int], dict]) -> None:
    caminho = diretorio_referencias / MANIFEST_NAME
    temporario = caminho.with_suffix(".csv.tmp")
    with temporario.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=MANIFEST_FIELDS)
        escritor.writeheader()
        for chave in sorted(linhas):
            escritor.writerow(linhas[chave])
    temporario.replace(caminho)


def ler_manifesto_existente(diretorio_referencias: Path) -> dict[tuple[str, int], dict]:
    caminho = diretorio_referencias / MANIFEST_NAME
    if not caminho.exists():
        return {}
    return carregar_manifesto(diretorio_referencias)


def referencia_existente_valida(diretorio_referencias: Path, manifesto: dict, caso: str, resolucao: int) -> bool:
    try:
        obter_referencia(diretorio_referencias, manifesto, caso, resolucao, verificar_hash=True)
        return True
    except (KeyError, OSError, ValueError, TypeError):
        return False


def executar_serial(
    executavel: Path,
    caso: str,
    resolucao: int,
    cwd: Path,
    timeout: float,
) -> tuple[float, int, str, str, Path]:
    nome_binario = nome_arquivo_binario(caso, resolucao)
    arquivo_saida = cwd / "saida" / nome_binario
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    arquivo_saida.unlink(missing_ok=True)

    comando = [str(executavel), caso, str(resolucao)]
    processo = subprocess.run(
        comando,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if processo.returncode != 0:
        raise RuntimeError(
            "A execucao serial terminou com erro.\n"
            f"Comando: {' '.join(comando)}\n"
            f"Codigo: {processo.returncode}\n"
            f"STDOUT:\n{processo.stdout}\nSTDERR:\n{processo.stderr}"
        )

    tempo_match = RE_TEMPO_CALC.search(processo.stdout)
    iter_match = RE_MAX_ITER.search(processo.stdout)
    if not tempo_match:
        raise ValueError(
            "Nao foi possivel ler 'Tempo de calculo' da saida serial.\n"
            f"STDOUT:\n{processo.stdout}"
        )
    if not iter_match:
        raise ValueError(
            "Nao foi possivel ler 'MAX_ITER' da saida serial.\n"
            f"STDOUT:\n{processo.stdout}"
        )
    if not arquivo_saida.is_file():
        raise FileNotFoundError(f"O serial nao gerou o binario esperado: {arquivo_saida}")

    tamanho_esperado = resolucao * resolucao * 4
    if arquivo_saida.stat().st_size != tamanho_esperado:
        raise ValueError(
            f"Binario serial com tamanho inesperado: {arquivo_saida.stat().st_size} bytes; "
            f"esperado {tamanho_esperado}."
        )

    return float(tempo_match.group(1)), int(iter_match.group(1)), processo.stdout, processo.stderr, arquivo_saida


def gerar_uma_referencia(
    executavel: Path,
    cwd: Path,
    diretorio_referencias: Path,
    caso: str,
    resolucao: int,
    repeticoes: int,
    timeout: float,
) -> dict:
    diretorio_binarios = diretorio_referencias / "binarios"
    diretorio_binarios.mkdir(parents=True, exist_ok=True)

    nome_binario = nome_arquivo_binario(caso, resolucao)
    destino_final = diretorio_binarios / nome_binario
    destino_temporario = diretorio_binarios / f".{nome_binario}.tmp"
    destino_temporario.unlink(missing_ok=True)

    tempos = []
    max_iter_referencia = None
    try:
        for repeticao in range(1, repeticoes + 1):
            print(f"  Serial {caso} {resolucao}x{resolucao}: repeticao {repeticao}/{repeticoes}")
            tempo, max_iter, stdout, stderr, arquivo_serial = executar_serial(
                executavel, caso, resolucao, cwd, timeout
            )
            if max_iter_referencia is None:
                max_iter_referencia = max_iter
                shutil.copy2(arquivo_serial, destino_temporario)
            else:
                if max_iter != max_iter_referencia:
                    raise ValueError(
                        f"MAX_ITER mudou entre repeticoes: {max_iter_referencia} e {max_iter}."
                    )
                diferentes, _, max_diferenca = comparar_binarios(destino_temporario, arquivo_serial)
                if diferentes != 0:
                    raise ValueError(
                        "A versao serial produziu resultados diferentes entre repeticoes: "
                        f"{diferentes} pixels diferentes, diferenca maxima={max_diferenca}. "
                        "A referencia nao foi atualizada."
                    )
            tempos.append(tempo)

        destino_temporario.replace(destino_final)
        data_geracao = datetime.now().astimezone().isoformat(timespec="seconds")
        return {
            "caso": caso,
            "resolucao": resolucao,
            "max_iter": max_iter_referencia,
            "repeticoes": repeticoes,
            "tempo_serial_min_s": f"{min(tempos):.10f}",
            "tempo_serial_mediana_s": f"{statistics.median(tempos):.10f}",
            "tempo_serial_max_s": f"{max(tempos):.10f}",
            "tempos_serial_s": ";".join(f"{t:.10f}" for t in tempos),
            "arquivo_binario": f"binarios/{nome_binario}",
            "sha256_binario": sha256_arquivo(destino_final),
            "data_geracao": data_geracao,
        }
    finally:
        destino_temporario.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera e guarda referencias seriais do Mandelbrot.")
    parser.add_argument("--serial-exe", type=Path, default=Path("./serial/mandelbrot_serial"))
    parser.add_argument("--serial-cwd", type=Path, default=Path("./serial"))
    parser.add_argument("--references-dir", type=Path, default=Path("./referencias"))
    parser.add_argument("--cases", nargs="+", choices=["padrao", "cavalos"], default=["padrao"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[4096])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--force", action="store_true", help="Recalcula mesmo referencias validas existentes.")
    args = parser.parse_args()

    if args.repeats <= 0 or args.timeout <= 0:
        parser.error("--repeats e --timeout devem ser positivos.")
    if any(resolucao <= 0 for resolucao in args.resolutions):
        parser.error("Todas as resolucoes devem ser positivas.")

    executavel = args.serial_exe.expanduser().resolve()
    cwd = args.serial_cwd.expanduser().resolve()
    diretorio_referencias = args.references_dir.expanduser().resolve()
    if not executavel.is_file():
        parser.error(f"Executavel serial nao encontrado: {executavel}")
    if not cwd.is_dir():
        parser.error(f"Diretorio de trabalho serial nao encontrado: {cwd}")

    diretorio_referencias.mkdir(parents=True, exist_ok=True)
    manifesto = ler_manifesto_existente(diretorio_referencias)

    print("=========== GERADOR DE REFERENCIAS SERIAIS ===========")
    print(f"Executavel serial: {executavel}")
    print(f"Diretorio serial:  {cwd}")
    print(f"Referencias:      {diretorio_referencias}")
    print(f"Casos:             {args.cases}")
    print(f"Resolucao:         {args.resolutions}")
    print(f"Repeticoes:        {args.repeats}")
    print(f"Forcar regeneracao: {args.force}")
    print("======================================================")

    for caso in args.cases:
        for resolucao in args.resolutions:
            chave = (caso, resolucao)
            if not args.force and referencia_existente_valida(
                diretorio_referencias, manifesto, caso, resolucao
            ):
                print(f"\nReferencia ja valida; reutilizando: {caso} {resolucao}x{resolucao}")
                continue

            print(f"\nGerando referencia: {caso} {resolucao}x{resolucao}")
            try:
                linha = gerar_uma_referencia(
                    executavel, cwd, diretorio_referencias, caso, resolucao, args.repeats, args.timeout
                )
                manifesto[chave] = linha
                escrever_manifesto(diretorio_referencias, manifesto)
                print(f"  MAX_ITER: {linha['max_iter']}")
                print(f"  Tempo serial mediano: {float(linha['tempo_serial_mediana_s']):.6f} s")
                print(f"  Binario: {diretorio_referencias / linha['arquivo_binario']}")
                print(f"  SHA-256: {linha['sha256_binario']}")
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                print(f"ERRO ao gerar referencia para {caso} {resolucao}: {exc}", file=sys.stderr)
                return 1

    print(f"\nReferencias prontas. Manifesto: {diretorio_referencias / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
