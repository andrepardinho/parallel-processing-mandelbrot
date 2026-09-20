#!/usr/bin/env python3
"""
Compara dois arquivos binários de Mandelbrot gravados como int32_t em
ordem row-major.

Uso:
    python3 validar_bin.py arquivo_serial.bin arquivo_omp.bin

Critério de corretude:
- igualdade exata, ou
- no máximo 0,01% dos pixels diferentes;
- cada pixel diferente pode variar no máximo 1 iteração.

Código de saída:
    0 -> correto (igualdade exata ou dentro da tolerância)
    1 -> incorreto
    2 -> erro de uso/arquivo/formato
"""

import argparse
import math
import struct
import sys
from pathlib import Path


INT32_SIZE = 4
DEFAULT_CHUNK_INTS = 1_000_000


def comparar_arquivos(
    arquivo_a: Path,
    arquivo_b: Path,
    max_percent: float,
    max_diff: int,
    chunk_ints: int,
):
    if not arquivo_a.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo_a}")

    if not arquivo_b.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {arquivo_b}")

    tamanho_a = arquivo_a.stat().st_size
    tamanho_b = arquivo_b.stat().st_size

    if tamanho_a % INT32_SIZE != 0:
        raise ValueError(
            f"'{arquivo_a}' nao possui tamanho multiplo de {INT32_SIZE} bytes."
        )

    if tamanho_b % INT32_SIZE != 0:
        raise ValueError(
            f"'{arquivo_b}' nao possui tamanho multiplo de {INT32_SIZE} bytes."
        )

    if tamanho_a != tamanho_b:
        print("RESULTADO: REPROVADO")
        print(f"Tamanho arquivo 1: {tamanho_a} bytes")
        print(f"Tamanho arquivo 2: {tamanho_b} bytes")
        print("Os arquivos possuem tamanhos diferentes.")
        return 1

    total_pixels = tamanho_a // INT32_SIZE
    max_diferencas_permitidas = math.floor(
        total_pixels * (max_percent / 100.0)
    )

    diferentes = 0
    diferentes_acima_do_limite = 0
    max_diferenca_observada = 0
    primeira_diferenca = None

    # int32_t e gravado em ordem nativa pela fwrite(). Os dois arquivos
    # devem ter sido produzidos no mesmo ambiente para esta comparação.
    if struct.calcsize("i") != INT32_SIZE:
        raise RuntimeError(
            "A plataforma Python nao possui int com 4 bytes; "
            "nao foi possivel interpretar int32_t."
        )

    item_size = INT32_SIZE
    bloco_bytes = chunk_ints * item_size

    with arquivo_a.open("rb") as fa, arquivo_b.open("rb") as fb:
        indice_global = 0

        while True:
            dados_a = fa.read(bloco_bytes)
            dados_b = fb.read(bloco_bytes)

            if not dados_a and not dados_b:
                break

            if len(dados_a) != len(dados_b):
                raise RuntimeError(
                    "Os arquivos mudaram de tamanho durante a leitura."
                )

            # memoryview.cast('i') interpreta os bytes como inteiros
            # nativos de 4 bytes sem criar uma grande lista Python.
            valores_a = memoryview(dados_a).cast("i")
            valores_b = memoryview(dados_b).cast("i")

            for offset, (valor_a, valor_b) in enumerate(
                zip(valores_a, valores_b)
            ):
                if valor_a != valor_b:
                    diferenca = abs(valor_a - valor_b)
                    diferentes += 1

                    if diferenca > max_diferenca_observada:
                        max_diferenca_observada = diferenca

                    if diferenca > max_diff:
                        diferentes_acima_do_limite += 1

                    if primeira_diferenca is None:
                        primeira_diferenca = (
                            indice_global + offset,
                            valor_a,
                            valor_b,
                        )

            indice_global += len(valores_a)

            valores_a.release()
            valores_b.release()

    percentual_diferente = (
        (diferentes / total_pixels) * 100.0 if total_pixels else 0.0
    )

    dentro_da_tolerancia = (
        diferentes <= max_diferencas_permitidas
        and diferentes_acima_do_limite == 0
    )

    print()
    print("========================================")
    print("        VALIDACAO DE CORRETUDE")
    print("========================================")
    print(f"Arquivo 1:               {arquivo_a}")
    print(f"Arquivo 2:               {arquivo_b}")
    print(f"Pixels comparados:       {total_pixels:,}")
    print(f"Pixels diferentes:       {diferentes:,}")
    print(f"Percentual diferente:    {percentual_diferente:.8f}%")
    print(
        f"Max. diferencas permitidas: "
        f"{max_diferencas_permitidas:,} ({max_percent:.4f}%)"
    )
    print(f"Maior diferenca:         {max_diferenca_observada}")
    print(f"Diferencas > {max_diff}:        {diferentes_acima_do_limite:,}")

    if primeira_diferenca is not None:
        indice, valor_a, valor_b = primeira_diferenca
        print(
            f"Primeira diferenca:      indice {indice} "
            f"(arquivo 1 = {valor_a}, arquivo 2 = {valor_b})"
        )

    if diferentes == 0:
        print("----------------------------------------")
        print("RESULTADO: CORRETO")
        print("As matrizes sao identicas.")
        print("========================================")
        return 0

    if dentro_da_tolerancia:
        print("----------------------------------------")
        print("RESULTADO: CORRETO DENTRO DA TOLERANCIA")
        print(
            "As diferencas observadas estao dentro do criterio "
            "de 0,01% dos pixels e no maximo 1 iteracao por pixel."
        )
        print("========================================")
        return 0

    print("----------------------------------------")
    print("RESULTADO: REPROVADO")
    print("As diferencas ultrapassam o criterio de corretude.")
    print("========================================")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Compara matrizes Mandelbrot em arquivos binarios int32_t."
    )

    parser.add_argument(
        "arquivo_1",
        type=Path,
        help="Arquivo binario de referencia (serial).",
    )

    parser.add_argument(
        "arquivo_2",
        type=Path,
        help="Arquivo binario a ser validado (OpenMP).",
    )

    parser.add_argument(
        "--max-percent",
        type=float,
        default=0.01,
        help="Percentual maximo de pixels diferentes (padrao: 0.01).",
    )

    parser.add_argument(
        "--max-diff",
        type=int,
        default=1,
        help="Maior diferenca de iteracoes permitida por pixel (padrao: 1).",
    )

    parser.add_argument(
        "--chunk-ints",
        type=int,
        default=DEFAULT_CHUNK_INTS,
        help=(
            "Quantidade de int32_t lidos por bloco "
            f"(padrao: {DEFAULT_CHUNK_INTS})."
        ),
    )

    args = parser.parse_args()

    if args.max_percent < 0:
        parser.error("--max-percent deve ser >= 0.")

    if args.max_diff < 0:
        parser.error("--max-diff deve ser >= 0.")

    if args.chunk_ints <= 0:
        parser.error("--chunk-ints deve ser > 0.")

    try:
        return comparar_arquivos(
            args.arquivo_1,
            args.arquivo_2,
            args.max_percent,
            args.max_diff,
            args.chunk_ints,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
