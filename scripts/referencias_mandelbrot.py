#!/usr/bin/env python3
"""Utilitarios compartilhados para as referencias seriais do Mandelbrot."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

MANIFEST_NAME = "serial.csv"

MANIFEST_FIELDS = [
    "caso",
    "resolucao",
    "max_iter",
    "repeticoes",
    "tempo_serial_min_s",
    "tempo_serial_mediana_s",
    "tempo_serial_max_s",
    "tempos_serial_s",
    "arquivo_binario",
    "sha256_binario",
    "data_geracao",
]


def nome_arquivo_binario(caso: str, resolucao: int) -> str:
    if caso == "padrao":
        prefixo = "vista_completa"
    elif caso == "cavalos":
        prefixo = "cavalos_marinhos"
    else:
        raise ValueError(f"Caso desconhecido: {caso}")
    return f"mandelbrot_{prefixo}_{resolucao}.bin"


def sha256_arquivo(caminho: Path, bloco_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while True:
            bloco = arquivo.read(bloco_bytes)
            if not bloco:
                break
            digest.update(bloco)
    return digest.hexdigest()


def carregar_manifesto(diretorio_referencias: Path) -> dict[tuple[str, int], dict]:
    """Lê serial.csv e retorna os registros indexados por (caso, resolucao)."""
    caminho = diretorio_referencias / MANIFEST_NAME
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Manifesto de referencias nao encontrado: {caminho}. "
            "Execute primeiro gerar_referencias.py."
        )

    resultado = {}
    with caminho.open("r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.DictReader(arquivo)
        campos = set(leitor.fieldnames or [])
        faltantes = set(MANIFEST_FIELDS) - campos
        if faltantes:
            raise ValueError(
                f"Manifesto {caminho} sem as colunas: {', '.join(sorted(faltantes))}"
            )

        for numero, linha in enumerate(leitor, start=2):
            try:
                chave = (linha["caso"], int(linha["resolucao"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Linha {numero} invalida em {caminho}: {exc}") from exc
            if chave in resultado:
                raise ValueError(
                    f"Referencia duplicada para caso={chave[0]}, resolucao={chave[1]}"
                )
            resultado[chave] = linha
    return resultado


def obter_referencia(
    diretorio_referencias: Path,
    manifesto: dict[tuple[str, int], dict],
    caso: str,
    resolucao: int,
    *,
    verificar_hash: bool = True,
) -> dict:
    """Valida e retorna os dados de uma referencia serial do manifesto."""
    chave = (caso, resolucao)
    if chave not in manifesto:
        raise KeyError(
            f"Nao existe referencia para {caso}, {resolucao}x{resolucao} em "
            f"{diretorio_referencias / MANIFEST_NAME}. "
            "Gere-a com gerar_referencias.py."
        )

    linha = manifesto[chave]
    caminho_relativo = Path(linha["arquivo_binario"])
    if caminho_relativo.is_absolute() or ".." in caminho_relativo.parts:
        raise ValueError(
            f"Caminho de referencia invalido no manifesto: {linha['arquivo_binario']}"
        )
    binario = diretorio_referencias / caminho_relativo
    if not binario.is_file():
        raise FileNotFoundError(f"Binario de referencia nao encontrado: {binario}")

    tamanho_esperado = resolucao * resolucao * 4  # int32_t por pixel
    tamanho_real = binario.stat().st_size
    if tamanho_real != tamanho_esperado:
        raise ValueError(
            f"Tamanho incorreto do binario de referencia {binario}: "
            f"{tamanho_real} bytes; esperado {tamanho_esperado}."
        )

    hash_esperado = linha["sha256_binario"].strip()
    if verificar_hash:
        hash_real = sha256_arquivo(binario)
        if not hash_esperado or hash_real != hash_esperado:
            raise ValueError(
                f"SHA-256 da referencia nao confere: {binario}. "
                "Gere novamente a referencia com --force."
            )

    return {
        "caso": caso,
        "resolucao": resolucao,
        "max_iter": int(linha["max_iter"]),
        "repeticoes": int(linha["repeticoes"]),
        "tempo_serial_min_s": float(linha["tempo_serial_min_s"]),
        "tempo_serial_mediana_s": float(linha["tempo_serial_mediana_s"]),
        "tempo_serial_max_s": float(linha["tempo_serial_max_s"]),
        "arquivo_binario": linha["arquivo_binario"],
        "binario": binario,
        "sha256_binario": hash_esperado,
        "data_geracao": linha["data_geracao"],
    }


def comparar_binarios(
    arquivo_referencia: Path,
    arquivo_teste: Path,
    elementos_por_bloco: int = 250_000,
) -> tuple[int, float, int]:
    """Compara todos os int32 dos binarios e retorna (diferentes, %, max_diferenca)."""
    if not arquivo_referencia.is_file():
        raise FileNotFoundError(f"Binario de referencia nao encontrado: {arquivo_referencia}")
    if not arquivo_teste.is_file():
        raise FileNotFoundError(f"Binario de teste nao encontrado: {arquivo_teste}")

    tamanho_ref = arquivo_referencia.stat().st_size
    tamanho_teste = arquivo_teste.stat().st_size
    if tamanho_ref != tamanho_teste:
        raise ValueError(
            f"Tamanhos de binario diferentes: referencia={tamanho_ref}, "
            f"teste={tamanho_teste} bytes."
        )
    if tamanho_ref % 4 != 0:
        raise ValueError("O tamanho dos binarios nao e multiplo de 4 bytes (int32_t).")
    if elementos_por_bloco <= 0:
        raise ValueError("elementos_por_bloco deve ser positivo.")

    total_pixels = tamanho_ref // 4
    diferentes = 0
    max_diferenca = 0
    bytes_por_bloco = elementos_por_bloco * 4

    with arquivo_referencia.open("rb") as fa, arquivo_teste.open("rb") as fb:
        while True:
            dados_a = fa.read(bytes_por_bloco)
            dados_b = fb.read(bytes_por_bloco)
            if not dados_a and not dados_b:
                break
            if len(dados_a) != len(dados_b):
                raise ValueError("Os binarios terminaram em posicoes diferentes.")

            valores_a = memoryview(dados_a).cast("i")
            valores_b = memoryview(dados_b).cast("i")
            try:
                for valor_a, valor_b in zip(valores_a, valores_b):
                    if valor_a != valor_b:
                        diferentes += 1
                        diferenca = abs(valor_a - valor_b)
                        if diferenca > max_diferenca:
                            max_diferenca = diferenca
            finally:
                valores_a.release()
                valores_b.release()

    percentual = (diferentes / total_pixels * 100.0) if total_pixels else 0.0
    return diferentes, percentual, max_diferenca
