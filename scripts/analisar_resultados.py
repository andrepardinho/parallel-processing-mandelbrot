#!/usr/bin/env python3
"""
Processa os resultados brutos (CSV) do benchmark OpenMP de Mandelbrot.
Gera graficos academicos de Speedup, Eficiencia, F_LB e Weak Scaling,
alem de comparativos agrupados para politicas de escalonamento.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuração de estilo visual acadêmico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.autolayout': True
})

PASTA_BRUTOS = Path("resultados/brutos")
PASTA_PROCESSADOS = Path("resultados/processados")
PASTA_GRAFICOS = Path("resultados/graficos")

def garantir_pastas():
    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

def carregar_csv(nome_arquivo):
    caminho = PASTA_BRUTOS / nome_arquivo
    if not caminho.is_file():
        print(f"  [Aviso] Arquivo nao encontrado, ignorando: {caminho}")
        return []
    with open(caminho, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def agrupar_por_chave(dados, chave_x):
    agrupado = {}
    for linha in dados:
        try:
            x = float(linha[chave_x]) if linha[chave_x].replace('.','',1).isdigit() else linha[chave_x]
            tempo = float(linha["tempo_calc_s"])
            flb = float(linha["f_lb"])
            if x not in agrupado:
                agrupado[x] = {"tempos": [], "flb": []}
            agrupado[x]["tempos"].append(tempo)
            agrupado[x]["flb"].append(flb)
        except (ValueError, KeyError):
            continue
    return agrupado

def analisar_strong_scaling():
    print("\nAnalisando Strong Scaling...")
    dados = carregar_csv("strong_scaling.csv")
    if not dados: return

    agrupado = agrupar_por_chave(dados, "threads_utilizadas")
    threads = sorted([int(k) for k in agrupado.keys()])
    
    t_medios = [np.mean(agrupado[t]["tempos"]) for t in threads]
    flb_medios = [np.mean(agrupado[t]["flb"]) for t in threads]
    
    t_base = t_medios[0] # Tempo com 1 thread
    speedups = [t_base / t for t in t_medios]
    eficiencias = [(s / t) * 100 for s, t in zip(speedups, threads)]

    # 1. Gráfico de Speedup
    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedups, marker='o', linewidth=2.5, label='Speedup Real', color='#1f77b4')
    plt.plot(threads, threads, linestyle='--', color='gray', label='Speedup Ideal (Linear)')
    plt.title('Strong Scaling: Speedup (Caso Padrão)')
    plt.xlabel('Número de Threads')
    plt.ylabel('Speedup S(p)')
    plt.xticks(threads)
    plt.legend(loc='upper left')
    plt.savefig(PASTA_GRAFICOS / "1_strong_scaling_speedup.png", dpi=300)
    plt.close()

    # 2. Gráfico de Eficiência
    plt.figure(figsize=(8, 5))
    plt.plot(threads, eficiencias, marker='s', linewidth=2.5, color='#2ca02c', label='Eficiência Paralela')
    plt.axhline(y=100, linestyle='--', color='gray', alpha=0.7, label='100% Ideal')
    plt.title('Strong Scaling: Eficiência Paralela')
    plt.xlabel('Número de Threads')
    plt.ylabel('Eficiência (%)')
    plt.xticks(threads)
    plt.ylim(0, max(max(eficiencias) + 10, 110))
    plt.legend(loc='upper right')
    plt.savefig(PASTA_GRAFICOS / "2_strong_scaling_eficiencia.png", dpi=300)
    plt.close()

    # 3. Gráfico de F_LB (Desbalanceamento)
    plt.figure(figsize=(8, 5))
    plt.plot(threads, flb_medios, marker='^', linewidth=2.5, color='#d62728', label='Fator de Carga ($F_{LB}$)')
    plt.axhline(y=0.0, linestyle='--', color='gray', alpha=0.7, label='Balanço Perfeito ($F_{LB} = 0.0$)')
    plt.title('Strong Scaling: Fator de Balanceamento de Carga ($F_{LB}$)')
    plt.xlabel('Número de Threads')
    plt.ylabel('Fator $F_{LB}$')
    plt.xticks(threads)
    plt.legend(loc='upper left')
    plt.savefig(PASTA_GRAFICOS / "3_strong_scaling_flb.png", dpi=300)
    plt.close()

def analisar_weak_scaling():
    print("Analisando Weak Scaling...")
    dados = carregar_csv("weak_scaling.csv")
    if not dados: return

    agrupado = agrupar_por_chave(dados, "threads_utilizadas")
    threads = sorted([int(k) for k in agrupado.keys()])
    t_medios = [np.mean(agrupado[t]["tempos"]) for t in threads]
    
    t_base = t_medios[0]
    eficiencias_fracas = [(t_base / t) * 100 for t in t_medios]

    # 4. Tempo de Execução vs Threads (Idealmente constante)
    plt.figure(figsize=(8, 5))
    plt.plot(threads, t_medios, marker='o', linewidth=2.5, color='#9467bd', label='Tempo Real (s)')
    plt.axhline(y=t_base, linestyle='--', color='gray', alpha=0.7, label='Tempo Ideal Constante')
    plt.title('Weak Scaling: Tempo de Cálculo (Carga Aumenta com Threads)')
    plt.xlabel('Número de Threads (e Resolução Proporcional)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.xticks(threads)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / "4_weak_scaling_tempo.png", dpi=300)
    plt.close()

def analisar_chunks_e_schedules():
    print("Analisando Schedules e Chunks (Caso Cavalos-Marinhos)...")
    dados = carregar_csv("chunks.csv")
    if not dados: return

    # Extrair schedules únicos formatados e chunks
    schedules = {}
    for linha in dados:
        sched = linha["schedule_utilizado"].strip().lower()
        if not sched or sched == "desconhecido": sched = "static"
        chunk = int(linha["chunk_utilizado"])
        tempo = float(linha["tempo_calc_s"])
        flb = float(linha["f_lb"])

        if sched not in schedules:
            schedules[sched] = {}
        if chunk not in schedules[sched]:
            schedules[sched][chunk] = {"tempos": [], "flb": []}
        schedules[sched][chunk]["tempos"].append(tempo)
        schedules[sched][chunk]["flb"].append(flb)

    cores = {'static': '#1f77b4', 'dynamic': '#ff7f0e', 'guided': '#2ca02c'}
    marcadores = {'static': 'o', 'dynamic': 's', 'guided': '^'}

    # 5. Gráfico de Tempo por Schedule/Chunk (Curvas)
    plt.figure(figsize=(9, 6))
    for sched, chunks_dict in schedules.items():
        chunks_ord = sorted(chunks_dict.keys())
        t_medios = [np.mean(chunks_dict[c]["tempos"]) for c in chunks_ord]
        
        plt.plot([str(c) for c in chunks_ord], t_medios, 
                 marker=marcadores.get(sched, 'x'), 
                 linewidth=2.5, color=cores.get(sched, 'black'), 
                 label=f'{sched.capitalize()}')

    plt.title('Tempo de Cálculo no Desbalanceamento Severo (Cavalos)')
    plt.xlabel('Tamanho do Chunk (Lote de Iterações)')
    plt.ylabel('Tempo de Cálculo (s)')
    plt.legend(title="Política (Schedule)")
    plt.savefig(PASTA_GRAFICOS / "5_chunks_tempo_comparativo.png", dpi=300)
    plt.close()

    # 6. Gráfico F_LB por Schedule/Chunk (Curvas)
    plt.figure(figsize=(9, 6))
    for sched, chunks_dict in schedules.items():
        chunks_ord = sorted(chunks_dict.keys())
        flb_medios = [np.mean(chunks_dict[c]["flb"]) for c in chunks_ord]
        
        plt.plot([str(c) for c in chunks_ord], flb_medios, 
                 marker=marcadores.get(sched, 'x'), 
                 linewidth=2.5, color=cores.get(sched, 'black'), 
                 label=f'{sched.capitalize()}')

    plt.axhline(y=0.0, linestyle='--', color='gray', alpha=0.7, label='Balanço Perfeito')
    plt.title('Fator $F_{LB}$ no Desbalanceamento Severo (Cavalos)')
    plt.xlabel('Tamanho do Chunk (Lote de Iterações)')
    plt.ylabel('Fator de Balanceamento ($F_{LB}$)')
    plt.legend(title="Política (Schedule)")
    plt.savefig(PASTA_GRAFICOS / "6_chunks_flb_comparativo.png", dpi=300)
    plt.close()

def main():
    print("========================================")
    print(" INICIANDO GERAÇÃO DOS GRÁFICOS AVANÇADOS")
    print("========================================")
    garantir_pastas()
    analisar_strong_scaling()
    analisar_weak_scaling()
    analisar_chunks_e_schedules()
    print("\nAnálise concluída com sucesso!")
    print(f"Os gráficos de alta resolução foram salvos em: {PASTA_GRAFICOS}/")

if __name__ == "__main__":
    main()