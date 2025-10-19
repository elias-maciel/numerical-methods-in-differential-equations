# -*- coding: utf-8 -*-
"""
Código para simulação de EDP Parabólica - Métodos Forward Difference e Crank-Nicolson
CORRIGIDO - Problema de dimensões resolvido
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Configurar para não usar LaTeX
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Arial",
})

def solucao_exata(x, t):
    """Solução exata/manufaturada da EDP"""
    return np.exp(-t) * np.sin(np.pi/2 * x) * np.cos(np.pi/2 * x)

def termo_fonte(x, t):
    """Termo fonte g(x,t) - CORRIGIDO conforme imagem"""
    return ((np.pi**2 - 1) / 2) * np.exp(-t) * np.sin(np.pi * x)

def metodo_forward_difference(h, T=1.0):
    """
    Implementação do método Forward Difference
    """
    # Malha espacial
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)
    
    # Critério de estabilidade
    k_inicial = 0.99 * 0.5 * h**2
    Nt = max(1, int(np.ceil(T / k_inicial)))
    k = T / Nt
    r = k / h**2
    
    # Condição inicial
    U = solucao_exata(x, 0)
    
    # Loop temporal
    for n in range(Nt):
        t = n * k
        U_novo = np.zeros_like(U)
        
        # Aplicar esquema para pontos internos
        for i in range(1, Nx):
            U_novo[i] = (U[i] + r * (U[i-1] - 2*U[i] + U[i+1]) + 
                        k * termo_fonte(x[i], t))
        
        # Condições de contorno
        U_novo[0] = 0
        U_novo[Nx] = 0
        U = U_novo
    
    # Calcular erro
    u_exata = solucao_exata(x, T)
    erro = np.sqrt(h * np.sum((U[1:Nx] - u_exata[1:Nx])**2))
    
    return erro, k, Nt, x, U, u_exata

def metodo_crank_nicolson(h, k_alvo, T=1.0):
    """
    Implementação do método Crank-Nicolson
    """
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)
    Nt = max(1, int(np.ceil(T / k_alvo)))
    k = T / Nt
    r = k / (2 * h**2)
    
    # Condição inicial
    U = solucao_exata(x, 0)
    
    # Matrizes do sistema linear
    N = Nx - 1
    diagonal_principal = (1 + 2*r) * np.ones(N)
    diagonal_secundaria = -r * np.ones(N-1)
    
    # Matriz A (lado esquerdo)
    A = diags([diagonal_secundaria, diagonal_principal, diagonal_secundaria], 
              [-1, 0, 1], format='csc')
    
    for n in range(Nt):
        t = n * k
        t_meio = t + k/2
        
        # Vetor b (lado direito)
        b = np.zeros(N)
        for i in range(1, Nx):
            termo_diff = r * (U[i-1] - 2*U[i] + U[i+1])
            termo_fonte_medio = 0.5 * k * (termo_fonte(x[i], t) + termo_fonte(x[i], t_meio))
            b[i-1] = U[i] + termo_diff + termo_fonte_medio
        
        # Resolver sistema linear
        U_interno = spsolve(A, b)
        
        # Atualizar solução
        U_novo = np.zeros_like(U)
        U_novo[1:Nx] = U_interno
        U_novo[0] = 0
        U_novo[Nx] = 0
        U = U_novo
    
    # Calcular erro
    u_exata = solucao_exata(x, T)
    erro = np.sqrt(h * np.sum((U[1:Nx] - u_exata[1:Nx])**2))
    
    return erro, k, Nt, x, U, u_exata

def medir_tempo_computacional(h, k_fd, k_cn, T=1.0, repeticoes=1000):
    """
    Mede tempo computacional médio dos métodos
    """
    print(f"\nMedindo tempos para h = {h} ({repeticoes} repetições)...")
    
    # Medir tempo Forward Difference
    inicio = time.time()
    for _ in range(repeticoes):
        metodo_forward_difference(h, T)
    tempo_fd = (time.time() - inicio) / repeticoes
    
    # Medir tempo Crank-Nicolson
    inicio = time.time()
    for _ in range(repeticoes):
        metodo_crank_nicolson(h, k_cn, T)
    tempo_cn = (time.time() - inicio) / repeticoes
    
    return tempo_fd, tempo_cn

def encontrar_k_crank_nicolson(h, erro_alvo_fd, T=1.0):
    """
    Encontra o maior k para Crank-Nicolson que dá erro similar ao Forward Difference
    """
    print(f"Buscando k ótimo para Crank-Nicolson (h = {h})...")
    
    # Chutes iniciais para k
    k_min = 0.001
    k_max = 0.1
    
    melhor_k = k_min
    menor_erro = float('inf')
    
    for iteracao in range(15):  # Busca binária
        k_teste = (k_min + k_max) / 2
        erro_cn, k_efetivo, Nt, _, _, _ = metodo_crank_nicolson(h, k_teste, T)
        
        # Atualizar melhor encontrado
        if abs(erro_cn - erro_alvo_fd) < abs(menor_erro - erro_alvo_fd):
            menor_erro = erro_cn
            melhor_k = k_teste
        
        if erro_cn <= erro_alvo_fd:
            k_min = k_teste  # Pode aumentar k
        else:
            k_max = k_teste
        
        if k_max - k_min < 1e-5:
            break
    
    return melhor_k, menor_erro

def executar_simulacoes_completas():
    """
    Executa todas as simulações e gera resultados
    """
    print("=" * 70)
    print("SIMULAÇÃO DE EDP PARABÓLICA - MÉTODOS NUMÉRICOS")
    print("=" * 70)
    print("Termo fonte: g(x,t) = (π² - 1)/2 * e^(-t) * sin(πx)")
    print("Solução exata: u(x,t) = e^(-t) * sin(πx/2) * cos(πx/2)")
    print("=" * 70)
    
    # Parâmetros da simulação
    h_valores = [0.1, 0.05, 0.025]
    T = 1.0
    
    # Armazenar resultados
    resultados_fd = []
    resultados_cn = []
    tempos_computacionais = []
    
    # ITEM A: Forward Difference
    print("\n" + "="*50)
    print("ITEM A - MÉTODO FORWARD DIFFERENCE")
    print("="*50)
    
    for h in h_valores:
        print(f"\n>>> Processando h = {h}")
        erro_fd, k_fd, Nt_fd, x, U_fd, u_exata = metodo_forward_difference(h, T)
        resultados_fd.append((h, k_fd, Nt_fd, erro_fd, x, U_fd, u_exata))
        print(f"   k = {k_fd:.6f}, Nt = {Nt_fd}, Erro = {erro_fd:.2e}")
    
    # ITEM B: Crank-Nicolson
    print("\n" + "="*50)
    print("ITEM B - MÉTODO CRANK-NICOLSON")
    print("="*50)
    
    for i, h in enumerate(h_valores):
        print(f"\n>>> Processando h = {h}")
        erro_alvo_fd = resultados_fd[i][3]  # Erro do Forward Difference
        k_cn, erro_cn = encontrar_k_crank_nicolson(h, erro_alvo_fd, T)
        _, k_efetivo, Nt_cn, x, U_cn, u_exata = metodo_crank_nicolson(h, k_cn, T)
        resultados_cn.append((h, k_cn, Nt_cn, erro_cn, x, U_cn, u_exata))
        print(f"   k = {k_cn:.6f}, Nt = {Nt_cn}, Erro = {erro_cn:.2e}")
        print(f"   Ganho em k: {k_cn/resultados_fd[i][1]:.1f}x maior que FD")
    
    # ITEM C: Tempos computacionais
    print("\n" + "="*50)
    print("ITEM C - TEMPOS COMPUTACIONAIS")
    print("="*50)
    
    for i, h in enumerate(h_valores):
        k_fd = resultados_fd[i][1]
        k_cn = resultados_cn[i][1]
        tempo_fd, tempo_cn = medir_tempo_computacional(h, k_fd, k_cn, T, repeticoes=1000)
        tempos_computacionais.append((h, tempo_fd, tempo_cn))
        print(f"h = {h}: FD = {tempo_fd:.6f}s, CN = {tempo_cn:.6f}s")
        print(f"   Razão CN/FD: {tempo_cn/tempo_fd:.2f}")
    
    # Gerar gráficos e tabelas
    gerar_resultados_graficos(resultados_fd, resultados_cn, tempos_computacionais)
    
    return resultados_fd, resultados_cn, tempos_computacionais

def gerar_resultados_graficos(resultados_fd, resultados_cn, tempos):
    """
    Gera todos os gráficos e tabelas de resultados - CORRIGIDO
    """
    print("\n" + "="*50)
    print("GERANDO GRÁFICOS E TABELAS")
    print("="*50)
    
    # Extrair dados CORRETAMENTE
    h_valores = [r[0] for r in resultados_fd]
    erros_fd = [r[3] for r in resultados_fd]
    erros_cn = [r[3] for r in resultados_cn]
    k_fd = [r[1] for r in resultados_fd]
    k_cn = [r[1] for r in resultados_cn]
    Nt_fd = [r[2] for r in resultados_fd]
    Nt_cn = [r[2] for r in resultados_cn]
    tempos_fd = [t[1] for t in tempos]
    tempos_cn = [t[2] for t in tempos]
    
    # Usar dados do PRIMEIRO h para gráfico de comparação
    x_comparacao = resultados_fd[0][4]  # x para h=0.1
    U_fd_comparacao = resultados_fd[0][5]  # U_fd para h=0.1
    U_cn_comparacao = resultados_cn[0][5]  # U_cn para h=0.1
    u_exata_comparacao = resultados_fd[0][6]  # u_exata para h=0.1
    
    # GRÁFICO 1: Comparação de erros
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.loglog(h_valores, erros_fd, 'bo-', label='Forward Difference', linewidth=2, markersize=8)
    plt.loglog(h_valores, erros_cn, 'ro-', label='Crank-Nicolson', linewidth=2, markersize=8)
    
    # Linha teórica O(h²)
    if len(h_valores) >= 2:
        h_ref = np.linspace(h_valores[-1], h_valores[0], 10)
        C = erros_fd[0] / (h_valores[0]**2)
        erro_teorico = C * h_ref**2
        plt.loglog(h_ref, erro_teorico, 'k--', label='O(h²) teórico', linewidth=2)
    
    plt.xlabel('h (passo espacial)')
    plt.ylabel('Erro L²')
    plt.title('Comparação de Erros - FD vs CN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRÁFICO 2: Comparação de soluções (CORRIGIDO - usando mesmo x)
    plt.subplot(2, 3, 2)
    plt.plot(x_comparacao, u_exata_comparacao, 'k-', label='Solução Exata', linewidth=2)
    plt.plot(x_comparacao, U_fd_comparacao, 'bo', markersize=3, label='Forward Difference')
    plt.plot(x_comparacao, U_cn_comparacao, 'rx', markersize=4, label='Crank-Nicolson')
    plt.xlabel('x')
    plt.ylabel('u(x, T=1)')
    plt.title('Comparação de Soluções (h = 0.1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRÁFICO 3: Passos temporais
    plt.subplot(2, 3, 3)
    ind = np.arange(len(h_valores))
    largura = 0.35
    plt.bar(ind - largura/2, Nt_fd, largura, label='Forward Difference', alpha=0.7)
    plt.bar(ind + largura/2, Nt_cn, largura, label='Crank-Nicolson', alpha=0.7)
    plt.xlabel('Refinamento de Malha')
    plt.ylabel('Número de Passos Temporais (Nt)')
    plt.title('Passos Temporais por Método')
    plt.xticks(ind, [f'h={h}' for h in h_valores])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRÁFICO 4: Tempos computacionais
    plt.subplot(2, 3, 4)
    plt.bar(ind - largura/2, tempos_fd, largura, label='Forward Difference', alpha=0.7)
    plt.bar(ind + largura/2, tempos_cn, largura, label='Crank-Nicolson', alpha=0.7)
    plt.xlabel('Refinamento de Malha')
    plt.ylabel('Tempo Computacional (s)')
    plt.title('Tempos Computacionais Médios')
    plt.xticks(ind, [f'h={h}' for h in h_valores])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRÁFICO 5: Relação k vs h
    plt.subplot(2, 3, 5)
    plt.loglog(h_valores, k_fd, 'bo-', label='FD: k utilizado', linewidth=2, markersize=8)
    plt.loglog(h_valores, k_cn, 'ro-', label='CN: k utilizado', linewidth=2, markersize=8)
    k_limite = [0.5 * h**2 for h in h_valores]
    plt.loglog(h_valores, k_limite, 'k--', label='Limite estabilidade FD', linewidth=2)
    plt.xlabel('h')
    plt.ylabel('k')
    plt.title('Relação entre k e h')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRÁFICO 6: Eficiência computacional
    plt.subplot(2, 3, 6)
    razao_tempo = [t_cn/t_fd for t_fd, t_cn in zip(tempos_fd, tempos_cn)]
    plt.plot(h_valores, razao_tempo, 'gs-', linewidth=2, markersize=8)
    plt.axhline(y=1, color='r', linestyle='--', label='Limite igualdade')
    plt.xlabel('h')
    plt.ylabel('Tempo CN / Tempo FD')
    plt.title('Razão de Eficiência Computacional')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_completos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # TABELAS DE RESULTADOS
    print("\n" + "="*70)
    print("TABELA 1: RESULTADOS NUMÉRICOS")
    print("="*70)
    print(f"{'h':<8} {'Método':<15} {'k':<12} {'Nt':<8} {'Erro L²':<12} {'Taxa':<8}")
    print("-"*70)
    
    for i in range(len(h_valores)):
        # Forward Difference
        if i == 0:
            taxa_fd = '--'
        else:
            taxa_fd = np.log(erros_fd[i-1]/erros_fd[i]) / np.log(h_valores[i-1]/h_valores[i])
            taxa_fd = f"{taxa_fd:.2f}"
        
        print(f"{h_valores[i]:<8.4f} {'FD':<15} {k_fd[i]:<12.6f} {Nt_fd[i]:<8} {erros_fd[i]:<12.2e} {taxa_fd:<8}")
        
        # Crank-Nicolson
        if i == 0:
            taxa_cn = '--'
        else:
            taxa_cn = np.log(erros_cn[i-1]/erros_cn[i]) / np.log(h_valores[i-1]/h_valores[i])
            taxa_cn = f"{taxa_cn:.2f}"
        
        print(f"{h_valores[i]:<8.4f} {'CN':<15} {k_cn[i]:<12.6f} {Nt_cn[i]:<8} {erros_cn[i]:<12.2e} {taxa_cn:<8}")
        print("-"*70)
    
    print("\n" + "="*70)
    print("TABELA 2: TEMPOS COMPUTACIONAIS")
    print("="*70)
    print(f"{'h':<8} {'Tempo FD (s)':<15} {'Tempo CN (s)':<15} {'Razão CN/FD':<12}")
    print("-"*70)
    
    for i in range(len(h_valores)):
        razao = tempos_cn[i] / tempos_fd[i]
        print(f"{h_valores[i]:<8.4f} {tempos_fd[i]:<15.6f} {tempos_cn[i]:<15.6f} {razao:<12.2f}")

# Executar simulações
if __name__ == "__main__":
    resultados_fd, resultados_cn, tempos = executar_simulacoes_completas()
    
    print("\n" + "="*70)
    print("SIMULAÇÃO CONCLUÍDA!")
    print("="*70)
    print("Arquivos gerados:")
    print("- resultados_completos.png: Gráficos com todos os resultados")
    print("- Resultados impressos nas tabelas acima")