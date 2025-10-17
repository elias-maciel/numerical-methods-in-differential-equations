import numpy as np
import matplotlib.pyplot as plt
import time

# Configurar para NAO usar LaTeX
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Arial",
})

def exact_solution(x, t):
    return 0.5 * np.exp(-t) * np.sin(np.pi * x)

def source_term(x, t):
    return (np.pi**2 - 1)/2 * np.exp(-t) * np.sin(np.pi * x)

def forward_difference_real(h, T=1.0):
    """
    Implementacao REAL do metodo Forward Difference
    """
    # Malha espacial
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)
    
    # Criterio de estabilidade
    k = 0.99 * 0.5 * h**2
    Nt = int(T / k)
    k = T / Nt  # Ajuste para chegar exatamente em T=1
    r = k / h**2
    
    print(f"h={h:.4f}, k={k:.6f}, Nt={Nt}, r={r:.4f}")
    
    # Condicao inicial
    U = exact_solution(x, 0)
    
    # Loop temporal
    for n in range(Nt):
        t = n * k
        U_new = U.copy()
        
        # Aplicar o esquema para pontos internos
        for i in range(1, Nx):
            U_new[i] = U[i] + r * (U[i-1] - 2*U[i] + U[i+1]) + k * source_term(x[i], t)
        
        U = U_new
    
    # Calcular erro
    u_exact = exact_solution(x, T)
    error = np.sqrt(h * np.sum((U[1:Nx] - u_exact[1:Nx])**2))
    
    return error, k, Nt, x, U, u_exact

def gerar_graficos_com_simulacoes_reais():
    """Gera graficos com dados REAIS das simulacoes"""
    
    print("Executando simulacoes REAIS...")
    
    # Valores de h para testar
    h_values = [0.1, 0.05, 0.025]
    errors = []
    k_values = []
    all_results = []
    
    # Executar simulacoes REAIS
    for h in h_values:
        print(f"\nSimulando h = {h}...")
        start_time = time.time()
        
        error, k, Nt, x, U, u_exact = forward_difference_real(h)
        errors.append(error)
        k_values.append(k)
        all_results.append((x, U, u_exact))
        
        elapsed = time.time() - start_time
        print(f"Erro: {error:.2e}, Tempo: {elapsed:.2f}s")
    
    print("\nGerando graficos com dados REAIS...")
    
    # GRAFICO 1: Convergencia do erro
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.loglog(h_values, errors, 'bo-', linewidth=2, markersize=8, 
               label='Erro numerico REAL')
    
    # Linha teorica O(h²) para comparacao
    if len(h_values) >= 2:
        h_ref = np.linspace(h_values[-1], h_values[0], 10)
        C = errors[0] / (h_values[0]**2)
        error_theoretical = C * h_ref**2
        plt.loglog(h_ref, error_theoretical, 'r--', linewidth=2, 
                   label='O(h²) teorico')
    
    plt.xlabel('h (passo espacial)')
    plt.ylabel('Erro L²')
    plt.title('Convergencia - Forward Difference (DADOS REAIS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRAFICO 2: Solucao numerica vs exata
    plt.subplot(1, 3, 2)
    x, U, u_exact = all_results[0]
    
    plt.plot(x, u_exact, 'r-', linewidth=2, label='Solucao Exata')
    plt.plot(x, U, 'bo', markersize=3, label='Solucao Numerica REAL')
    
    error_max = np.max(np.abs(U - u_exact))
    plt.text(0.05, 0.15, f'Erro max: {error_max:.1e}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel('x')
    plt.ylabel('u(x, T=1)')
    plt.title(f'Solucoes - h = {h_values[0]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GRAFICO 3: Relacao k vs h
    plt.subplot(1, 3, 3)
    plt.loglog(h_values, k_values, 'go-', linewidth=2, markersize=8, 
               label='k utilizado REAL')
    
    k_limite = [0.5 * h**2 for h in h_values]
    plt.loglog(h_values, k_limite, 'r--', linewidth=2, 
               label='Limite estabilidade')
    
    plt.xlabel('h')
    plt.ylabel('k')
    plt.title('Relacao k vs h')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    gerar_graficos_com_simulacoes_reais()
