import numpy as np
import matplotlib.pyplot as plt
import time

def exact_solution(x, t):
    return 0.5 * np.exp(-t) * np.sin(np.pi * x)

def source_term(x, t):
    return (np.pi**2 - 1) * 0.5 * np.exp(-t) * np.sin(np.pi * x)

def forward_difference(h, T=1.0):
    """
    Implementacao do metodo Forward Difference (Explicito)
    """
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)

    # Criterio de estabilidade para o metodo explicito
    k_max = 0.5 * h**2
    k = 0.9 * k_max # Usar um k ligeiramente menor para garantir estabilidade
    Nt = int(T / k)
    k = T / Nt  # Ajuste para chegar exatamente em T=1
    r = k / h**2

    print(f"h={h:.4f}, k={k:.6f}, Nt={Nt}, r={r:.4f}")

    U = exact_solution(x, 0)

    for n in range(Nt):
        t = n * k
        U_new = np.zeros_like(U)
        U_new[0] = 0.0  # Condicao de contorno de Dirichlet
        U_new[Nx] = 0.0 # Condicao de contorno de Dirichlet

        for i in range(1, Nx):
            U_new[i] = U[i] + r * (U[i-1] - 2*U[i] + U[i+1]) + k * source_term(x[i], t)
        U = U_new

    u_exact = exact_solution(x, T)
    error = np.sqrt(h * np.sum((U[1:Nx] - u_exact[1:Nx])**2))

    return error, k, Nt, x, U, u_exact

def backward_difference(h, T=1.0):
    """
    Implementacao do metodo Backward Difference (Implicito)
    """
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)

    # Para consistencia com Crank-Nicolson, usamos o mesmo k
    k = 0.5 * h**2 # Pode ser maior, mas mantemos para comparacao
    Nt = int(T / k)
    k = T / Nt  # Ajuste para chegar exatamente em T=1
    r = k / h**2

    print(f"h={h:.4f}, k={k:.6f}, Nt={Nt}, r={r:.4f}")

    U = exact_solution(x, 0)

    # Montagem da matriz tridiagonal (constante no tempo)
    main_diag = (1 + 2 * r) * np.ones(Nx - 1)
    off_diag = -r * np.ones(Nx - 2)
    
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    for n in range(Nt):
        t = (n + 1) * k # Tempo para o lado direito
        b = U[1:Nx] + k * source_term(x[1:Nx], t)
        
        # Resolver o sistema linear A * U_new[1:Nx] = b
        U_new_internal = np.linalg.solve(A, b)
        
        U_new = np.zeros_like(U)
        U_new[0] = 0.0
        U_new[Nx] = 0.0
        U_new[1:Nx] = U_new_internal
        U = U_new

    u_exact = exact_solution(x, T)
    error = np.sqrt(h * np.sum((U[1:Nx] - u_exact[1:Nx])**2))

    return error, k, Nt, x, U, u_exact

def crank_nicolson(h, T=1.0):
    """
    Implementacao do metodo de Crank-Nicolson
    """
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)

    # Crank-Nicolson e incondicionalmente estavel, mas k afeta a precisao
    k = 0.5 * h**2 # Escolha de k para comparacao com os outros metodos
    Nt = int(T / k)
    k = T / Nt  # Ajuste para chegar exatamente em T=1
    r = k / h**2

    print(f"h={h:.4f}, k={k:.6f}, Nt={Nt}, r={r:.4f}")

    U = exact_solution(x, 0)

    # Matrizes para o sistema implicito (A * U_new = B * U_old + RHS)
    # Matriz A (lado esquerdo): (1 + r) * I - (r/2) * (Dxx)
    main_diag_A = (1 + r) * np.ones(Nx - 1)
    off_diag_A = (-r / 2) * np.ones(Nx - 2)
    A = np.diag(main_diag_A) + np.diag(off_diag_A, k=1) + np.diag(off_diag_A, k=-1)

    # Matriz B (lado direito): (1 - r) * I + (r/2) * (Dxx)
    main_diag_B = (1 - r) * np.ones(Nx - 1)
    off_diag_B = (r / 2) * np.ones(Nx - 2)
    B = np.diag(main_diag_B) + np.diag(off_diag_B, k=1) + np.diag(off_diag_B, k=-1)

    for n in range(Nt):
        t_old = n * k
        t_new = (n + 1) * k

        # Lado direito da equacao (B * U_old + k * (source_term_old + source_term_new) / 2)
        RHS = k * 0.5 * (source_term(x[1:Nx], t_old) + source_term(x[1:Nx], t_new))
        b = B @ U[1:Nx] + RHS

        # Resolver o sistema linear A * U_new[1:Nx] = b
        U_new_internal = np.linalg.solve(A, b)

        U_new = np.zeros_like(U)
        U_new[0] = 0.0
        U_new[Nx] = 0.0
        U_new[1:Nx] = U_new_internal
        U = U_new

    u_exact = exact_solution(x, T)
    error = np.sqrt(h * np.sum((U[1:Nx] - u_exact[1:Nx])**2))

    return error, k, Nt, x, U, u_exact

def generate_plots_for_method(method_func, method_name):
    """
    Gera graficos para um dado metodo numerico.
    """
    print(f"\nExecutando simulacoes para {method_name}...")

    h_values = [0.1, 0.05, 0.025]
    errors = []
    k_values = []
    all_results = []

    for h in h_values:
        #print(f"Simulando h = {h}...")
        print()
        start_time = time.time()

        error, k, Nt, x, U, u_exact = method_func(h)
        errors.append(error)
        k_values.append(k)
        all_results.append((x, U, u_exact))

        elapsed = time.time() - start_time
        print(f"Erro: {error:.2e}, Tempo: {elapsed:.2f}s")

    print(f"\nGerando graficos para {method_name}...")

    plt.figure(figsize=(18, 5))

    # GRAFICO 1: Convergencia do erro
    plt.subplot(1, 3, 1)
    plt.loglog(h_values, errors, 'bo-', linewidth=2, markersize=8,
               label=f'Erro numerico {method_name}')

    if len(h_values) >= 2:
        h_ref = np.linspace(h_values[-1], h_values[0], 10)
        # A ordem de convergencia para Forward e Backward Difference é O(h^2 + k) ou O(h^2 + k^2) se k ~ h^2
        # Para Crank-Nicolson eh O(h^2 + k^2)
        # Assumindo k ~ h^2, a ordem eh O(h^2)
        C = errors[0] / (h_values[0]**2)
        error_theoretical = C * h_ref**2
        plt.loglog(h_ref, error_theoretical, 'r--', linewidth=2,
                   label='O(h²) teorico')

    plt.xlabel('h (passo espacial)')
    plt.ylabel('Erro L²')
    plt.title(f'Convergencia - {method_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # GRAFICO 2: Solucao numerica vs exata para o primeiro h
    plt.subplot(1, 3, 2)
    x, U, u_exact = all_results[0]

    plt.plot(x, u_exact, 'r-', linewidth=2, label='Solucao Exata')
    plt.plot(x, U, 'bo', markersize=3, label=f'Solucao Numerica {method_name}')

    error_max = np.max(np.abs(U - u_exact))
    plt.text(0.05, 0.15, f'Erro max: {error_max:.1e}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.xlabel('x')
    plt.ylabel('u(x, T=1)')
    plt.title(f'Solucoes - h = {h_values[0]} ({method_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # GRAFICO 3: Relacao k vs h
    plt.subplot(1, 3, 3)
    plt.loglog(h_values, k_values, 'go-', linewidth=2,
               label=f'k utilizado {method_name}')

    # Para Forward Difference, mostrar limite de estabilidade
    if method_name == "Forward Difference":
        k_limite = [0.5 * h**2 for h in h_values]
        plt.loglog(h_values, k_limite, 'r--', linewidth=2,
                   label='Limite estabilidade (k=0.5h^2)')
    elif method_name == "Backward Difference" or method_name == "Crank-Nicolson":
        # Ambos sao incondicionalmente estaveis, mas k ainda afeta a precisao
        # Podemos mostrar a relacao k=h^2 para referencia de precisao
        k_h_squared = [h**2 for h in h_values]
        plt.loglog(h_values, k_h_squared, 'm--', linewidth=2,
                   label='k=h^2 (referencia)')

    plt.xlabel('h')
    plt.ylabel('k')
    plt.title(f'Relacao k vs h ({method_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'graficos_{method_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    generate_plots_for_method(forward_difference, "Forward Difference")
    generate_plots_for_method(backward_difference, "Backward Difference")
    generate_plots_for_method(crank_nicolson, "Crank-Nicolson")

    print("\nTodos os graficos foram gerados e salvos como PNG.")

