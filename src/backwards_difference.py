import numpy as np

def exact_solution(x, t):
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

def backward_difference_real(h, T=1.0):
    """
    Implementacao REAL do metodo Backward Difference
    """
    # Malha espacial
    Nx = int(1.0 / h)
    x = np.linspace(0, 1, Nx + 1)
    # mesma inicializacao do forward difference
    
    # Apesar de incondicionalmente estavel, estamos mantendo a malha em dimensoes identicas
    # Estamos fazendo isso para manter uma consistencia de erros com os mesmos tamanhos delta-t
    # Visando um Crank-Nicolson coerente
    k = 0.99 * 0.5 * h**2
    Nt = int(T / k)
    k = T / Nt  # Ajuste para chegar exatamente em T=1
    r = k / h**2
    
    print(f"h={h:.4f}, k={k:.6f}, Nt={Nt}")
    
    # Condicao inicial
    U = exact_solution(x, 0)
    
    #Resolvendo o sistema como o pseudocodigo apresentado nos slides da aula 10
    lower = np.zeros(Nx+1)
    upper = np.zeros(Nx+1)
    z = np.zeros(Nx+1)
    
    lower[1] = 1+2*r
    upper[1] = -r/lower[1]
    for i in range(2, Nx-1):
        lower[i] = 1+2*r+r*upper[i-1]
        upper[i] = -r/lower[i]
    lower[Nx] = 1+2*r+r*upper[Nx-1]
    
    for j in range(1, Nt):
        t = j*k
        U_new = U.copy()
        z[1] = U_new[1]/lower[1]

        for i in range(2, Nx):
            z[i] = (U_new[i]+r*z[i-1])/lower[i]

        U_new[Nx] = z[Nx]

        for i in range(Nx-1, 1):
            U_new[i] = z[i]-upper[i]*U_new[i+1]

        U = U_new
    
    # Calcular erro
    u_exact = exact_solution(x, T)
    error = np.sqrt(h * np.sum((U[1:Nx] - u_exact[1:Nx])**2))
    
    return error, k, Nt, x, U, u_exact