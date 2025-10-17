# Numerical Methods in Differential Equations

Descrição
- Implementação e estudo de métodos numéricos para equações diferenciais ordinárias (ex.: diferenças finitas forward, estudos de convergência e estabilidade).
- Exemplos de comparação entre solução numérica e solução exata, com geração de gráficos de erro e relação entre parâmetros (ex.: \(k\) vs \(h\)).

Resultados
- O repositório contém scripts para gerar figuras semelhantes à imagem anexa:
  - Convergência do erro em norma L¹ em função do passo espacial \(h\).
  - Comparação da solução numérica com a solução exata para um \(h\) fixo.
  - Relação entre o parâmetro \(k\) utilizado e \(h\).

Requisitos
- Windows, Python 3.11 (compatível com `pyproject.toml`).
- Dependências principais:
  - `numpy`
  - `matplotlib` (para plotagem)
- Recomenda-se uso de `poetry` (opcional).

Instalação
- Com Poetry:
  - Instalar o Poetry (se ainda não tiver): seguir instruções em https://python-poetry.org/docs/#installation
  - No diretório do projeto:
    - `poetry install`
    - Para executar scripts com o ambiente virtual do Poetry: `poetry run python scripts/plot_convergence.py`
- Sem Poetry:
  - Criar e ativar um ambiente virtual:
    - `python -m venv .venv`
    - `.\.venv\Scripts\activate`
  - Instalar dependências:
    - `pip install numpy matplotlib`
  - Executar scripts:
    - `python scripts/plot_convergence.py`

Uso
- Scripts (exemplos):
  - `scripts/plot_convergence.py` — gera os três subplots de convergência, solução comparada e relação \(k\) vs \(h\).
  - `scripts/solve_forward_difference.py` — rotina para resolver o problema com forward difference e salvar erros.
- Parâmetros configuráveis:
  - passo espacial \(h\), passo temporal \(k\), função exata, condições de contorno/condição inicial.
  - editar variáveis no topo dos scripts ou usar argumentos de linha de comando, conforme implementado.

Estrutura sugerida de arquivos
- `pyproject.toml` — metadados do projeto (Poetry).
- `README.md` — este ficheiro.
- `scripts/plot_convergence.py` — script para reproduzir os gráficos.
- `src/` — código-fonte das implementações numéricas.
- `data/` — (opcional) dados reais ou resultados pré-computados.
- `figures/` — figuras geradas (PNG, PDF).

Como reproduzir a figura mostrada
1. Configurar dependências conforme seção "Instalação".
2. Executar:
   - `python scripts/plot_convergence.py`
   - ou `poetry run python scripts/plot_convergence.py`
3. A saída será salva em `figures/convergence_and_solution.png` (ou exibida interativamente), dependendo da implementação do script.

Contribuição
- Abrir issues para bugs ou melhorias.
- Pull requests com testes e breves descrições das alterações.
