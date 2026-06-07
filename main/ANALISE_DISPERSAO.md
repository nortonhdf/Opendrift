# Modelagem de Dispersão de Poluentes — Bacia de Campos
## Documentação Técnica do Fluxo de Análise

---

## 1. Proposta do Projeto

O projeto modela a dispersão de óleo cru no oceano a partir de vazamentos hipotéticos nos seis principais campos petrolíferos da **Bacia de Campos**, costa sudeste brasileira. O objetivo é fornecer mapas probabilísticos de risco de exposição costeira e de encalhe para uso operacional em cenários de resposta a emergências ambientais.

O sistema responde a perguntas como:
- *Dado um vazamento no campo X no mês Y, quais regiões costeiras têm maior probabilidade de ser atingidas?*
- *Em quanto tempo o óleo atinge a costa?*
- *Qual fração da massa derramada evapora, fica na superfície ou encalha?*

---

## 2. Campos Modelados

| Campo | Operadora | Profundidade (m) | API | Tipo de óleo (ADIOS) |
|---|---|---|---|---|
| Peregrino | PRIO / Equinor | 100 | 13° | GENERIC HEAVY CRUDE |
| Marlim | Petrobras | 720 | 20° | GENERIC MEDIUM CRUDE |
| Roncador | Petrobras | 1800 | 18° | GENERIC HEAVY CRUDE |
| Jubarte | Petrobras / Shell | 1300 | 16.5° | GENERIC HEAVY CRUDE |
| Frade | Petrobras / Chevron | 1100 | 18° | GENERIC HEAVY CRUDE |
| Albacora | Petrobras | 300 | 19° | GENERIC MEDIUM CRUDE |

A gravidade API determina o tipo de óleo no catálogo ADIOS (NOAA), que por sua vez define as constantes físico-químicas usadas no modelo de intemperismo.

---

## 3. Framework Científico

### 3.1 Rastreamento Lagrangiano de Partículas

O modelo usa **OpenDrift v1.14.7** com o módulo **OpenOil** — framework de código aberto mantido pelo Norwegian Meteorological Institute (MET Norway).

O método Lagrangiano representa a mancha de óleo como um conjunto de **N partículas independentes**, cada uma com posição (lon, lat, z) e estado próprios. A equação de movimento de cada partícula é:

```
dx/dt = u_corrente(x, t) + α · u_vento(x, t) + ξ(t)
```

Onde:
- `u_corrente` — correntes oceânicas interpoladas (CMEMS)
- `u_vento` — componente de deriva pelo vento (fator α ≈ 3% da velocidade)
- `ξ(t)` — termo estocástico de difusão horizontal (turbulência sub-grade)

### 3.2 Configurações da Simulação

| Parâmetro | Cenários | Ensemble |
|---|---|---|
| Partículas por run | 500 | 200 |
| Duração | 120 horas (5 dias) | 120 horas |
| Passo de tempo interno | 600 s (10 min) | 600 s |
| Passo de tempo de saída | 1800 s (30 min) | 1800 s |
| Dimensões | 2D superfície | 2D superfície |
| Mixing vertical | desativado | desativado |
| Velocidade máxima | 2,0 m/s | 2,0 m/s |

### 3.3 Modelo de Intemperismo (Weathering)

OpenOil usa o modelo **NOAA** de intemperismo, que simula os processos de alteração físico-química do óleo ao longo do tempo. Para cada partícula, são rastreadas as seguintes massas:

| Variável | Descrição |
|---|---|
| `mass_oil` | Massa de óleo remanescente em superfície |
| `mass_evaporated` | Massa perdida por evaporação |
| `mass_dispersed` | Massa dispersa na coluna d'água |
| `mass_biodegraded` | Massa biodegradada |
| `water_fraction` | Fração de água na emulsão (grau de emulsificação) |
| `density` | Densidade da emulsão (kg/m³) |
| `viscosity` | Viscosidade dinâmica (Pa·s) |

O **oil budget sidecar** (`_budget.npz`) agrega esses valores em escala da mancha inteira ao longo do tempo, permitindo análise da evolução temporal da partição de massa.

### 3.4 Dados de Forçamento

| Dado | Fonte | Resolução | Cobertura |
|---|---|---|---|
| Correntes oceânicas | CMEMS (Copernicus Marine) | ~1/12° (~8 km) diária | 2025 completo |
| Vento superficial | ERA5 (ECMWF Reanalysis) | ~0,25° (~28 km) horária | 2025 completo |
| Ondas / Stokes drift | ERA5 (futuro) | ~0,5° | não incluído |

Os arquivos de entrada são normalizados para convenções CF (Climate and Forecast) via scripts de pré-processamento (`prep_currents.py`, `prep_era5_wind.py`) antes de serem lidos pelo OpenDrift.

---

## 4. Fluxo de Análise — Pipeline Completo

```
[Dados brutos]          [Pré-processamento]       [Simulações]
CMEMS currents.nc  ──►  prep_currents.py    ──►
ERA5 wind_cf.nc    ──►  prep_era5_wind.py   ──►   run_open_oil.py
                                                         │
                   ┌─────────────────────────────────────┤
                   │                                     │
                   ▼                                     ▼
        [Estágio 1: Cenários]              [Estágio 2: Ensemble]
        precompute_scenarios.py            run_ensemble.py
        48 runs = 6 campos                 240 runs = 6 campos
               × 4 estações                      × 4 estações
               × 2 wind states                   × 10 datas de início
               500 partículas                    200 partículas
               outputs/scenarios/               outputs/ensemble/
                   │                                     │
                   └─────────────────┬───────────────────┘
                                     │
                          [Estágio 3: Risk Grids]
                          compute_risk_grids.py
                          24 grids = 6 × 4 estações
                          resolução 0,1° (~11 km)
                          outputs/risk_grids/
                                     │
                          [Estágio 4: Beaching]
                          compute_beaching.py
                          24 grids = 6 × 4 estações
                          outputs/beaching/
                                     │
                          [Aplicação Streamlit]
                          main/app.py
                          4 abas de visualização
```

### 4.1 Estágio 1 — Cenários Pré-computados (48 runs)

**Script:** `main/scripts/precompute_scenarios.py`

Cada cenário representa uma combinação de campo, estação e estado de vento:

- **6 campos** × **4 estações** (jan / abr / jul / out) × **2 estados de vento** (on/off)
- Data de início fixa por estação: 15 de janeiro, 15 de abril, 15 de julho, 15 de outubro de 2025
- Estado `wind_off` desativa o reader de vento para isolar o efeito das correntes

**Saídas por cenário:**
- `<campo>_<estação>_<wind>.nc` — trajetórias (lon, lat, status, z, massas)
- `<campo>_<estação>_<wind>_budget.npz` — orçamento de massa ao longo do tempo
- `<campo>_<estação>_<wind>.png` — mapa cartográfico (cosmético, usado no app)

### 4.2 Estágio 2 — Ensemble de Monte Carlo (240 runs)

**Script:** `main/scripts/run_ensemble.py`

Para capturar a **variabilidade das condições reais** dentro de cada estação, são rodados 10 membros com datas de início diferentes:

- 10 datas de início espaçadas uniformemente no mês: dias 1, 4, 7, 10, 13, 16, 19, 22, 25, 28
- Todos os membros usam `use_wind=True` (condições realistas)
- Cada membro gera seu próprio `_budget.npz`

**Justificativa:** correntes e ventos variam significativamente dentro de um mesmo mês. Usar apenas uma data de início subestima a incerteza. O ensemble de 10 membros amostra essa variabilidade sem precisar perturbar parâmetros físicos.

### 4.3 Estágio 3 — Grids de Risco (24 grids)

**Script:** `main/scripts/compute_risk_grids.py`

Agrega os 10 membros do ensemble de cada par (campo, estação) num grid de probabilidade de exposição:

**Domínio:** lon ∈ [-43,0°, -38,5°], lat ∈ [-25,0°, -21,0°], resolução 0,1° (~11 km)

**Duas camadas calculadas por grid:**

```
prob_any[i,j]   = (n° membros que visitaram célula (i,j) em qualquer momento) / 10
prob_final[i,j] = (n° membros com partícula em (i,j) no instante final) / 10
```

- `prob_any` — **risco de exposição**: probabilidade de a célula ser tocada pelo óleo em algum momento dos 5 dias
- `prob_final` — **risco de persistência**: probabilidade de o óleo ainda estar presente ao final

**Saída:** `<campo>_<estação>_risk.npz` com arrays `prob_any`, `prob_final`, `lons`, `lats`, `n_members`

### 4.4 Estágio 4 — Análise de Encalhe (24 grids)

**Script:** `main/scripts/compute_beaching.py`

Uma partícula "encalhou" quando o OpenDrift atribui `status = 1` (elemento desativado por contato com a costa via `roaring-landmask`). O último ponto válido da trajetória é o local de encalhe; o tempo decorrido até esse instante é o **tempo de encalhe**.

**Métricas calculadas:**

| Métrica | Descrição |
|---|---|
| `strand_grid` | Probabilidade de encalhe por célula de 0,1° |
| `stranded_fraction` | Fração total de partículas que encalharam |
| `hours_p10 / p50 / p90` | Percentis do tempo de encalhe (horas) |
| `centroid_lon / centroid_lat` | Baricentro ponderado dos encalhes |

**Saída:** `<campo>_<estação>_beaching.npz`

---

## 5. Resultados Obtidos

### 5.1 Risco de Exposição — Janeiro (todos os campos)

| Campo | Peak prob_any | Células afetadas (0,1°) |
|---|---|---|
| Albacora | 100% | 221 |
| Roncador | 100% | 175 |
| Marlim | 100% | 170 |
| Jubarte | 100% | 148 |
| Frade | 100% | 113 |
| Peregrino | 100% | 96 |

O peak de 100% indica que todos os 10 membros do ensemble passaram pela mesma célula em pelo menos um momento — área de risco determinístico no horizonte de 5 dias.

### 5.2 Encalhe — Variação Sazonal (Peregrino)

| Estação | % encalhado | Mediana (h) |
|---|---|---|
| Outubro | **85,0%** | 70 h |
| Janeiro | 52,6% | 76 h |
| Abril | 30,1% | 70 h |
| Julho | **9,6%** | 112 h |

A variação sazonal é expressiva: em outubro (primavera austral) as correntes dominantes aproximam o óleo da costa rapidamente; em julho (inverno) a circulação afasta as partículas para o largo.

### 5.3 Encalhe — Comparação entre Campos (Janeiro)

| Campo | % encalhado | Mediana (h) |
|---|---|---|
| Peregrino | 52,6% | 76 h |
| Marlim | 41,3% | 102 h |
| Roncador | 10,1% | 96 h |
| Jubarte | 10,1% | 36 h |
| Frade | 2,3% | 67 h |
| Albacora | 9,7% | 108 h |

Peregrino (mais próximo da costa, ~85 km offshore) apresenta a maior fração de encalhe e tempo mediano intermediário. Jubarte tem baixa fração de encalhe mas quando ocorre é o mais rápido (36 h), indicando que quando as correntes a conduzem à costa o fazem de forma direta.

---

## 6. Ferramentas de Análise Estatística e Regressão Aplicáveis

Os outputs do pipeline (grids .npz, budgets .npz, trajetórias .nc) são estruturas de dados regulares que permitem uma série de análises estatísticas e de aprendizado de máquina. As seguintes abordagens são diretamente aplicáveis:

### 6.1 Regressão Logística — Previsão de Encalhe

**Objetivo:** prever a probabilidade de encalhe de uma simulação a partir de variáveis meteorológicas e oceanográficas do instante de início.

**Variáveis preditoras (X):**
- Velocidade e direção do vento médio nas primeiras 24h
- Velocidade e direção da corrente superficial na fonte
- Mês / estação (variável categórica)
- Distância à costa do campo

**Variável resposta (y):** `stranded_fraction` binarizada (encalhou acima de threshold ou não)

**Dados disponíveis:** 240 membros do ensemble, cada um com seu `_budget.npz` e a posição das forçantes ao longo do tempo.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# X: [wind_u, wind_v, curr_u, curr_v, season_sin, season_cos]
# y: (stranded_fraction > 0.2).astype(int)
```

### 6.2 Regressão Linear Múltipla — Tempo de Encalhe

**Objetivo:** modelar `hours_p50` (tempo mediano de encalhe) em função das condições ambientais.

**Variáveis preditoras:** mesmas da logística + componentes de corrente integradas ao longo de 24h, velocidade de deriva resultante.

**Interpretação:** os coeficientes revelam a sensibilidade do tempo de encalhe a cada componente de corrente/vento, útil para priorização de resposta de emergência.

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
```

### 6.3 Regressão de Poisson / Binomial Negativa — Área Afetada

**Objetivo:** modelar o número de células afetadas (`(prob_any > 0).sum()`) em função das condições de vento e corrente.

**Justificativa:** contagem de células é uma variável inteira e positiva — modelos de contagem (Poisson, Binomial Negativa) são mais adequados que OLS, especialmente para distribuições com cauda longa.

```python
import statsmodels.api as sm
# sm.GLM com family=sm.families.NegativeBinomial()
```

### 6.4 Regressão Espacial — Interpolação de Risco entre Campos

**Objetivo:** dado que temos grids de risco para 6 campos, estimar o risco em posições intermediárias (ex.: um campo hipotético em uma nova localização).

**Método recomendado:** Kriging Ordinário ou Regressão com Kernel de Distância Inversa (IDW) sobre os campos de `prob_any` normalizados.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
# Treinar GP nos 6 pontos (lon_campo, lat_campo) → prob_any_max
```

### 6.5 Análise de Componentes Principais (PCA) — Padrões de Trajetória

**Objetivo:** identificar os modos dominantes de variabilidade das trajetórias do ensemble.

**Método:** representar cada membro do ensemble como uma sequência de posições (lon_t, lat_t) e aplicar PCA sobre a matriz de trajetórias. Os primeiros componentes correspondem aos padrões dominantes de dispersão.

**Utilidade:** agrupar membros do ensemble em "regimes" de circulação (ex.: pluma nordeste vs. pluma sul), identificar outliers.

```python
from sklearn.decomposition import PCA
# X: matriz (n_membros, 2 * n_timesteps) com lon e lat concatenados
```

### 6.6 Random Forest / Gradient Boosting — Classificação de Cenários de Alto Risco

**Objetivo:** classificar automaticamente combinações (campo, estação, data de início) em categorias de risco (baixo / médio / alto encalhe).

**Vantagem sobre modelos lineares:** captura interações não-lineares entre vento e corrente sem especificação prévia da forma funcional.

```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
```

**Features importantes esperadas:** velocidade do vento, orientação da corrente superficial, posição no ciclo sazonal.

### 6.7 Séries Temporais — Evolução do Oil Budget

**Objetivo:** ajustar modelos de decaimento à curva temporal de massa em superfície (`mass_surface(t)`), estimando a taxa de intemperismo efetiva por campo e estação.

**Modelo de decaimento exponencial:**
```
M_surface(t) = M_0 · exp(-λ · t)
```

**Regressão:** ajustar λ por mínimos quadrados não-lineares (`scipy.optimize.curve_fit`) para cada combinação (campo, estação). Comparar λ entre campos de API diferente para validar sensibilidade do modelo ao tipo de óleo.

```python
from scipy.optimize import curve_fit
import numpy as np

def decay(t, m0, lam): return m0 * np.exp(-lam * t)
popt, pcov = curve_fit(decay, hours, mass_surface)
```

### 6.8 Modelo Ensemble — Média Probabilística Ponderada

**Objetivo:** combinar os 10 membros do ensemble num único campo de risco ponderado pela qualidade da previsão (skill score) de cada data de início, usando os dados de reanálise ERA5 como referência.

**Método:** Ensemble Model Output Statistics (EMOS) — regressão linear dos grids de probabilidade brutos contra uma "verdade terrestre" (ocorrência observada ou hindcast de alta resolução).

---

## 7. Arquitetura da Aplicação (Streamlit)

O app `main/app.py` expõe o pipeline em 4 abas:

| Aba | Conteúdo |
|---|---|
| **Cenários** | Animação das 48 trajetórias pré-computadas + oil budget interativo |
| **Risk Maps** | Mapa de calor `prob_any` do ensemble por campo/estação |
| **Beaching** | Grid de probabilidade de encalhe + métricas de tempo |
| **Custom Run** | Simulação ao vivo configurável (campo, data, vento, Stokes) |

**Stack técnica:** Streamlit 1.58 · Plotly 5.x · xarray · NumPy · OpenDrift/OpenOil in-place

---

## 8. Possíveis Extensões

1. **Ondas / Stokes drift** — adicionar `waves_cf.nc` do ERA5 e habilitar `use_waves=True` nos cenários pré-computados; espera-se aumento de ~10–20% na fração de encalhe em campos mais costeiros.

2. **Regressão de encalhe em produção** — treinar o modelo logístico (§ 6.1) com os 240 membros disponíveis e expor a previsão como métrica adicional no app (tab 2), permitindo estimativa de risco sem rodar nova simulação.

3. **Sensibilidade ao volume derramado** — o modelo atual usa um volume fixo representativo; uma análise de sensibilidade com regressão linear sobre volume → área afetada quantificaria o impacto de diferentes tamanhos de derramamento.

4. **Dados de resposta observada** — integrar registros históricos de derramamentos na Bacia de Campos para validação estatística do modelo (skill score de probabilidade de Brier).

5. **Validação cruzada sazonal** — usar 9 dos 10 membros do ensemble para construir o grid de risco e o 10° como hold-out; calcular Brier Score para cada campo/estação e mapear onde o modelo é mais/menos confiável.
