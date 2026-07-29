# DIAGNÓSTICO — todos os achados, por severidade

> Auditoria 2026-07-29. Números citados foram **medidos** (scripts de verificação executados no
> env `opendrift`), nunca estimados. Decisões do autor já tomadas: corrigir beaching com
> ampliação de domínio + regeração (1B), incluir SST (2), desligar mixing vertical mantendo-o
> como opção (3).

---

## 🔴 CRÍTICOS

### 🔴 1 — Partículas que saem do domínio são contadas como "encalhadas"

- **Onde:** `main/scripts/compute_beaching.py:52` (`STRANDED = 1`) e `:74`.
- **O que acontece:** o OpenDrift grava o significado dos códigos de `status` **por arquivo**
  (attr `flag_meanings`), na ordem em que os tipos de desativação ocorrem no run. Varredura dos
  240 NetCDF do ensemble: 181 arquivos `'active'`, 48 `'active missing_data'`,
  10 `'active stranded'`, 1 `'active stranded missing_data'`. Em **48 arquivos o código 1
  significa `missing_data`** (partícula que cruzou a borda da caixa de forçantes, onde o
  fallback de correntes/vento é `None`), não `stranded`. O script assume 1=stranded sempre.
- **Por que está errado:** a fração de encalhe, o grid espacial e os percentis de tempo passam a
  medir majoritariamente *saída de domínio*, não encalhe.
- **Evidência:** recontagem usando o `flag_meanings` de cada arquivo:
  **8.870 partículas reportadas como encalhadas vs 1.702 encalhes reais**. Casos extremos:
  `peregrino_oct` 85%→**0%** real, `marlim_jan` 41%→**0%**, `jubarte_apr` 30%→**0%**.
  Encalhe verdadeiro só existe em **Frade** (2–45%). As células "quentes" de
  `peregrino_oct_beaching.npz` estão todas em lon −42,5°/−42,6° — a **borda oeste da caixa de
  forçantes, oceano aberto** (a costa não está lá).
- **Impacto:** os 24 grids de `outputs/beaching/`, a aba 3 do app, os percentis
  `hours_p10/p50/p90` e as frases "beaching 0–89% sazonal" de `main/CLAUDE.md:89-90` e
  `main/README.md` estão invalidados.
- **Correção (aprovada pelo autor — opção B):** (a) resolver o código `stranded` pelo
  `flag_meanings` de cada arquivo; (b) ampliar a caixa de download das forçantes e **regerar o
  ensemble** para que a deriva real caiba no domínio; (c) recomputar beaching + risk; (d) teste
  de regressão com um NetCDF sintético contendo `missing_data` antes de `stranded`.

### 🔴 2 — Mixing vertical LIGADO enquanto o projeto declara "2D sem mixing vertical"

- **Onde:** `main/run_open_oil.py:44-45` e `:218-219`.
- **O que acontece:** `USE_3D=False` faz o bloco `if USE_3D and DISABLE_VERTICAL_MIXING:` nunca
  executar; além disso a chave usada (`processes:vertical_mixing`) **não existe** neste OpenOil
  (`ValueError` se executasse — a chave real é `drift:vertical_mixing`). Resultado: todas as 288
  simulações rodaram com `drift:vertical_mixing = True` e `processes:dispersion = True`
  (verificado no modelo instanciado).
- **Por que está errado:** o docstring do runner, `main/CLAUDE.md:16` e a defesa do método
  descrevem um modelo de superfície 2D; o que rodou é um modelo com entranhamento e mistura
  vertical. Partícula submersa não recebe deriva de vento → as trajetórias são materialmente
  diferentes das de um modelo 2D.
- **Evidência:** budgets medidos: até **520 kg submersos de ~963 kg** totais em alguns runs
  (>50% da massa em subsuperfície) e até 77% de massa dispersa.
- **Impacto:** nenhum resultado é "falso" em si (o modelo 3D é até mais completo), mas **a
  descrição do método está errada** — numa defesa pública isso é indefensável.
- **Correção (aprovada — opção "desligar, mantendo opção"):** parâmetro `vertical_mixing: bool = False`
  em `run_simulation`, aplicado via `o.set_config("drift:vertical_mixing", ...)`; batches usam
  `False`; documentação corrigida; regerar produtos (já necessário pelo 🔴 1).

---

## 🟠 GRAVES

### 🟠 3 — Weathering calculado a 10 °C num mar de ~24 °C

- **Onde:** ausência de reader de SST em `run_open_oil.py:92-118`; fallback default
  `environment:fallback:sea_water_temperature = 10` (verificado na config efetiva).
- **O que acontece:** evaporação e emulsificação do modelo NOAA dependem fortemente de
  temperatura; todas as 288 simulações usaram 10 °C.
- **Evidência:** config dump do modelo instanciado exatamente como `run_simulation` monta;
  nenhum dos inputs contém variável de temperatura (verificado abrindo os arquivos).
- **Impacto:** budgets (evaporado/dispersado/superfície) sistematicamente enviesados —
  evaporação subestimada. Afeta as % exibidas nas 4 abas do app.
- **Correção (aprovada):** incluir `thetao` no download CMEMS (mesmo dataset) e propagar no
  prep; alternativa mínima: `environment:fallback:sea_water_temperature` ≈ climatologia local.
  Requer regerar budgets (já necessário).

### 🟠 4 — Caixa de forçantes pequena demais: 15,9% das partículas morrem na borda

- **Onde:** `download_cmems_currents.py:26-29` e `download_era5_wind.py:15` (caixa
  −42,5..−39,0 / −24,5..−21,0); grade de análise maior que a caixa
  (`compute_risk_grids.py:39-40`: −43..−38,5 / −25..−21).
- **O que acontece:** 7.611/48.000 partículas do ensemble e 3.378/24.000 dos cenários são
  desativadas por `missing_data` dentro das 120 h.
- **Impacto:** além de alimentar o 🔴 1, o mapa de persistência (`prob_final`) exclui
  silenciosamente essas partículas → viés de borda não documentado.
- **Correção (aprovada):** ampliar caixa de download (sugestão: lon −45..−36, lat −27..−19 —
  validar com o autor) e regerar; adicionar aviso quando >N% dos elementos desativarem por
  `missing_data`.

### 🟠 5 — Fallback silencioso para reader constante fictício

- **Onde:** `main/run_open_oil.py:243-245` (`if not used_real: add_smoke_test_reader(o)`).
- **O que acontece:** se `currents.nc` estiver ausente/corrompido (e vento off/ausente), o
  runner injeta corrente constante de 0,3 m/s para leste e **segue rodando** com apenas um
  `[INFO]` no stdout.
- **Por que está errado:** num batch, geraria 48 cenários plausíveis e fisicamente falsos sem
  nenhum erro — exatamente a classe "erro silencioso com número plausível".
- **Evidência:** leitura do código; caminho nunca ativado nos outputs atuais (verificado:
  trajetórias respondem a forçantes reais).
- **Correção:** tornar o smoke-mode **opt-in** (`smoke_test=True`) e levantar exceção se nenhum
  reader real carregar; teste cobrindo o caso.

### 🟠 6 — Volume de derrame = 1 m³, herdado do default e nunca declarado

- **Onde:** `run_simulation` nunca seta `seed:m3_per_hour` (default 1); confirmação empírica:
  `mass_total[0] = 963,5 kg` com densidade 963,5 kg/m³ no budget de `peregrino_jan_wind_on`.
- **Impacto:** as % do budget são consistentes, mas processos dependentes de espessura de filme
  não escalam linearmente; qualquer leitura em massa absoluta é enganosa. Precisa virar
  parâmetro explícito e premissa declarada.
- **Correção:** expor `spill_m3` em `run_simulation` + documentar o cenário de referência
  (decisão do autor sobre o valor — ver PERGUNTAS_ABERTAS #3).

---

## 🟡 MÉDIOS

| # | Achado | Onde | Evidência/Impacto |
|---|---|---|---|
| 7 | App rotula `missing_data` como "Stranded / inactive" | `app.py:397-408, 432-434` | Cenários exibem até 14% de "encalhe" que é saída de domínio |
| 8 | Encalhe no último timestep descartado (`last < nt-1`) | `compute_beaching.py:74` | 23 partículas reais descartadas no ensemble atual |
| 9 | Regra API↔óleo contradita pelos próprios dados | `fields_config.py:4-6` vs `:31,:40,:49` | Roncador 18°/Jubarte 16,5°/Frade 18° = "HEAVY", regra diz 15–22 → MEDIUM |
| 10 | `environment.yml` diverge do env real | `environment.yml:18` (`xarray<=2025.9.0`) vs instalado 2026.4.0 | Recriar o env não reproduz o ambiente que gerou os outputs |
| 11 | Constantes triplicadas (grade, datas) | `compute_risk_grids.py:39-41`, `compute_beaching.py:46-48`, `app.py:33,36-41`, `precompute_scenarios.py:37-42` | Coincidem hoje por disciplina, não por construção |
| 12 | Zero testes automatizados para `main/` | — | Só 2 smoke tests manuais; nada em CI |
| 13 | Entulho versionado | `outputs/openoil_smoketest.nc` (órfão), `test_wind_off.*`, `openoil_run.nc`/`tracks.png` obsoletos (pré-fix de oil-type), 240 PNGs de ensemble não lidos por nada | ~150 MB de git para artefatos regeneráveis/mortos |
| 14 | Aba Custom mostra run antigo com rótulos novos | `app.py:819-826` | 1ª abertura exibe o `.nc` commitado de 02/06 legendado com os toggles atuais da sidebar |
| 15 | Difusão estocástica implícita não documentada | defaults `drift:current_uncertainty=0.05`, `drift:wind_uncertainty=0.5`; `horizontal_diffusivity=0` | Única fonte de espalhamento entre partículas; deveria ser escolha declarada |
| 16 | Advecção Euler (dt=600 s) | default `drift:advection_scheme='euler'` | RK4 é a recomendação usual e custa pouco |

---

## 🔵 BAIXOS

| # | Achado | Onde |
|---|---|---|
| 17 | Cadeia prep+patch em 2 passos com `wind.nc` intermediário | `prep_era5_wind.py` + `patch_wind_cf.py` |
| 18 | `rebuild_all.log` em UTF-16 (redirect `*>` do PowerShell) — ilegível para grep/parsers | `main/rebuild_all.log` |
| 19 | APIs datetime deprecadas (`utcnow`, `utcfromtimestamp`) | `precompute_scenarios.py:115`, `app.py:79-80` |
| 20 | Selectbox de campo da aba Risco montado por comparação frágil com a 1ª entrada do manifest | `app.py:553-557` |
| 21 | Densitymapbox com `radius` em px como visual de probabilidade — suaviza/borra células de 0,1° (estética, mas pode enganar leitura quantitativa) | `app.py:303-311, 359-371` |

---

## Estatística da varredura de integridade (para referência)

- 288 NetCDF abertos: **0 corrompidos**; manifests ↔ disco batem 100% nos dois sentidos.
- Status finais (significados reais): scenarios 20.622 active / 3.378 missing_data;
  ensemble 38.687 active / 7.611 missing_data / 1.702 stranded.
- Runs deterministas: 2 execuções idênticas → desvio médio 0,00 km.
- App headless (`AppTest`): 4 abas, 0 exceções, 17,5 s.
