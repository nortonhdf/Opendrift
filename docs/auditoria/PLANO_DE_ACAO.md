# PLANO DE AÇÃO — Fase 8 (correções) e além

> Separado em **CONSERTAR** (executável pelo auditor, na branch `audit/revisao-completa`, commits
> atômicos, cada correção com teste) e **DECIDIR/FAZER** (ações que são do autor — decisões de
> ciência, credenciais, custo de máquina).
>
> **Decisões do autor (2026-07-29, completas — ver PERGUNTAS_ABERTAS.md):** beaching via opção B
> (ampliar domínio + regerar); incluir SST; mixing vertical OFF com opção; caixa nova
> lon −45..−36 / lat −27..−19; volume default declarado (proposta 10 m³); classificação de óleo
> corrigida pela regra 15–22°→MEDIUM; **Jubarte → Papa-Terra**; coordenadas de todos os campos
> derivadas do shapefile ANP; outputs permanecem no git; entulho aprovado para remoção; idioma
> padronizado em inglês (código/UI/docs de projeto); 2024 = ano holdout futuro do ML.

## Bloco 0 — Preparação

| # | Ação | Quem | Esforço |
|---|---|---|---|
| 0.1 | Criar branch `audit/revisao-completa` a partir de `feature/oil-budget-beaching` | auditor | min |
| 0.2 | Montar esqueleto de testes (`main/tests/`, pytest já no env) com fixture de NetCDF sintético | auditor | 1–2 h |

## Bloco 1 — Críticos (🔴), ordem de execução

| # | Ação | Quem | Detalhe |
|---|---|---|---|
| 1.1 | **Status por `flag_meanings`** em `compute_beaching.py`, `compute_risk_grids.py` e `app.py` (helper único, ex. `main/status_utils.py`) + teste de regressão com arquivo sintético `active missing_data stranded` | auditor | corrige 🔴 1 no código; inclui o caso `last == nt-1` (🟡 8) e o rótulo da UI (🟡 7) |
| 1.2 | **Mixing vertical**: parâmetro `vertical_mixing: bool = False` em `run_simulation`, aplicado via `drift:vertical_mixing`; remover o bloco morto `USE_3D` (chave inexistente); corrigir docstrings | auditor | corrige 🔴 2; opção fica disponível para runs 3D futuros |
| 1.3 | **Fallback smoke-test vira opt-in** (`smoke_test=True`), senão exceção clara | auditor | corrige 🟠 5 |
| 1.4 | **SST**: incluir `thetao` no download/prep CMEMS e no reader; fallback documentado (~24 °C) só como rede de segurança | auditor (código) + **autor (rodar download — credenciais CMEMS)** | corrige 🟠 3 |
| 1.5 | **Ampliar caixa de forçantes** nos scripts de download (**aprovado: lon −45..−36, lat −27..−19**) + aviso automático se >2% de `missing_data` num run | auditor (código) + **autor (downloads: CMEMS + ERA5, ~GBs)** | corrige 🟠 4 |
| 1.6 | **Volume de derrame**: expor `spill_m3` em `run_simulation`, default **10 m³ instantâneo** declarado na UI/docs | auditor | corrige 🟠 6 (decisão: "definir um default; tipo não é o foco") |
| 1.8 | **Campos**: trocar Jubarte → **Papa-Terra** (14–17,4° API, sul de Campos); corrigir lat de Roncador (−21,977 conforme referência); **derivar as 6 coordenadas do shapefile ANP** (fonte citável única) e corrigir classificação de óleo pela regra 15–22°→MEDIUM (Roncador/Frade/Papa-Terra conforme API) | auditor | resolve 🟡 9 + perguntas 3/4/5 |
| 1.7 | **Regerar tudo**: `rebuild_all.py --fresh` com o código corrigido (~3,5–4 h máquina) e conferir novo balanço de `missing_data` ≈ 0 | **autor (custo de máquina)**, script pronto | materializa 1B |

⚠ Dependência: 1.7 só depois de 1.1–1.6 mesclados. Os produtos atuais em `outputs/` ficam
inválidos para beaching até lá — não apagar antes (são a única base de comparação).

## Bloco 2 — Médios (🟡)

| # | Ação | Quem |
|---|---|---|
| 2.1 | Unificar constantes (domínio da grade, `GRID_RES`, datas de temporada) num módulo único (ex. `main/domain_config.py`) | auditor |
| 2.2 | Alinhar `environment.yml` ao env real (pins verificados: xarray 2026.4.0 etc.) | auditor |
| 2.3 | ~~Contradição API↔categoria~~ → **decidido**: regra do cabeçalho vale; classificação corrigida no item 1.8 | — |
| 2.4 | Limpeza de entulho versionado (**aprovada**): remover `openoil_smoketest.nc` (órfão), `test_wind_off.*`, `openoil_run.nc`/`tracks.png` obsoletos; parar de versionar `.png` de ensemble (gitignore) | auditor |
| 2.5 | ~~Armazenamento dos outputs~~ → **decidido: manter no git** | — |
| 2.6 | Aba Custom: não exibir run pré-existente sem `custom_cfg` correspondente (ou rotular "run anterior") | auditor |
| 2.7 | **Decidido (delegado ao auditor)**: manter `current_uncertainty=0.05`/`wind_uncertainty=0.5` e **declará-los** na doc do método; trocar advecção para `runge-kutta4` junto da regeração (avisado: mudança científica leve, entra no mesmo rebuild) | auditor |
| 2.9 | Experimento de convergência (autorizado): risk grid com 5 vs 10 membros e 200 vs 500 partículas; registrar em docs antes de fixar números na metodologia | auditor |
| 2.8 | Testes: cobertura mínima para `status_utils`, `budget_path_for`/`save_oil_budget`, `particles_to_grid`/`stranding_events` (casos analíticos), smoke do `run_simulation` com reader constante explícito | auditor |

## Bloco 3 — Baixos (🔵)

| # | Ação |
|---|---|
| 3.1 | Fundir `prep_era5_wind.py` + `patch_wind_cf.py` num único prep (elimina `wind.nc`) |
| 3.2 | Logs do rebuild em UTF-8 (usar `Tee-Object`/encoding explícito ou logging em Python) |
| 3.3 | Substituir `datetime.utcnow()`/`utcfromtimestamp` pelas formas timezone-aware |
| 3.4 | Robustecer selectbox da aba Risco (`app.py:553-557`) |
| 3.5 | Atualizar `main/CLAUDE.md` e `main/README.md` (remover claims falsos: "2D sem mixing", "beaching 0–89% correto") — parcialmente coberto pelos docs desta auditoria |

## Depois da Fase 8 (fora do escopo da auditoria)

- Plano de amostragem para a camada de IA (LHS sobre posição/data/volume; anos adicionais) —
  ver CAMADA_IA.md.
- Ondas ERA5 reais (cadeia pronta) se o Stokes parametrizado não bastar.
- Deploy do Streamlit (atenção: aba 4 exige o env OpenDrift no servidor).

## Estimativa de esforço total da Fase 8

- Blocos 0–1 (código): ~1 dia de trabalho do auditor + downloads do autor + ~4 h de máquina.
- Bloco 2: ~½ dia (itens 2.3/2.5/2.7 aguardam decisões).
- Bloco 3: ~2 h.
