# MAPA DO PROJETO

> Auditoria técnica de 2026-07-29. Cada afirmação foi verificada lendo o código,
> executando-o ou abrindo os dados. Estados: **ATIVO** / **MORTO** / **DUPLICADO** /
> **EXPERIMENTAL** / **QUEBRADO** / **NUNCA PERCORRIDO**.

## Visão de 10 segundos

O repositório é um **clone intocado do OpenDrift v1.14.7** (31.365 linhas, zero modificações
locais — verificado com `git diff c7b9a0d7..HEAD -- opendrift/ tests/`). Todo o projeto vive em
**`main/`**: 2.270 linhas de Python em 18 arquivos + 39 MB de forçantes + 202 MB de simulações
pré-computadas, tudo versionado no git.

```
Opendrift/                  ← clone do upstream (NÃO tocar; não é seu código)
├── opendrift/              ← núcleo OpenDrift v1.14.7, usado in-place    [ATIVO, upstream]
├── tests/, docs/, examples/← do upstream                                  [upstream]
├── environment.yml         ← modificado: +streamlit/plotly, pin openblas [ATIVO]
└── main/                   ← ══ SEU PROJETO ══
```

## main/ — arquivo por arquivo

### Núcleo (tudo ATIVO)

| Arquivo | Linhas | O que faz | Quem chama |
|---|---|---|---|
| `run_open_oil.py` | 305 | **Motor único.** `run_simulation()`: monta OpenOil (weathering NOAA), carrega readers de correntes/vento(/ondas), semeia, roda, grava `.nc` + `.png` + `_budget.npz` | app, precompute, run_ensemble, smoke tests, CLI |
| `fields_config.py` | 64 | Dicionário `CAMPOS_FIELDS`: 6 campos (lon/lat, API, óleo ADIOS, operador) | todos os scripts + app |
| `app.py` | 828 | Streamlit, 4 abas: (1) cenários pré-computados, (2) mapas de risco, (3) beaching, (4) run ao vivo. Abas 1–3 só **leem** outputs; aba 4 chama `run_simulation` | `streamlit run main/app.py` |
| `rebuild_all.ps1` | 60 | Wrapper Windows: localiza o python do env conda `opendrift` e repassa args ao orquestrador | usuário |

### Scripts de batch (todos ATIVOS, encadeados por manifests)

| Arquivo | O que faz | Produz |
|---|---|---|
| `scripts/precompute_scenarios.py` | 48 cenários = 6 campos × 4 meses × wind on/off; 500 partículas, 120 h; resumível | `outputs/scenarios/` |
| `scripts/run_ensemble.py` | 240 runs = 6 × 4 × 10 datas de início (dias 1–28); 200 partículas | `outputs/ensemble/` |
| `scripts/compute_risk_grids.py` | Agrega ensemble → prob. exposição (`prob_any`) e persistência (`prob_final`), grade 0,1° | `outputs/risk_grids/` |
| `scripts/compute_beaching.py` | Agrega ensemble → prob. de encalhe por célula + percentis de tempo. **⚠ contém o bug 🔴 #1 do DIAGNOSTICO** | `outputs/beaching/` |
| `scripts/rebuild_all.py` | Orquestrador: `--fresh/--resume/--only`; ordem scenarios→ensemble→risk→beaching (~3,5–4 h total) | tudo acima |

### Pipeline de forçantes (uso esporádico; exige credenciais CMEMS/CDS)

| Cadeia | Estado |
|---|---|
| `download_cmems_currents.py` → `prep_cmems_currents.py` (`uo/vo`→CF, camada superficial) → `inputs/currents.nc` | ATIVO |
| `download_era5_wind.py` → `prep_era5_wind.py` (`u10/v10`→`x_wind/y_wind`) → `wind.nc` → `patch_wind_cf.py` (+`standard_name`) → `inputs/wind_cf.nc` | ATIVO, mas **DUPLICADO em espírito**: prep+patch são duas metades da mesma transformação; `wind.nc` é intermediário sem outro uso |
| `download_era5_waves.py` → `prep_era5_waves.py` → `waves_cf.nc` | **NUNCA PERCORRIDO** — nem `waves_raw.nc` nem `waves_cf.nc` existem; código pronto esperando dado |

### Utilitários de debug (prefixo `_`)

| Arquivo | Estado |
|---|---|
| `scripts/_query_adios.py` | EXPERIMENTAL útil — lista óleos do catálogo ADIOS; sem efeitos colaterais |
| `scripts/_smoke_budget.py` | ATIVO (smoke test manual do budget; 24 h/50 partículas) |
| `scripts/_test_wind_off.py` | ATIVO (smoke test manual wind off; 6 h/10 partículas) |

### inputs/ (39 MB, versionado)

| Arquivo | Conteúdo | Estado |
|---|---|---|
| `currents.nc` | CMEMS 1/12°, diário, 2025 inteiro, m/s, domínio −42,5..−39,0 / −24,5..−21,0. NaN 17,9% = máscara de terra | ATIVO |
| `wind_cf.nc` | ERA5 0,25°, horário, 2025 inteiro, m/s, mesmo domínio | ATIVO |
| `currents_raw.nc`, `wind_raw.nc` | Originais pré-renomeação | ATIVOS (fonte para re-prep) |
| `wind.nc` | Intermediário entre prep e patch | DUPLICADO (regenerável) |

### outputs/ (202 MB, versionado — 921 arquivos no git)

| Pasta | Conteúdo | Estado |
|---|---|---|
| `scenarios/` | 48 `.nc` + 48 `_budget.npz` + 48 `.png` + manifest — íntegros (varredura completa) | ATIVO |
| `ensemble/` | 240 `.nc` + 240 `_budget.npz` + 240 `.png` + manifest — íntegros | ATIVO; **os 240 `.png` não são lidos por nada** (≈ maior parte dos 148 MB) |
| `risk_grids/` | 24 `.npz` + manifest | ATIVO (com viés de borda — ver DIAGNOSTICO 🟠 #4) |
| `beaching/` | 24 `.npz` + manifest | **QUEBRADO cientificamente** (🔴 #1 — números dominados por saída de domínio) |
| `openoil_run.nc`, `tracks.png` | Slot do custom run do app; cópias commitadas são de 02/06, **anteriores ao fix de oil-type** | OBSOLETO no git (regenerado a cada run da aba 4) |
| `openoil_smoketest.nc` | Nenhum código o gera ou lê (`grep -rn smoketest` → só o nome do arquivo) | **MORTO/ÓRFÃO** |
| `test_wind_off.nc`, `test_wind_off.png` | Artefato do `_test_wind_off.py`, commitado sem função | MORTO no git |

### Documentação

| Arquivo | Estado |
|---|---|
| `main/CLAUDE.md` | ATIVO, mas contém **afirmações hoje sabidamente falsas**: "2D superfície, sem mixing vertical" (mixing está LIGADO) e "beaching 0–89% sazonal … correto" (números dominados pelo bug 🔴 #1). Corrigir na Fase 8 |
| `main/README.md` | ATIVO; mesma ressalva sobre beaching |
| `docs/auditoria/` (esta pasta) | Relatórios da auditoria de 2026-07-29 |

## Grafo de dependências (sem ciclos)

```
fields_config.py ──┬────────────────────────────┐
run_open_oil.py ───┼──> precompute_scenarios ──> outputs/scenarios ──> app (aba 1)
                   ├──> run_ensemble ──────────> outputs/ensemble ─┬─> compute_risk_grids ──> app (aba 2)
                   │                                               └─> compute_beaching ───> app (aba 3)
                   └──> app (aba 4, ao vivo)
rebuild_all.py ──> importa e chama os 4 scripts de batch
```

Acoplamento app↔batch é **somente por arquivos/manifests** (nenhum import) — saudável, porém
implícito: mudança de schema nos `.npz`/manifests quebra o app sem aviso estático.
