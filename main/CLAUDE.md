# Projeto — Dispersão de óleo na Bacia de Campos (OpenDrift)

> Contexto portátil do projeto (viaja no git, carrega automaticamente em qualquer máquina).
> Consolidado em 2026-06-02 a partir das memórias locais do Claude.

## Visão geral

Modelagem de dispersão de óleo na **Bacia de Campos**, construída sobre o **OpenDrift v1.14.7**
(usado *in-place*, **NÃO** pip-instalado). Todo o código customizado vive em `main/`.

**Componentes:**
- `main/app.py` — app **Streamlit** com 4 abas: cenários pré-computados, mapas de risco
  (ensemble), run customizado ao vivo, e beaching/encalhe. Plotly + mapbox open-street-map.
- `main/fields_config.py` — 6 campos de óleo (Peregrino, Marlim, Roncador, Jubarte, Frade,
  Albacora) com lon/lat/API/tipo de óleo ADIOS.
- `main/run_open_oil.py` — runner OpenOil (`run_simulation(...)`). 2D superfície, sem mixing
  vertical, weathering NOAA. Exporta lon/lat/status/z/massa + sidecar `_budget.npz`.
- `main/scripts/` — download (CMEMS correntes, ERA5 vento/ondas), prep/patch (renomeia vars p/
  convenções CF), `precompute_scenarios.py` (48 cenários), `run_ensemble.py` (240 runs),
  `compute_risk_grids.py` (24 grids de prob.), `compute_beaching.py` (24 grids de encalhe),
  `rebuild_all.py` (orquestrador único de todos os estágios).

## Como rodar

- **Env conda** `opendrift` (miniforge, Python 3.14):
  `C:\Users\nfreitas\AppData\Local\miniforge3\envs\opendrift\python.exe`.
  OpenDrift importa OK in-place. Recriar via `environment.yml`.
- Comandos com caminhos relativos (`main\inputs\...`) exigem **cwd = raiz do repo** `Opendrift/`.
- App: `streamlit run main/app.py` a partir da raiz.

### ⚠️ Fix crítico de ambiente — BLAS/MKL (não esquecer ao recriar o env)

Em build py3.14, `numpy` linkado contra **Intel MKL** crashava nativamente em QUALQUER simulação
com `Windows fatal exception: code 0xc06d007f` (entry point ausente em DLL MKL/TBB) — até
`np.dot`/`lstsq` quebravam. Trocar versão do MKL **não** resolve. Correção (só ambiente):

```
conda install -n opendrift -c conda-forge "blas=*=openblas" --force-reinstall -y
```

Troca o backend para **OpenBLAS**; depois disso as simulações OpenOil rodam OK. Se voltar a
crashar nativamente, suspeitar de BLAS/MKL de novo (diagnóstico: `PYTHONFAULTHANDLER=1`).

## Dados de entrada (`main/inputs/`, versionados no git, ~39 MB)

- Correntes diárias e vento horário cobrindo o **ano inteiro de 2025**
  (`currents.nc`, `wind_cf.nc` + os `_raw`).
- **Ondas (`waves_cf.nc`) NÃO existem** — o toggle de Stokes drift no app exige rodar
  `download_era5_waves.py` + `prep_era5_waves.py` antes.

## Estado atual e PENDÊNCIA principal

**Pipeline existente está 100% computado, porém STALE** (foi gerado com óleo default e sem oil
budget). Os 48 cenários + 240 ensemble + 24 risk grids ainda precisam de **re-run em lote** para
refletir: tipo de óleo por campo, oil budget (sidecar `_budget.npz`) e demais correções.

### Re-run em lote (orquestrador pronto e validado end-to-end)

Da **raiz do repo**, no env `opendrift`:
```
.\main\rebuild_all.ps1                            # mostra o plano (não muda nada)
.\main\rebuild_all.ps1 --fresh                    # rebuild completo (~3,5–4h)
.\main\rebuild_all.ps1 --fresh --only scenarios   # só os 48 cenários (~47 min)
.\main\rebuild_all.ps1 --resume                   # continua run interrompido
```
Estágios em ordem: **scenarios → ensemble → risk → beaching**. 1 cenário ≈ 59s.
Para background numa sessão do Claude: usar `run_in_background` do Bash/PowerShell com `--fresh`.

Sinal de que o rebuild rodou: aparecem sidecars `*_budget.npz` em `main/outputs/scenarios/`
(hoje há **0** — confirma que ainda está stale).

## Correções já aplicadas e validadas (sessão 2026-06-02)

- Bug `resolve_oil_type` (API morta do `adios_db`) corrigido → tipo de óleo por campo aplicado.
- Bug `st.toggle` sem `key` (quebrava no Streamlit 1.58 com 2 abas com dados) → key único.
- Ondas/Stokes via `drift:use_tabularised_stokes_drift` (do vento, sem ERA5).
- Oil budget: `run_open_oil` exporta massa + `_budget.npz`; painel `show_budget` no app.
- Beaching: `compute_beaching.py` → 24 grids em `outputs/beaching/` + aba no app (0–89%, sazonal).
- Scripts em lote fazem `sys.stdout.reconfigure(encoding="utf-8")` (consoles Windows cp1252
  crashavam nos glifos ✓/✗/→).
- App validado headless via `streamlit.testing.v1.AppTest` (4 abas, zero exceções).
- `streamlit`+`plotly` adicionados ao `environment.yml`.

## Possíveis próximos passos

Deploy do Streamlit; oil budget também para os cenários (depende do re-run); waves nos cenários
pré-computados (hoje só wind on/off, waves off).
