# Projeto — Dispersão de óleo na Bacia de Campos (OpenDrift)

> Contexto portátil do projeto (viaja no git, carrega automaticamente em qualquer máquina).
> Atualizado em 2026-06-03.

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

- **Env conda** `opendrift` (miniforge, Python 3.14). Recriar via `environment.yml`.
- `rebuild_all.ps1` é **portátil** — busca `python.exe` do env `opendrift` em locais padrão
  (`%USERPROFILE%\miniforge3`, `%LOCALAPPDATA%\miniforge3`, etc.) sem precisar de ativação.
- Comandos com caminhos relativos (`main\inputs\...`) exigem **cwd = raiz do repo** `Opendrift/`.
- App: `streamlit run main/app.py` a partir da raiz.

## ⚠️ Fixes críticos de ambiente (não esquecer ao recriar o env)

### 1 — BLAS/MKL crash (Windows, Python 3.14)

Em build py3.14, `numpy` linkado contra **Intel MKL** crasha nativamente em QUALQUER simulação
com `Windows fatal exception: code 0xc06d007f` (entry point ausente em DLL MKL/TBB).

**Sintoma:** exit code 0xC06D007F, stdout/stderr completamente vazios — crash silencioso.

**Causa:** mesmo com `libblas=*=*openblas` no `environment.yml`, o conda pode resolver para
o build MKL. Verificar: `conda list -n opendrift blas` — build string deve ter `openblas`, NÃO `mkl`.

**Correção (obrigatória após criar o env):**
```
conda install -n opendrift -c conda-forge "blas=*=openblas" --force-reinstall -y
```

### 2 — Matplotlib figure accumulation (batch rebuild)

Em batch de 48+ simulações no mesmo processo Python, figuras matplotlib não fechadas acumulam
memória e causam crash nativo (STATUS_ACCESS_VIOLATION, exit code 5 no PowerShell).

**Correção** já aplicada em `run_open_oil.py`: `plt.close("all")` após cada `o.plot()`.

### 3 — Captura de output do subprocess

Ao chamar `powershell -File script.ps1` como subprocess com `*>` redirect, a saída de
processos nativos (Python) pode não ser capturada. Usar sempre o `python.exe` diretamente:
```powershell
& "path\to\python.exe" "main\scripts\rebuild_all.py" --fresh *> rebuild_all.log
```

## Dados de entrada (`main/inputs/`, versionados no git, ~39 MB)

- Correntes diárias e vento horário cobrindo o **ano inteiro de 2025**
  (`currents.nc`, `wind_cf.nc` + os `_raw`).
- **Ondas (`waves_cf.nc`) NÃO existem** — o toggle de Stokes drift no app exige rodar
  `download_era5_waves.py` + `prep_era5_waves.py` antes.

## Estado atual dos outputs (2026-06-03)

### O que está PRONTO ✓
- **48 cenários** (`main/outputs/scenarios/`): todos re-computados com oil-type correto por campo
  e com sidecar `_budget.npz`. Manifest atualizado. Prontos para o app.

### O que ainda precisa ser rodado ✗
- **240 ensemble** (`main/outputs/ensemble/`): os .nc existem mas são **stale** (gerados sem
  oil-type por campo e sem budget). Manifest foi deletado pelo `--fresh`. Precisam re-run.
- **24 risk grids** (`main/outputs/risk_grids/`): vazios — dependem do ensemble atualizado.
- **24 beaching grids** (`main/outputs/beaching/`): vazios — dependem do ensemble atualizado.

### Como continuar o rebuild na próxima sessão

Da **raiz do repo** (`Opendrift/`), rodar os estágios restantes:

```powershell
# Mostra o plano sem alterar nada:
.\main\rebuild_all.ps1

# Rebuild só do ensemble + grids dependentes (~3 h):
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONUTF8 = "1"
& "C:\Users\<user>\miniforge3\envs\opendrift\python.exe" `
    "main\scripts\rebuild_all.py" --resume --only ensemble,risk,beaching `
    *>> "main\rebuild_all.log"
```

Ou usando o wrapper portátil (mas redirecionar output para log, não via `powershell -File`):
```powershell
# EVITAR: powershell -File main\rebuild_all.ps1  (captura de output não confiável)
# PREFERIR: python.exe direto como acima
```

Para background numa sessão do Claude: usar `run_in_background` do PowerShell com o comando
`python.exe` direto (NÃO `powershell -File`).

**Rebuild completo do zero** (se necessário):
```powershell
.\main\rebuild_all.ps1 --fresh         # ~3,5–4h total
.\main\rebuild_all.ps1 --resume        # continua interrompido
```

Estágios em ordem: **scenarios → ensemble → risk → beaching**. 1 cenário ≈ 59s, 1 ensemble ≈ 45s.

## Correções já aplicadas e validadas

- Bug `resolve_oil_type` (API morta do `adios_db`) corrigido → tipo de óleo por campo aplicado.
- Bug `st.toggle` sem `key` (quebrava no Streamlit 1.58 com 2 abas com dados) → key único.
- Ondas/Stokes via `drift:use_tabularised_stokes_drift` (do vento, sem ERA5).
- Oil budget: `run_open_oil` exporta massa + `_budget.npz`; painel `show_budget` no app.
- Beaching: `compute_beaching.py` → 24 grids em `outputs/beaching/` + aba no app (0–89%, sazonal).
- Scripts em lote fazem `sys.stdout.reconfigure(encoding="utf-8")` (consoles Windows cp1252).
- App validado headless via `streamlit.testing.v1.AppTest` (4 abas, zero exceções).
- `streamlit`+`plotly` adicionados ao `environment.yml`.
- `rebuild_all.ps1` portátil: busca `python.exe` em locais padrão de conda sem path hardcoded.
- `plt.close("all")` após cada plot em `run_simulation` (evita crash por acumulação de memória).

## Possíveis próximos passos

1. **Concluir o rebuild** (ensemble + risk + beaching) — `--resume --only ensemble,risk,beaching`.
2. Deploy do Streamlit.
3. Oil budget também para os cenários (já feito) — confirmar que app exibe o painel corretamente.
4. Waves nos cenários pré-computados (hoje só wind on/off, waves off).
