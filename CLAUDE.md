# Contexto do repositório (raiz)

Clone do **OpenDrift v1.14.7** usado *in-place* (NÃO pip-instalado, NÃO modificado — verificado
por diff contra o ponto de fork). O projeto real vive em **`main/`**: dispersão de óleo na Bacia
de Campos (6 campos × 4 meses), app Streamlit + 288 simulações pré-computadas. Contexto
detalhado do projeto: `main/CLAUDE.md`. Projeto de pesquisa acadêmica; camada de ML planejada
mas ainda inexistente.

## Auditoria técnica (2026-07-29) — LER ANTES DE MEXER

`docs/auditoria/` contém o mapa completo do projeto, verificado empiricamente:

- `MAPA_DO_PROJETO.md` — o que cada arquivo faz + o que é entulho
- `ARQUITETURA.md` — fluxo fim-a-fim (Mermaid), caminhos mortos
- `DIAGNOSTICO.md` — **achados por severidade (2 🔴, 4 🟠, 10 🟡, 5 🔵)**
- `PIPELINE_CIENTIFICO.md` — config OpenOil efetiva, unidades, catálogo
- `CAMADA_IA.md` — viabilidade da futura camada de ML + referências
- `PLANO_DE_ACAO.md` — Fase 8 (correções) com decisões já aprovadas
- `PERGUNTAS_ABERTAS.md` — decisões pendentes do autor

### Fatos críticos que invalidam partes de `main/CLAUDE.md` (não corrigido ainda)

1. 🔴 `outputs/beaching/` está **cientificamente inválido**: `compute_beaching.py` hardcoda
   `status==1` como "stranded", mas em 48/240 arquivos código 1 = `missing_data` (saída do
   domínio de forçantes). 8.870 "encalhes" reportados vs 1.702 reais; encalhe real só em Frade.
   O claim "beaching 0–89% sazonal correto" do `main/CLAUDE.md` é falso.
2. 🔴 As 288 simulações rodaram com **mixing vertical LIGADO** (`drift:vertical_mixing=True`,
   default), ao contrário do claim "2D sem mixing" — o bloco que tentava desligar
   (`run_open_oil.py:218`) é inalcançável e usa chave inexistente.
3. 🟠 Weathering a **10 °C** (fallback; não há reader de SST), derrame de **1 m³**
   (`seed:m3_per_hour` default), e 15,9% das partículas do ensemble morrem na borda da caixa de
   forçantes (3,5°×3,5°, pequena demais).

### Decisões aprovadas pelo autor para a Fase 8 (branch `audit/revisao-completa`)

- Beaching: resolver status via `flag_meanings` **e** ampliar caixa de forçantes + regerar tudo.
- Nova caixa de forçantes aprovada: **lon −45..−36, lat −27..−19**.
- Incluir SST (thetao do CMEMS) no pipeline.
- Mixing vertical: OFF por default, exposto como parâmetro.
- Volume de derrame: expor `spill_m3`; default declarado 10 m³ (tipo de vazamento não é o foco).
- Campos: **Jubarte → Papa-Terra**; corrigir lat de Roncador (−21,977); derivar as 6 coordenadas
  do shapefile ANP; classificação de óleo segue a regra 15–22° API → GENERIC MEDIUM CRUDE.
- Outputs permanecem no git; entulho aprovado para remoção (lista no MAPA_DO_PROJETO).
- Idioma: código/UI/docs de projeto padronizados em inglês (docs da auditoria ficam em PT).
- ML futuro: alvos (a) surrogate de transporte de patch + (b) estatísticas-resumo; ano 2024
  reservado como holdout; métrica principal Liu–Weisberg SS + IoU (ver CAMADA_IA.md).
- Nenhuma correção foi aplicada ainda — os outputs atuais são a base de comparação; não apagar.

## Ambiente (Windows)

- Env real: conda `opendrift` em `%LOCALAPPDATA%\miniforge3\envs\opendrift` (Python 3.14.5,
  BLAS **openblas** — MKL crasha, ver `main/CLAUDE.md`). O Python do PATH (3.14.3) NÃO tem as
  dependências — não usar.
- Rodar sempre da raiz do repo. App: `python -m streamlit run main/app.py`.
  Rebuild: `.\main\rebuild_all.ps1` (ou `python main/scripts/rebuild_all.py`).
- `environment.yml` diverge do env real (ex.: xarray pin ≤2025.9.0 vs 2026.4.0 instalado) —
  alinhamento previsto na Fase 8 (item 2.2).
- Testes automatizados de `main/`: **não existem ainda** (previstos na Fase 8).

## Regras para pós-processar os NetCDF de saída

O mapeamento de `status` **varia por arquivo** — sempre ler `flag_meanings`/`flag_values` da
variável `status`; nunca assumir 1=stranded. `active` é sempre 0. `missing_data` = partícula
saiu da cobertura das forçantes (não é encalhe).
