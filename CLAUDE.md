# Contexto do repositório (raiz)

Clone do **OpenDrift v1.14.7** usado *in-place* (NÃO pip-instalado, NÃO modificado — verificado
por diff contra o ponto de fork). O projeto real vive em **`main/`**: dispersão de óleo na Bacia
de Campos (6 campos × 4 meses), app Streamlit + 2.232 simulações pré-computadas (1,56 GB) + uma
camada de ML de previsão em nível de cenário.

## ⇒ LEIA PRIMEIRO: `docs/auditoria/ESTADO_ATUAL.md`

É o único documento que descreve o **estado de hoje**: cronologia, inventário, tabela de
afirmações publicadas com o comando que reproduz cada número, limitações declaradas e agenda.
Contexto técnico detalhado do projeto: `main/CLAUDE.md`. Registro completo da camada de ML:
`docs/auditoria/CAMADA_IA.md`.

**Branch:** todo o trabalho vive em `audit/revisao-completa`. O branch default (`main`) tem uma
versão do projeto **anterior à auditoria** — não use como referência.

## Documentos congelados (📷 fotos de 2026-07-29)

`DIAGNOSTICO.md`, `MAPA_DO_PROJETO.md`, `ARQUITETURA.md`, `PIPELINE_CIENTIFICO.md` e
`PLANO_DE_ACAO.md` são o registro da auditoria técnica e descrevem bugs **já corrigidos** e um
projeto **anterior a `main/ml/`**. São evidência datada, não diagnóstico corrente — a leitura
atual está em `ESTADO_ATUAL.md`.

Resumo do que a auditoria achou e a Fase 8 corrigiu (2026-07-30, `REGENERACAO.md`):

1. 🔴 `compute_beaching.py` hardcodava `status==1` como "encalhado", mas o código 1 varia por
   arquivo — em 48/240 significava `missing_data` (saída do domínio). → status resolvido por
   `flag_meanings` (`main/status_utils.py`); encalhe real ≈ 0, não 0–89%.
2. 🔴 As simulações rodavam com **mixing vertical LIGADO**, ao contrário da doc — o bloco que
   tentava desligar era inalcançável e usava chave inexistente. → 2D por default, exposto como
   parâmetro.
3. 🟠 Weathering a 10 °C (sem reader de SST), derrame de 1 m³ (default herdado), 15,9% das
   partículas morrendo na borda da caixa de forçantes. → SST do CMEMS, `spill_m3=10` declarado,
   caixa ampliada para lon −45..−36 / lat −27..−19; 0 saídas de domínio na regeração.
4. Campos: Jubarte (que é da Bacia do Espírito Santo) → **Papa-Terra**; as 6 coordenadas passaram
   a vir dos polígonos oficiais da ANP.

## Ambiente (Windows)

- Env real: conda `opendrift` em `%LOCALAPPDATA%\miniforge3\envs\opendrift` (Python 3.14.5,
  BLAS **openblas** — MKL crasha nativamente, exit `0xC06D007F` sem output). O Python do PATH
  NÃO tem as dependências — não usar.
- Rodar sempre da raiz do repo. App: `python -m streamlit run main/app.py`.
  Rebuild dos produtos de 120 h: `.\main\rebuild_all.ps1`.
  Arquivos de 168 h da camada de ML: `python -m main.ml.multiyear generate <ano>`.
- Testes: `python -m pytest main/tests -o addopts=""` (81 testes).

## Regras para pós-processar os NetCDF de saída

O mapeamento de `status` **varia por arquivo** — sempre ler `flag_meanings`/`flag_values` da
variável `status`; nunca assumir 1=stranded. `active` é sempre 0. `missing_data` = partícula
saiu da cobertura das forçantes (não é encalhe).

## Reprodutibilidade

As simulações são determinísticas: o OpenDrift semeia o RNG global no construtor
(`np.random.seed(seed)`, default 0) e as incertezas declaradas do projeto sorteiam dele.
Exposto como `run_simulation(random_seed=0)`. A semente precisa ir ao **construtor** do
OpenOil — `np.random.seed()` antes dele é sobrescrito e não faz nada.
