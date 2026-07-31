# REGENERAÇÃO DOS PRODUTOS — 2026-07-30

> Registro da regeneração completa pós-Fase 8: incidentes, validação final e
> comparação com a geração antiga. Complementa DIAGNOSTICO.md e PLANO_DE_ACAO.md.

## O que rodou

Forçantes novas (CMEMS correntes+**SST** e ERA5 vento, caixa −45..−36 / −27..−19,
2025 completo) → 48 cenários → 240 membros de ensemble → 24 risk grids →
24 beaching grids, tudo com o código corrigido da branch `audit/revisao-completa`
(status por `flag_meanings`, 2D sem mixing vertical, RK4, 10 m³, coordenadas ANP,
Papa-Terra no lugar de Jubarte).

## Incidentes durante a regeneração (ambos corrigidos e testados)

1. **`NetCDF: HDF error` + segfault (exit 139)**: aberturas repetidas do novo
   `wind_cf.nc` (HDF5) no mesmo processo começaram a falhar após ~8 runs e
   derrubaram o batch. Correções: inputs regravados em **NETCDF3_64BIT** (sem
   camada HDF5; preps agora gravam nesse formato por padrão) e reader de arquivo
   existente que falha ao carregar agora **aborta na hora** (commit `9044925a`).
2. **Mistura de gerações via `--fresh` parcial**: o `--fresh` antigo apagava o
   manifest de cada estágio só quando o estágio começava; com o crash no meio
   dos cenários, o manifest do ensemble sobreviveu e o `--resume` seguinte
   reaproveitou 200 membros da geração antiga ao lado de 40 novos. Correções:
   manifests de todos os estágios selecionados são apagados **no início**
   (commit `5ef2dd56`, testado); poda determinística por conteúdo (volume
   10 m³) e recomputação limpa dos 200 membros.

## Validação final (varredura abrindo todos os arquivos)

| Métrica | Geração antiga | Geração nova |
|---|---|---|
| Arquivos íntegros | 288/288 | **288/288** (48+240, manifests 100% consistentes) |
| Volume por run | 1 m³ (default silencioso) | **10,00 m³ em 288/288** |
| `missing_data` (saída de domínio) | 14–16% das partículas | **0,00%** |
| Falhas de run | — | **0** |
| Evaporação final média (ensemble) | 17,9% (a 10 °C) | **27,5%** (SST real; faixa 13,5–32,4%) |
| Encalhe reportado vs real | 8.870 vs 1.702 (falsos positivos de borda) | **0 vs 0 — consistentes** |
| App headless (4 abas) | 0 exceções | **0 exceções** |

## Resultado científico novo: encalhe zero em 120 h

Na geração antiga, o único encalhe real ocorria em **Frade** — cuja coordenada
estava **~118 km a oeste da posição oficial** (muito mais perto da costa). Com
as 6 posições oficiais ANP/EPE, **nenhuma partícula encalha em 120 h em nenhum
dos 24 (campo × mês)**. O "beaching 0–89% sazonal" anterior era uma cadeia de
artefatos (saída de domínio contada como encalhe + posição errada de Frade).
A aba Beaching do app agora mostra corretamente "negligible beaching" em todos
os casos. Se encalhe for um produto de interesse do projeto, as opções são:
janela >120 h, cenários costeiros dedicados, ou aceitar o resultado (que é
defensável: os 6 campos ficam a ~80–200 km da costa).

## Convergência do ensemble (experimento autorizado, item 2.9)

`prob_any` com 5 vs 10 membros (3 pares campo×mês): IoU 0,63–0,73; diferença
máxima de probabilidade por célula 0,30. **10 membros ainda não convergem** —
os mapas de risco são indicativos, não estatisticamente estáveis. Recomendação
antes de uso quantitativo/ML: ≥20–30 membros por (campo, mês) — custo ~2× a
4,5× o ensemble atual (152 min) — ou declarar explicitamente a incerteza.

## Atualização 2026-07-31 — ensemble 28 membros/mês + primeira IA

**Ensemble ampliado** (decisão pós-convergência): 672 runs (28 inícios diários,
dias 1–28, × 6 campos × 4 meses), 6,1 h, **0 falhas, 0 saídas de domínio,
10 m³ em 672/672**. Convergência melhorou: IoU 14↔28 = **0,78–0,82**
(era 0,63–0,73 em 5↔10), Δprob máx 0,14–0,18 (era 0,30).

**Encalhe real detectado**: a amostragem diária revelou o que a de 3 em 3 dias
perdia — `papa-terra_jan` com **3,41%** de partículas encalhadas (3 células
costeiras; campo mais próximo da costa). Demais 23 blocos: 0%. O produto de
beaching agora carrega sinal físico genuíno, raro e localizado.

**Camada de IA (main/ml/)** — surrogate de transporte de patch (alvo a),
avaliação leave-one-block-out (24 blocos campo×mês), horizonte 6 h,
dataset de 14.400 amostras (720 runs):

| modelo | erro médio de deslocamento | MAE espalhamento |
|---|---|---|
| persistência | 9,70 km | 0,08 |
| advecção passiva (correntes+3% vento) | 1,66 km | 0,08 |
| vizinho-mais-próximo | 2,65 km | 0,05 |
| **HGB (surrogate v1)** | **1,40 km** | **0,04** |

Artefato + metadados (hash do dataset, seed, sklearn, tabela completa) em
`main/outputs/ml/`. Próxima geração: features espaciais, horizonte
multi-passo, teste cego no holdout 2024.

## Pendências

- **2024 (holdout do ML)**: download aprovado em princípio, aguardando o "sim"
  final do autor. Scripts aceitam o ano como argumento; renomear os raw por ano
  fica para a fase de ML.
- PNGs de cenário/ensemble não são mais gerados (shapefile de costa do cartopy
  fora do ar — 404) nem versionados; o app não os usa.
