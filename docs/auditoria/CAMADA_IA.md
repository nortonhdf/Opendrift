# CAMADA DE IA — estado real e estudo de viabilidade

> Decisão de escopo (autor, 2026-07-29): a camada de ML **ainda não existe** e será construída
> após a limpeza. Este documento registra o que o código/dados atuais suportam e as referências
> externas relevantes — identificação, não construção.

## 1. Estado real hoje

**Não há ML neste repositório.** `grep -i "sklearn|torch|tensorflow|keras|xgboost|\.fit\(|predict\("`
em `main/` → zero ocorrências. Não há dataset de treino, modelo, métrica ou split. A "resposta
rápida" atual é **lookup de cenários pré-computados** via manifests (abordagem legítima de
consulta, ganho medido ~3.000×, mas sem generalização: só responde exatamente os 48+240 casos
computados).

Consequência: as perguntas clássicas desta fase (vazamento de split, baselines, seeds) **não têm
objeto hoje** — viram requisitos de projeto abaixo.

## 2. O que os dados atuais sustentam (e o que não)

| Alvo possível | Dados disponíveis | Veredito |
|---|---|---|
| Emulador de **mapa de risco** P(célula) = f(campo, mês, …) | 24 grids (6×4), 1 ano | ❌ 24 amostras não treinam nada |
| Emulador de **estatísticas-resumo** (encalha? %, quando?) | 288 runs, 6 sítios fixos, 1 volume, 1 ano | ⚠ Só após ampliar o espaço de parâmetros (posição contínua, volume, datas/anos variados) |
| **Surrogate de transporte** (estado do patch + forçantes locais → deslocamento/espalhamento em Δt) | ~86.400 trajetórias-partícula × ~240 passos ⇒ **milhões de transições** (features u,v,vento já em `inputs/`) | ✅ Único alvo com volume de dados real hoje |
| Interface conversacional sobre o catálogo | 48 cenários + manifests | ✅ Engenharia, não ML |

## 3. Referências externas (projetos análogos)

- Surrogate profundo para **transporte de patches de partículas** em ambiente costeiro — prediz
  deslocamento+espalhamento por período e acopla a um modelo lagrangiano simplificado; predição
  em segundos, 1–2 ordens de magnitude mais rápido que o solver
  ([Marine Pollution Bulletin, 2024](https://www.sciencedirect.com/science/article/pii/S0025326X24012281)).
- **LSTM treinada em simulações numéricas** para espalhamento de óleo (LSFO) sob vários ventos
  ([Marine Pollution Bulletin, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0025326X24003333)).
- **Redes de deriva lagrangiana** (DriftNet e sucessores) que ingerem campos geofísicos e
  predizem trajetórias na superfície do mar
  ([AIES, 2025](https://journals.ametsoc.org/view/journals/aies/4/3/AIES-D-24-0052.1.xml)).
- Panoramas de **IA para gestão de derrames** (famílias dominantes: emuladores data-driven e
  PINNs) — [Marine Environmental Research, 2026](https://www.sciencedirect.com/science/article/pii/S0141113626002771)
  e [Environments, 2025](https://doi.org/10.3390/environments12040132).
- Emulação neural de hazard geofísico com ganho de ~4 ordens de magnitude (análogo metodológico
  de "emular o solver caro") — [arXiv 2512.16221](https://arxiv.org/pdf/2512.16221).

## 4. Riscos de validade a projetar ANTES de treinar

1. **Vazamento por correlação temporal/espacial** (o risco nº 1 deste projeto): membros do
   mesmo (campo, mês) compartilham as mesmas forçantes com dias de defasagem; partículas do
   mesmo run são quase-duplicatas. **Split obrigatório por bloco** — por (campo×mês) ou por mês
   inteiro fora do treino; nunca por partícula/timestep. Ideal: teste em **ano não visto**
   (requer baixar outros anos — scripts já aceitam `python download_*.py 2024`).
2. **Baselines obrigatórios** (todos computáveis com o acervo atual, sem ML):
   persistência (patch parado), advecção pura pelas correntes, vizinho-mais-próximo no catálogo
   (= o app de hoje). O surrogate só tem valor se vencer os três.
3. **Herança de vieses**: qualquer modelo treinado nos outputs atuais aprende os artefatos da
   auditoria (weathering a 10 °C, mixing indocumentado, 16% de partículas mortas na borda,
   beaching falso). **Pré-requisito duro: corrigir 🔴 1/2 e 🟠 3/4 e regerar ANTES de gerar o
   dataset de treino.**
4. **Reprodutibilidade**: runs OpenDrift são deterministas (verificado), mas o pipeline de ML
   precisará de seeds explícitos, versionamento de dataset (hash dos .nc de origem) e o mesmo
   pré-processamento em treino e inferência — registrar desde o primeiro commit da camada.

## 5. Recomendação de sequência (quando chegar a hora)

1. Corrigir/regerar a base física (Fase 8 desta auditoria).
2. Definir o alvo: **surrogate de transporte de patch** é o único compatível com os dados já
   existentes; estatísticas-resumo exigem novo plano de amostragem (LHS sobre posição, data,
   volume — os scripts de batch são facilmente generalizáveis para isso).
3. Congelar um dataset versionado (features: u, v, vento interpolados na posição; alvo:
   deslocamento do centróide + dispersão do patch por Δt=1800 s ou maior).
4. Baselines → modelo simples (GBM/MLP) → só depois arquiteturas de sequência/campo
   (LSTM/U-Net/GNN), sempre com split por bloco e teste em ano não visto.

## 6. Decisões tomadas pelo autor (2026-07-29)

- **Alvos**: (a) surrogate de transporte de patch **e** (b) estatísticas-resumo por cenário —
  (b) condicionado a novo plano de amostragem (LHS sobre posição/volume/data).
- **Holdout**: ano **2024** inteiro reservado como teste (baixar junto com a regeração da
  Fase 8; nunca entra no treino).
- **Métrica principal** (pesquisa da auditoria): **Liu–Weisberg Skill Score** (separação
  lagrangiana cumulativa normalizada — padrão da literatura desde o Deepwater Horizon:
  [Liu & Weisberg 2011](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JC006837),
  [recomendações Frontiers 2021](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.630388/full))
  para trajetória/centróide, + **IoU/FSS** na grade 0,1° para a forma da mancha, sempre
  reportados como ganho sobre os baselines. Para o alvo (b): MAE em fração encalhada e
  **Brier score** nas probabilidades por célula.
