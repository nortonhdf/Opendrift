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

## 5b. Resultados empíricos — CV interna vs holdout cego (2026-07-31)

Surrogate v1 (HGB direto) e v2 (HGB residual sobre advecção), treinados nos
720 runs de 2025 (14.400 transições de 6 h), avaliação leave-one-block-out;
depois **rollout cego de 120 h em 72 runs de 2024** (ano congelado, jamais
visto), contra a advecção passiva:

| Modelo | CV 2025 (err 6 h) | Cego 2024: LW-SS med | err 120 h med | IoU |
|---|---|---|---|---|
| Advecção passiva | 1,66 km | **0,93** | **10,9 km** | **0,16** |
| HGB direto (v1) | 1,40 km | 0,90 | 16,0 km | 0,06 |
| HGB residual (v2) | **1,35 km** | 0,91 | 15,9 km | 0,12 |
| v2 sem lon/lat (ablação) | — | 0,91 | 15,5 km | 0,10 |

**Leitura honesta:** o ganho em validação cruzada dentro do ano de treino
NÃO sobrevive ao teste cego — o viés aprendido acumula em 20 passos de
rollout e a advecção (sem viés) vence. A ablação descarta a memorização de
posição como causa principal: **a correção aprendida é específica do ano de
treino**. Nota positiva: ambos os modelos atingem LW-SS ≥0,90 (escala em que
>0,5 já é considerado bom na literatura) — o problema é vencer um baseline
físico forte, não a qualidade absoluta.

**Agenda v3 (pré-requisitos antes de reivindicar o surrogate):**
1. **Treino multi-ano** — baixar 2022–2023 (scripts prontos, aceitam o ano)
   e treinar com ≥2–3 anos; validar em ano deixado fora. É a causa provável
   dominante e a correção mais barata.
2. Features espaciais (vizinhança do campo de correntes, não só o ponto).
3. Treino com rollout/scheduled sampling (mitiga acúmulo de erro).
4. Só então: reavaliar no cego 2024 — que permanece intocado para isso.

## 5c. v3 multi-ano e o diagnóstico decisivo (2026-08-07)

### O experimento

Hipótese da v3: o fracasso da v2 no cego vinha de **um único ano de treino**.
Baixamos 2022 e 2023 (240 runs de treino cada), formando um dataset de
**24.328 transições de 1.200 runs em 3 anos** (2022, 2023, 2025).

> Nota de proveniência: 2022 vem da reanálise GLORYS (`my`) porque o produto
> de análise `anfc` só começa em meados de 2022 — e a API do CMEMS **recorta
> silenciosamente** o pedido em vez de falhar. Detectado porque 80 runs
> morreram com "missing variables"; hoje o `download()` verifica cobertura e
> percorre uma cadeia de fallback (commit `e397a3d2`).

### Resultado 1 — mais anos consertam o *overfitting*, mas não geram ganho

Leave-one-year-out (treina em 2 anos, rollout de 120 h no ano de fora):

| Ano deixado de fora | adv+corr (SS / err) | advecção (SS / err) |
|---|---|---|
| 2022 | 0,94 / 13,3 km | 0,95 / 11,5 km |
| 2023 | 0,89 / 20,3 km | 0,90 / 23,0 km |
| 2025 | 0,92 / 14,0 km | 0,93 / 10,7 km |

Cego 2024 (72 runs, modelo final treinado nos 3 anos), com **teste pareado
de Wilcoxon**:

| Métrica | adv+corr | advecção | p | veredito |
|---|---|---|---|---|
| erro 120 h | 10,70 km | 10,88 km | 0,78 | indistinguível |
| Liu–Weisberg SS | 0,939 | 0,934 | 0,49 | indistinguível |
| IoU | — | — | 0,88 | indistinguível |

O treino multi-ano **reparou o dano da v2** (que era claramente pior:
15,9 km) e trouxe o modelo à paridade — mas o surrogate vence em apenas
39/72 runs, o equivalente a cara-ou-coroa. **Nenhum ganho estatístico.**

### Resultado 2 — a causa: o resíduo é erro de integração, não física oculta

Se o modelo aprende `deslocamento_real − advecção_de_ponto_único`, o que
sobra é dominado por **estrutura de integral de caminho**: em 6 h o patch
percorre ~15 km, cruzando 1–2 células da grade de correntes (1/12° ≈ 9 km),
e o esquema de ponto único ignora como o campo muda ao longo do trajeto.
Features amostradas num só ponto **não podem** representar isso.

Teste: trocar o esquema numérico por **ponto-médio (RK2)** — zero parâmetros,
zero treino — nos mesmos 72 runs cegos:

| modelo | LW-SS med | err 120 h med | IoU médio |
|---|---|---|---|
| persistência | — | — | — |
| HGB direto (v1) | 0,90 | 16,0 km | 0,06 |
| HGB residual 1 ano (v2) | 0,91 | 15,9 km | 0,12 |
| HGB residual 3 anos (v3) | 0,94 | 10,7 km | 0,15 |
| advecção ponto único | 0,93 | 10,9 km | 0,16 |
| **advecção ponto-médio (RK2)** | **0,97** | **4,6 km** | **0,35** |

Redução de **57% no erro** (p = 8,6×10⁻⁷), melhor em 54/72 runs, IoU mais
que dobrado. **Uma correção numérica de duas linhas superou todos os modelos
de ML por larga margem.**

### Leitura para a defesa

O ML não falhou por falta de dados nem de capacidade: ele estava resolvendo
o **problema errado** — tentando aprender, a partir de features pontuais, um
erro de truncamento numérico que a matemática resolve exatamente. É um
resultado positivo disfarçado de negativo: sabemos agora onde está o sinal.

### Agenda v4 (reformulação, não mais dados)

1. **RK2 vira o baseline oficial** (já em `holdout.rollout_rk2`, testado).
   Qualquer surrogate futuro deve superar 4,6 km, não 10,9 km.
2. **Features espaciais obrigatórias**: estêncil do campo de correntes ao
   redor do patch (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y) e amostragem em t e t+Δt —
   exatamente a informação que o RK2 explora e o modelo atual não vê.
3. **Resíduo sobre o RK2**, não sobre o ponto único: se o ML não agregar
   nem sobre o RK2, a conclusão científica é forte — o transporte de patch
   nesta escala é advecção pura, e o resto é dispersão estocástica
   irredutível (as incertezas declaradas de 0,05/0,5 m/s).
4. Só então reavaliar no cego 2024.

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
