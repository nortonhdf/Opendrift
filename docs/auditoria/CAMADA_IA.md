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

## 5d. v4 — reformulação para o objetivo real, e o primeiro ganho robusto (2026-08-07)

### A reformulação

O autor explicitou o objetivo final: **dado local, tipo de óleo, estação e as
condições de corrente dos últimos dias/semanas/meses, projetar a mancha em
D+1, D+2, D+5, D+14** — sem conhecer as forçantes futuras.

Isso expõe por que v1–v3 estavam numa disputa perdida: o surrogate recebia a
forçante *verdadeira* a cada passo, logo competia com integração numérica —
e numérica ganha (§5c). No problema real **não há campo futuro para
integrar**, então a advecção nem é candidata. Os concorrentes honestos passam
a ser climatologia, persistência dos antecedentes e análogo histórico.

| | v1–v3 | **v4** |
|---|---|---|
| Entrada | estado do patch + forçante instantânea | local, óleo, estação, correntes antecedentes (3/7/30/90 d) |
| Saída | deslocamento em 6 h, iterado | mancha em D+1…D+5 direto |
| Concorrente | integração numérica | climatologia / persistência / análogo |

Dataset: **1.200 cenários** (2022, 2023, 2025) × 36 features causais × 16
alvos; holdout cego de 72 cenários em 2024 (`main/ml/scenario.py`).
**Regra de causalidade**: toda feature vem de dados estritamente anteriores
ao instante do derrame; janela indisponível vira NaN (consumido nativamente
pelo HistGradientBoosting), nunca zero — testado em
`test_antecedent_features_are_strictly_causal`.

### Resultado 1 — cego 2024, local já conhecido

Erro mediano de posição do centróide (km):

| modelo | D+1 | D+2 | D+3 | D+5 |
|---|---|---|---|---|
| climatologia (campo×estação) | 24,6 | 46,7 | 64,3 | 91,7 |
| persistência (corrente de 7 d) | 25,4 | 48,8 | 72,0 | 106,7 |
| análogo histórico | 33,7 | 65,6 | 89,4 | 122,3 |
| **HGB** | **18,5** | **36,4** | **53,4** | **91,3** |

O HGB é o melhor em todos os horizontes, mas contra a climatologia a
diferença **não** é significativa (Wilcoxon pareado: p = 0,073 em D+1;
0,80 em D+5) — com n = 72 falta poder estatístico, e em D+5 a
previsibilidade a partir das condições iniciais já se esgotou. Contra o
análogo, é significativo em todos os horizontes (p < 0,001).

Detalhe importante: aqui a climatologia é um baseline **artificialmente
forte**, porque só existem 6 locais fixos e ela já viu aquele campo em
outros anos. Num vazamento em posição arbitrária — o caso de uso real —
essa tabela não existe.

### Resultado 2 — local NUNCA visto: o ganho robusto

Leave-one-field-out (treina em 5 campos, prevê no 6º; a climatologia degrada
para média sazonal entre os outros campos, exatamente como na prática):

| campo retido | D+1 | D+2 | D+3 | D+5 |
|---|---|---|---|---|
| Albacora | 17,2 / 26,7 | 34,1 / 45,1 | 43,7 / 59,0 | 60,4 / 84,8 |
| Frade | 14,9 / 29,4 | 28,1 / 50,0 | 41,9 / 66,9 | 56,9 / 96,8 |
| Marlim | 19,5 / 30,8 | 34,1 / 47,4 | 50,1 / 60,0 | 70,5 / 79,9 |
| Papa-Terra | 17,7 / 25,8 | 33,1 / 46,1 | 52,9 / 64,2 | 66,3 / 94,1 |
| Peregrino | 18,3 / 23,1 | 41,2 / 46,5 | 56,9 / 66,3 | 89,4 / 101,9 |
| Roncador | 11,8 / 26,0 | 23,9 / 46,7 | 37,5 / 66,5 | 53,0 / 98,4 |

(HGB / climatologia-sazonal, erro mediano em km — **HGB vence nas 24 células**)

Teste pareado agregando os 1.200 cenários retidos:

| horizonte | HGB | climatologia | ganho | vitórias | p |
|---|---|---|---|---|---|
| D+1 | 16,5 km | 26,9 km | **+39%** | 914/1200 | ~0 |
| D+2 | 31,7 km | 47,1 km | **+33%** | 880/1200 | ~0 |
| D+3 | 48,0 km | 63,9 km | **+25%** | 850/1200 | ~0 |
| D+5 | 64,7 km | 92,6 km | **+30%** | 860/1200 | ~0 |

**Este é o primeiro ganho estatisticamente robusto do projeto**, e é
justamente no cenário de implantação: prever a deriva de um vazamento numa
posição que o modelo nunca viu, com 25–39% menos erro do que a melhor
alternativa disponível sem ML.

### Resultado 3 — a tendência dominante

Importância por permutação (D+5, distância percorrida), medida no cego:

| feature | impacto |
|---|---|
| **`u_mean_3d`** (corrente zonal média dos últimos 3 dias) | **5,85 km** |
| `lon` | 1,08 km |
| `u_mean_90d` | 0,83 km |
| `speed_std_30d` | 0,60 km |

A corrente zonal dos **últimos 3 dias** domina todo o resto por um fator ~5.
Ou seja: o estado recente do oceano — e não a estação, o tipo de óleo ou a
profundidade — é o que determina para onde e quão longe a mancha vai. Isso
responde diretamente ao objetivo imediato do autor ("identificar trends nos
resultados do OpenDrift") e valida a escolha de features antecedentes.

### Limitações honestas

1. **D+14 não é avaliável hoje**: o arquivo tem runs de 120 h (D+5 é o teto).
   Estender exige gerar runs de 336 h — ver agenda.
2. O ganho em D+5 já opera perto do teto de previsibilidade; a extrapolação
   para D+14 provavelmente convergirá para climatologia. Medir antes de
   prometer.
3. Espalhamento (tamanho da mancha) ainda é mal previsto por todos os
   modelos (MAE ~1,4 km para todos) — não há sinal aproveitável nas features
   atuais para essa quantidade.
4. Só 6 locais de treino: a generalização espacial foi demonstrada entre
   campos vizinhos da mesma bacia, não para geografia arbitrária.

### Agenda v5

1. **Runs de 336 h** (D+14) num subconjunto de cenários — a única forma de
   medir onde a previsibilidade realmente morre.
2. Mais locais de semeadura (grade de pontos, não só os 6 campos) para
   sustentar a generalização espacial em produção.
3. Prever a **footprint** (grade de ocupação), não só o centróide — é o que
   o app precisa desenhar.
4. Quantificação de incerteza (quantile regression) — uma mancha prevista
   sem envelope de confiança não é operacionalmente utilizável.

## 5e. v4 em D+7, com controle linear e incerteza calibrada (2026-08-11)

Decisão do autor de 2026-08-07: escopo de trabalho **até D+7**. Isso exigiu um
arquivo novo — `training168_{2022,2023,2024,2025}`, 240 runs por ano
(6 campos × 4 meses × 10 dias de início, 200 partículas, **168 h**), gerado
com o mesmo `run_open_oil.py` das forçantes já auditadas. Os arquivos antigos
de 120 h continuam intactos como registro por trás de §5a–§5d.

Duas consequências metodológicas do arquivo novo, além do horizonte:

- **treino balanceado**: 720 cenários (240 × 3 anos) em vez dos 1.200
  desbalanceados de antes (48 + 672 de 2025 + 240 + 240);
- **cego de 240 cenários** em vez de 72 — pela primeira vez há poder
  estatístico real no holdout, e ele muda a leitura de §5d.

### Resultado 1 — no local JÁ CONHECIDO, a previsibilidade morre em D+3

Erro mediano de posição do centróide no cego 2024 (240 cenários, km):

| modelo | D+1 | D+2 | D+3 | D+5 | D+7 |
|---|---|---|---|---|---|
| climatologia (campo×estação) | 25,7 | 42,8 | 55,8 | **81,6** | **105,4** |
| persistência (corrente de 7 d) | 21,3 | 41,1 | 63,3 | 111,6 | 165,6 |
| análogo histórico | 31,0 | 59,7 | 81,2 | 121,7 | 136,0 |
| ridge (controle linear) | 21,4 | 41,6 | 58,2 | 83,8 | 105,4 |
| **HGB** | **19,8** | **39,3** | 57,0 | 83,2 | 110,6 |

Wilcoxon pareado (HGB vs climatologia): significativo em D+1 (−4,4 km,
p<0,001) e D+2 (−5,1 km, p=0,003); a partir de D+3 a climatologia empata ou
ganha (+0,9 / +2,8 / +4,2 km, p = 0,85 / 0,65 / 0,12). Em §5d isso foi
atribuído a falta de poder (n=72). Com n=240 a conclusão é outra e mais
firme: **num local já conhecido, o ganho sobre a climatologia sazonal existe
até D+2 e se esgota em D+3.** Contra persistência e análogo o HGB ganha em
todos os horizontes (p<0,001).

### Resultado 2 — no local NUNCA VISTO, a não-linearidade é que decide

Leave-one-field-out, agora com o controle linear (mediana em km, 720
cenários retidos):

| horizonte | HGB | ridge | climatologia sazonal | HGB vs clim | ridge vs clim | HGB vs ridge |
|---|---|---|---|---|---|---|
| D+1 | **17,0** | 27,1 | 26,6 | **+36%** (p≈0) | −2% | p≈0 (562/720) |
| D+2 | **33,8** | 51,8 | 46,3 | **+27%** (p=8e−42) | −12% | p≈0 (546/720) |
| D+3 | **46,9** | 69,4 | 62,6 | **+25%** (p=4e−40) | −11% | p≈0 (538/720) |
| D+5 | **69,8** | 101,3 | 91,9 | **+24%** (p=2e−39) | −10% | p≈0 (533/720) |
| D+7 | **82,5** | 138,4 | 113,8 | **+28%** (p=2e−40) | −22% | p≈0 (563/720) |

O HGB vence nas 30 células campo × horizonte. E o controle linear **perde
para a própria climatologia** em todos os horizontes.

Esse contraste é o achado desta rodada, e responde à pergunta que o controle
linear existia para fazer:

> No local conhecido, ridge ≈ HGB (p = 0,07 a 0,74) — ali a não-linearidade
> não paga nada. No local novo, ridge colapsa e o HGB mantém 24–36% de ganho.
> Ou seja: a relação entre **posição** e deriva é fortemente não-linear no
> espaço; um modelo linear não a transporta para um ponto que não viu.
> A não-linearidade não serve para ajustar melhor — serve para **generalizar
> espacialmente**, que é exatamente o caso de uso do projeto.

### Resultado 3 — incerteza: o envelope cru estava quebrado; conformal conserta

Medido no cego, o envelope P10–P90 dos quantile boosters cobria **35–49%**
contra 80% nominal — quantile trees ajustadas não são calibradas, e um ano
novo destrói a promessa. Correção aplicada: **CQR** (split-conformal de
Romano, Patterson & Candès 2019), com o alargamento estimado num **ano
inteiro deixado de fora** (ajuste em 2022+2023, calibração em 2025) — não num
split aleatório, porque cenários do mesmo ano compartilham o estado do oceano
e vazariam.

| horizonte | cobertura crua | largura crua | cobertura conformal | largura conformal | alargamento |
|---|---|---|---|---|---|
| D+1 | 35% | 22 km | **84%** | 60 km | +19 km/lado |
| D+2 | 36% | 39 km | **85%** | 109 km | +35 km/lado |
| D+3 | 41% | 58 km | **84%** | 151 km | +47 km/lado |
| D+5 | 49% | 87 km | **89%** | 237 km | +75 km/lado |
| D+7 | 36% | 99 km | **86%** | 306 km | +104 km/lado |

A calibração é honesta em ambas as direções: ela quase triplica a largura da
banda. Um envelope de ±150 km em D+7 é grande, mas é o tamanho real da
incerteza — a banda de 99 km que o modelo cru anunciava era ficção.

### Resultado 4 — a tendência dominante muda com o horizonte

Importância por permutação (D+7, distância, medida no cego):

| feature | impacto |
|---|---|
| `u_mean_3d` (corrente zonal, últimos 3 dias) | 4,98 km |
| `lon` | 4,22 km |
| `sst_mean_30d` | 1,63 km |
| `u_mean_90d` | 0,82 km |

Em D+5 (§5d) `u_mean_3d` dominava `lon` por ~5×; em D+7 os dois praticamente
empatam. Faz sentido físico: quanto mais longo o horizonte, menos o estado
recente do oceano explica e mais pesa **onde** o vazamento aconteceu — o que
é a mesma leitura do Resultado 2.

### Limitações que permanecem

1. Espalhamento continua não previsível (MAE 1,28–1,38 km para todos os
   modelos, HGB inclusive) — não há sinal nas features atuais.
2. D+14 continua fora de alcance: o arquivo agora vai até 168 h.
3. Ainda 6 locais de treino; a generalização foi demonstrada entre campos da
   mesma bacia.
4. A cobertura conformal fica em 84–89% contra 80% nominal — conservadora,
   como esperado ao calibrar num ano e aplicar em outro.

### Agenda v6

1. **Footprint** (grade de ocupação), não só centróide — é o que o app
   precisa desenhar, e é onde IoU/Brier voltam a ser as métricas certas.
2. Mais locais de semeadura (grade, não só os 6 campos) — hoje é a limitação
   que mais restringe a afirmação de generalização espacial.
3. Expor a previsão + envelope conformal no app (aba de previsão).
4. D+14 só depois de runs de 336 h; medir antes de prometer.

## 5f. Footprint — prever a área oleada, não só o centróide (2026-08-25)

Item 1 da agenda v6. Código: `main/ml/footprint.py` (alvos) e
`main/ml/footprint_forecast.py` (modelos + avaliação). Nenhuma simulação
nova: usa os mesmos arquivos de 168 h.

### Decisão 1 — o alvo é a área VARRIDA, e isso foi medido, não escolhido

A primeira coisa que o arquivo respondeu foi que o alvo óbvio não serve.
Medida no arquivo de 168 h, a mancha instantânea é minúscula:

| | D+1 | D+2 | D+3 | D+5 | D+7 |
|---|---|---|---|---|---|
| espalhamento RMS (km) | 0,35 | 0,55 | 0,80 | 1,22 | 1,25 |
| células ocupadas no instante | 1 | 1,5 | 1,5 | 2 | 2 |
| células varridas até o horizonte | 8 | 13 | 18 | 29 | 39 |

Na grade de 0,1° do app (11,1 km) o *snapshot* ocupa 1–2 células: prever isso
seria prever o centróide outra vez, com uma grade em volta. O que tem forma é
a área **cumulativa** — e é também o que a aba de risco já desenha como
`prob_any`. Ambas ficam no dataset; a varrida é o alvo.

### Decisão 2 — referencial relativo ao vazamento, células isotrópicas

Deslocamentos em km a partir do ponto de vazamento, células de 11,132 km
(0,1° de latitude), quadro de ±501 km. **Zero partículas saíram do quadro**
nos 960 cenários. A invariância a translação é o ponto: um modelo que lê
lon/lat absolutos não transfere para um local que nunca viu, que é o caso de
uso (§5e, Resultado 2).

### Os competidores

Todo modelo devolve **probabilidade por célula** — um único traçado
determinístico não é produto utilizável quando o próprio centróide erra
~80 km em D+7 (sete células).

| modelo | o que é |
|---|---|
| climatologia | frequência da célula por estação, no referencial relativo |
| persistência | corredor ao longo da corrente média dos 7 dias anteriores |
| análogo | frequência entre os k=25 cenários históricos mais parecidos |
| corredor do centróide | isotônica sobre a distância ao caminho previsto pelo v4 |
| **pluma** | mesmo caminho do v4, mas com a **forma medida** (núcleo empírico 2D em coordenadas ao-longo/transversal, normalizadas pelo deslocamento) |
| ocupação | classificador direto por célula (features do cenário + offset da célula, incluindo as coordenadas giradas para o eixo da corrente) |

Protocolo, igual para todos: ajuste em 2022+2023, forma do corredor/pluma e
ponto de operação do IoU calibrados no ano retido 2025, avaliação no cego
2024 e em leave-one-field-out. Os modelos sem estrutura de dois estágios
(climatologia, análogo, ocupação) são ajustados em ajuste+calibração, para
que **todos consumam os mesmos 720 cenários** — o ponto de operação deles sai
de dados que já viram, o que os favorece, não à pluma; e a métrica principal
(Brier) não usa limiar nenhum.

### Métricas — e por que elas discordam de propósito

- **Brier / BSS**: calibração + nitidez contra o desfecho individual.
- **IoU@limiar**: a leitura determinística, no ponto de operação escolhido na
  calibração, nunca no teste.
- **Área de captura**: km² a varrer, células mais prováveis primeiro, para
  cobrir 80% das células realmente oleadas. É a moeda operacional da camada.

### Resultado 1 — no local JÁ CONHECIDO, calibração empata e ordenamento não

Cego 2024, 240 cenários:

| modelo | Brier D+7 | BSS | IoU D+7 | área 80% (km²) |
|---|---|---|---|---|
| climatologia | 0,00517 | 0,00 | 0,186 | 43.496 |
| persistência | 0,00581 | −0,12 | 0,155 | 65.926 |
| análogo | 0,00523 | −0,01 | 0,191 | 44.612 |
| corredor do centróide | 0,00558 | −0,08 | 0,174 | 39.159 |
| pluma | 0,00522 | −0,01 | 0,190 | **35.194** |
| ocupação | 0,00535 | −0,03 | **0,198** | **32.467** |

A leitura: num local que a climatologia já viu, **nenhum modelo melhora a
calibração** — exatamente o padrão do §5e Resultado 1. Mas a área que
precisa ser varrida cai 19–25%. A footprint verdadeira mediana em D+7 são 41
células (5.080 km²), então a climatologia manda varrer 8,6× a mancha e a
ocupação 6,4×. Brier e área medem coisas diferentes: a primeira pergunta se o
número está certo, a segunda se a **ordem** está.

### Resultado 2 — no local NUNCA VISTO, o corredor do v4 ganha em tudo que ordena

Leave-one-field-out, 720 cenários retidos. Esta é a pergunta de implantação —
e, como em §5e, é onde a resposta muda.

IoU (limiar escolhido na calibração):

| horizonte | climatologia | **corredor** | análogo | pluma | ocupação |
|---|---|---|---|---|---|
| D+1 | 0,553 | **0,610** | 0,581 | 0,464 | 0,522 |
| D+2 | 0,361 | **0,422** | 0,401 | 0,349 | 0,345 |
| D+3 | 0,288 | **0,330** | 0,319 | 0,287 | 0,272 |
| D+5 | 0,216 | 0,242 | **0,250** | 0,216 | 0,208 |
| D+7 | 0,182 | 0,203 | **0,212** | 0,193 | 0,175 |

Área a varrer para cobrir 80% das células oleadas (km², mediana):

| horizonte | climatologia | **corredor** | análogo | pluma | ocupação |
|---|---|---|---|---|---|
| D+1 | 1.611 | **1.239** | 1.363 | 1.735 | 1.735 |
| D+2 | 5.205 | **3.594** | 4.089 | 4.957 | 5.329 |
| D+3 | 9.728 | **7.745** | 8.798 | 9.666 | 9.356 |
| D+5 | 19.827 | **18.588** | 20.075 | 20.509 | 19.332 |
| D+7 | 33.893 | **29.927** | 32.715 | 31.042 | 32.158 |

Wilcoxon pareado, corredor vs climatologia (720 cenários):

| horizonte | Δ IoU | p | Δ área | p |
|---|---|---|---|---|
| D+1 | **+0,054** | 1,7e−12 | **−124 km²** | 1,2e−15 |
| D+2 | **+0,047** | 8,8e−14 | **−991 km²** | 1,1e−20 |
| D+3 | **+0,023** | 2,3e−09 | **−1.363 km²** | 5,4e−15 |
| D+5 | +0,006 | 1,1e−04 | **−2.355 km²** | 5,7e−09 |
| D+7 | +0,011 | 9,2e−06 | **−3.718 km²** | 1,1e−10 |

**O corredor ganha da climatologia em IoU e em área nos cinco horizontes,
com significância.** É o resultado de §5e reaparecendo em espaço de forma:
o que transfere para um ponto novo é a previsão de *para onde* a mancha vai,
e envolvê-la numa faixa calibrada basta para desenhar a footprint. Note o
contraste com o cego: **num local conhecido o corredor PERDE IoU** para a
climatologia (Δ = −0,047 a −0,020, p<0,05 em todos), porque ali a
climatologia já viu o sítio. Os dois quadros juntos são a evidência; o de
local novo é o que vale para implantação.

Os dois modelos mais ambiciosos falham no local novo, e isso é informativo:

- **Ocupação** (classificador por célula) perde para a climatologia em IoU em
  todos os horizontes (Δ −0,008 a −0,017). Ele aprende a forma da pluma
  condicionada a lon/lat e não a transporta — mesmo padrão do controle linear
  do §5e, por motivo oposto (excesso de capacidade em vez de falta).
- **Pluma** (núcleo 2D) fica atrás do corredor isotrópico. Diagnóstico
  geométrico: as coordenadas do núcleo são normalizadas pelo deslocamento
  previsto L, com bins de 5% de L. Em D+1, L ≈ 31 km, então o bin vale 1,5 km
  — **sete vezes menor que a célula de 11,1 km**. O núcleo passa a estimar
  estrutura mais fina do que o dado tem. Em D+7 (L ≈ 200 km) o bin vale 10 km
  ≈ 1 célula, e ali a pluma de fato empata/ganha (cego: menor Brier entre os
  modelos de ML). A hipótese "referencial livre de escala" vale acima de
  L ≈ 1 célula por bin e falha abaixo — corrigir é item da agenda v7, não um
  ajuste silencioso.

### Resultado 3 — o corredor é calibrado; o Brier ruim é nitidez, não erro

O Brier do corredor é pior que o da climatologia (BSS −0,05 a −0,14), o que
lido sozinho sugeriria um modelo descalibrado. A curva de confiabilidade,
medida **através do artefato exportado** no cego 2024, diz o contrário:

| previsto | 0,00 | 0,03 | 0,07 | 0,13 | 0,24 | 0,31 |
|---|---|---|---|---|---|---|
| observado (D+7) | 0,00 | 0,03 | 0,07 | 0,12 | 0,50 | 0,30 |

Os desvios só aparecem nas faixas altas, que contêm **menos de 2% das
células** (1,80% acima de p=0,30 em D+1; 0,55% em D+7). Ou seja: a
decomposição confiabilidade/nitidez — o corredor se compromete com uma faixa
estreita e paga caro quando a mancha vai para outro lado, enquanto a
climatologia se protege espalhando probabilidade. Por isso ele perde no Brier
e ganha na área de busca, que é uma métrica de **ordenamento**. Para o uso
operacional, é o ordenamento que importa.

### O produto

`main/outputs/ml/footprint_plume.joblib` — modelos de centróide do v4 +
forma do corredor (isotônica) + núcleo da pluma + climatologia por estação,
tudo por horizonte, mais o ponto de operação avaliado. Default declarado:
**corredor**, escolhido pelo leave-one-field-out (local novo = caso de
implantação), não pelo cego. As duas formas viajam no artefato, então trocar
o default é decisão documentada e não edição de código.

```
python -m main.ml.footprint_forecast --export        # ajusta e grava o produto
python -m main.ml.footprint_forecast --reliability   # exporta + confere calibração
```

Consumo pelo app (item 3 da agenda): `predict_footprint(payload, x_row, h,
lon0, lat0, season)` devolve centros geográficos das células, probabilidade,
limiar e o caminho previsto — pronto para desenhar.

### Limitações desta rodada

1. A pluma 2D está quebrada abaixo de L ≈ 1 célula por bin (diagnóstico
   acima). O corredor isotrópico a substitui, mas a forma anisotrópica
   continua sendo a hipótese com mais margem.
2. A footprint é a **área varrida**, não a mancha instantânea — que é
   pequena demais para a grade do app (1–2 células). Uma footprint
   instantânea exigiria grade mais fina, e aí o alvo vira o centróide.
3. O modelo de ocupação amostra negativos: seus números carregam ruído de
   Monte Carlo de alguns por cento entre sementes. Os demais são
   determinísticos.
4. A LOFO retém **espaço, não tempo** (mesmo protocolo do §5e); quem retém
   tempo é o cego 2024. Nenhum dos dois retém os dois ao mesmo tempo.

### Agenda v7

1. Consertar a pluma: bins do núcleo em km absolutos (ou piso em L), para
   que a resolução do núcleo nunca fique abaixo da resolução do dado.
2. Expor no app (item 3 da v6, agora com produto pronto para consumir).
3. Mais locais de semeadura (item 2 da v6) — continua sendo a limitação #1.
4. D+14 (item 4 da v6) — continua exigindo runs de 336 h.

### Verificações do método

- **Empates na área de captura**: as ids de célula correm de sudoeste para
  nordeste, que é para onde o óleo vai, então desempatar por índice poderia
  favorecer o modelo de probabilidade mais grosseira. Medido contra desempate
  aleatório (5 sorteios): diferença ≤1,8% e **nenhuma troca de posição** no
  ranking.
- **Ruído de Monte Carlo**: o modelo de ocupação amostra 500 negativos por
  cenário, então seus números carregam ~alguns por cento de ruído entre
  execuções com sementes diferentes. Os demais modelos são determinísticos.

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
