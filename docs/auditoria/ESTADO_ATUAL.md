# ESTADO ATUAL — leia este arquivo primeiro

> Atualizado em **2026-08-26**, branch **`audit/revisao-completa`**.
> Este é o único documento desta pasta que descreve o **estado de hoje**. Os
> demais (`DIAGNOSTICO.md`, `MAPA_DO_PROJETO.md`, `ARQUITETURA.md`,
> `PIPELINE_CIENTIFICO.md`, `PLANO_DE_ACAO.md`) são **fotografias datadas de
> 2026-07-29**, mantidas como registro da auditoria: eles descrevem bugs que
> **já foram corrigidos**. Não os leia como diagnóstico corrente.
>
> **Aviso de branch:** o branch default do repositório (`main`) contém uma
> versão do projeto anterior à auditoria. Todo o trabalho descrito aqui está
> em `audit/revisao-completa`.

## Para quem vai fazer crosscheck

Comece pela seção **§3 (afirmações e como reproduzi-las)**. Cada número
publicado aponta para o arquivo que o contém e para o comando que o regera.
As simulações são determinísticas (§5), então os números devem bater
exatamente, não aproximadamente. A seção **§6** lista, sem maquiagem, onde
achamos que o trabalho é mais atacável.

---

## 1. O que é o projeto

Modelagem de dispersão de óleo para seis campos da **Bacia de Campos**
(Peregrino, Marlim, Roncador, Papa-Terra, Frade, Albacora) sobre o
**OpenDrift/OpenOil v1.14.7**, usado *in-place* e **não modificado** (o
diretório `opendrift/` é o upstream intocado; todo o código do projeto vive
em `main/`). Projeto de pesquisa acadêmica. Três camadas:

1. **Simulação física** — `main/run_open_oil.py` monta e roda o OpenOil.
2. **Produtos pré-computados** — cenários, ensemble, grades de risco e de
   encalhe, servidos por um app Streamlit de 5 abas.
3. **Camada de ML** — `main/ml/`, cujo objetivo é prever o destino da mancha
   a partir apenas do que se sabe **no instante do vazamento**. Desde
   2026-08-26 ela é servida na 5ª aba do app, sem rodar simulação.

## 2. Cronologia — o que foi feito, em ordem

| Quando | Fase | Resultado |
|---|---|---|
| até 2026-06 | Construção inicial | Pipeline + app funcionando (pré-auditoria) |
| 2026-07-29 | **Auditoria técnica** | 2 achados 🔴, 4 🟠, 10 🟡, 5 🔵 (`DIAGNOSTICO.md`). Os dois críticos: encalhe contava saída de domínio como encalhe, e o mixing vertical estava LIGADO ao contrário do que a doc afirmava |
| 2026-07-30 | **Fase 8 — correções + regeneração** | Todos os fixes + forçantes novas (caixa larga, SST, coordenadas ANP oficiais, Papa-Terra no lugar de Jubarte). 288 runs regerados, 0 falhas, 0 saídas de domínio (`REGENERACAO.md`) |
| 2026-07-31 | ML v1–v2 | Surrogate de transporte de patch; cego 2024 mostrou que não transferia entre anos |
| 2026-08-07 | ML v3 + **diagnóstico decisivo** | O resíduo que o modelo aprendia era **erro de integração numérica**, não física oculta. Trocar para integração midpoint (RK2) cortou o erro 57% com zero parâmetros (`CAMADA_IA.md` §5c) |
| 2026-08-07 | ML v4 — reformulação | Alvo passa a ser previsão **em nível de cenário** (sem forçante futura). Primeiro ganho robusto: local nunca visto (`CAMADA_IA.md` §5d) |
| 2026-08-11 | **Escopo D+7** | Arquivos de 168 h completos; controle linear no resultado-manchete; incerteza calibrada por conformal (`CAMADA_IA.md` §5e) |
| 2026-08-25 | **Footprint** (item 1 da agenda v6) | Alvo passa a ser a grade de células oleadas, não só o centróide. Num local nunca visto, o corredor calibrado em volta do caminho previsto pelo v4 bate a climatologia em IoU e em área de busca nos 5 horizontes; produto exportado para o app (`CAMADA_IA.md` §5f) |
| 2026-08-26 | **A camada vira produto** (item 1 da agenda v7) | Modelos passam a ser exportados (`forecast_product.joblib`, `footprint_plume.joblib`), features ganham fonte única (`scenario.feature_row`) e o app ganha a aba **Forecast (ML)**: ponto de vazamento arbitrário, resposta em ~1 s, sem simulação (`CAMADA_IA.md` §5g) |

### O que mudou de conclusão ao longo do caminho (importante para o crosscheck)

Duas afirmações publicadas foram **revistas pelos próprios dados**, e as
versões antigas continuam no histórico. Um revisor deve conferir que a versão
vigente é a que está aqui:

1. **"O surrogate de ML aprende física que a advecção não captura"** →
   **falso**. O que ele aprendia era erro de integração. Substituído por RK2,
   que não tem parâmetros. Registro: `CAMADA_IA.md` §5c.
2. **"HGB empata com a climatologia no cego porque n=72 falta poder"** →
   **incompleto**. Com n=240 o empate se confirma e ganha explicação: num
   local já conhecido, a previsibilidade se esgota em D+3. Registro:
   `CAMADA_IA.md` §5e, Resultado 1.

## 3. Afirmações publicadas → onde o número vive → como reproduzir

Todos os comandos rodam **da raiz do repositório**, com o env conda
`opendrift` ativo (§7).

| Afirmação | Onde está registrada | Artefato com o número | Comando que reproduz |
|---|---|---|---|
| 288 runs regerados, 0 falhas, 0 saída de domínio | `REGENERACAO.md` | `main/outputs/{scenarios,ensemble}/manifest.json` | `.\main\rebuild_all.ps1 --fresh` (~3,5–4 h) |
| Encalhe real ≈ 0 (só papa-terra_jan, 3,41%) | `main/CLAUDE.md`, `REGENERACAO.md` | `main/outputs/beaching/*.npz` | `.\main\rebuild_all.ps1 --only beaching` |
| RK2 bate os surrogates aprendidos (4,6 km vs 10,7 km em 120 h) | `CAMADA_IA.md` §5c | `main/outputs/ml/holdout_2024_report.json` | `python -m main.ml.holdout evaluate` |
| **Local nunca visto: HGB +24–36% sobre climatologia** | `CAMADA_IA.md` §5e Res. 2 | `forecast_report.json` → `lofo_new_location.paired` | `python -m main.ml.forecast` |
| **Local conhecido: previsibilidade acaba em D+3** | `CAMADA_IA.md` §5e Res. 1 | `forecast_report.json` → `paired_tests_by_horizon` | idem |
| **Ridge cai abaixo da climatologia no local novo** | `CAMADA_IA.md` §5e Res. 2 | `forecast_report.json` → `lofo_new_location.paired.*.median_ridge_km` | idem |
| **Envelope cru cobria 35–49%; conformal 84–89%** | `CAMADA_IA.md` §5e Res. 3 | `forecast_report.json` → `uncertainty_envelope{,_raw}` | idem |
| `u_mean_3d` é a feature dominante | `CAMADA_IA.md` §5e Res. 4 | `forecast_report.json` → `top_features_d7_dist` | idem |
| Espalhamento não é previsível (MAE 1,28–1,38 km) | `CAMADA_IA.md` §5e | `forecast_report.json` → `blind_2024.*.spread_mae_km` | idem |
| **Footprint, local novo: corredor bate a climatologia em IoU e área nos 5 horizontes** | `CAMADA_IA.md` §5f Res. 2 | `footprint_report.json` → `lofo_new_location.*._paired_vs_climatology.centroid` | `python -m main.ml.footprint_forecast` (~50 min) |
| Footprint, local conhecido: o corredor perde IoU e ganha área | `CAMADA_IA.md` §5f Res. 1 | `footprint_report.json` → `blind_2024` | idem |
| O corredor é calibrado; o Brier pior é nitidez | `CAMADA_IA.md` §5f Res. 3 | `footprint_reliability.json` | `python -m main.ml.footprint_forecast --reliability` |
| Mancha instantânea = 1–2 células; varrida = 8–39 | `CAMADA_IA.md` §5f Decisão 1 | saída de `main.ml.footprint` | `python -m main.ml.footprint [--holdout]` |

Reconstrução dos datasets de ML (pré-requisito de `main.ml.forecast` e
`main.ml.footprint_forecast`):

```
python -m main.ml.scenario              # -> outputs/ml/scenario_dataset.npz       (720 cenários)
python -m main.ml.scenario --holdout    # -> outputs/ml/scenario_dataset_2024.npz  (240 cenários)
python -m main.ml.footprint             # -> outputs/ml/footprint_dataset.npz      (720, células varridas)
python -m main.ml.footprint --holdout   # -> outputs/ml/footprint_dataset_2024.npz (240)
```

Reconstrução dos arquivos de simulação que alimentam tudo isso (≈40 min por
ano, resumível — só refaz o que falta):

```
python -m main.ml.multiyear generate 2022     # idem 2023, 2024, 2025
```

## 4. Inventário de dados — 2.232 simulações versionadas

| Pasta em `main/outputs/` | Runs | Duração | Ano(s) | Para que serve |
|---|---|---|---|---|
| `scenarios/` | 48 | 120 h | 2025 | Aba 1 do app (6 campos × 4 meses × vento on/off) |
| `ensemble/` | 672 | 120 h | 2025 | Grades de risco e encalhe (28 datas de início × 6 × 4) |
| `risk_grids/` | 24 grades | — | 2025 | Aba 2 (`prob_any`, `prob_final`, grade 0,1°) |
| `beaching/` | 24 grades | — | 2025 | Aba 3 |
| `training_{2022,2023}/` | 240 cada | 120 h | 2022, 2023 | Treino do surrogate de patch (v1–v3) — **legado**, mantido como registro de §5a–5d |
| `holdout_2024/` | 72 | 120 h | 2024 | Cego do surrogate de patch — **legado** |
| `training168_{2022,2023,2024,2025}/` | 240 cada | **168 h** | 4 anos | **Camada de cenário vigente**: treino = 2022+2023+2025 (720), cego = 2024 (240) |

Forçantes em `main/inputs/` (também versionadas): correntes CMEMS 1/12° diárias
+ SST, vento ERA5 0,25° horário, caixa lon −45..−36 / lat −27..−19, anos 2022
a 2025. 2022 vem da reanálise GLORYS `my` (o produto de análise `anfc` começa
em meados de 2022 e a API do CMEMS **recorta silenciosamente** — `download()`
verifica cobertura desde então).

**Por que tudo isso está no git** (`outputs/` 1,56 GB + `inputs/` 0,85 GB;
`.git` 2,49 GB): decisão do autor, registrada em
`PERGUNTAS_ABERTAS.md` #9 e reconfirmada em 2026-08-11. Os arquivos são
regeráveis (§5), então é uma escolha de conveniência, não de necessidade.

## 5. Reprodutibilidade — verificada, não assumida

As simulações são **determinísticas**. O OpenDrift chama `np.random.seed(seed)`
no construtor do modelo (`opendrift/models/basemodel/__init__.py:325`, default
`seed=0`), e as incertezas declaradas do projeto
(`drift:current_uncertainty=0,05`, `drift:wind_uncertainty=0,5`) sorteiam desse
mesmo RNG a cada passo. Verificação empírica: dois runs do mesmo cenário com
forçante real produzem `max|Δlon| = 0`.

Isso está exposto como `run_simulation(random_seed=0)`, com testes em
`main/tests/test_run_reproducibility.py`. **Armadilha registrada:** chamar
`np.random.seed()` *antes* de construir o `OpenOil` não funciona — o construtor
sobrescreve. A semente precisa ir como argumento do construtor.

Consequência prática para o crosscheck: qualquer número desta documentação
deve ser reproduzível **exatamente**. Divergência é bug, não ruído.

## 6. Onde este trabalho é mais atacável

Lista deliberada de fraquezas — se o revisor for atrás de algo, que seja daqui:

1. **Só 6 locais de treino.** A afirmação de generalização espacial vem de
   leave-one-field-out entre campos vizinhos da mesma bacia. Não há evidência
   para geografia arbitrária. É a limitação #1 e o item 2 da agenda v6.
2. **Espalhamento não é previsto por ninguém** (MAE ~1,3 km para todos os
   modelos, inclusive climatologia). Ou não há sinal nas features, ou o alvo
   está mal definido — não sabemos qual.
3. **A cobertura conformal fica em 84–89% contra 80% nominal.** Conservadora,
   como esperado ao calibrar num ano e aplicar em outro, mas significa que a
   garantia de troca (*exchangeability*) do CQR não vale estritamente entre
   anos.
4. **A climatologia é um baseline forte de forma artificial** no cego com
   local conhecido: só existem 6 locais fixos, e ela já viu aquele campo em
   outros anos. Por isso o resultado que importa é o de local novo (§3).
5. **Um único modelo de forçante.** Erros sistemáticos do CMEMS/ERA5 entram
   igualmente no "verdadeiro" e no previsto — o ML aprende a reproduzir o
   OpenDrift, não o oceano. Nenhuma validação contra deriva observada foi
   feita (não há dado observado no projeto).
6. **Ensemble por datas de início**, não por perturbação de física: a
   dispersão do ensemble mede variabilidade temporal, não incerteza de modelo.
7. **Ondas reais (ERA5) nunca entraram**: o Stokes drift, quando ligado, vem
   de parametrização a partir do vento.
8. **A footprint prevista é a área VARRIDA, não a mancha instantânea.** Não é
   uma escolha de conveniência — a mancha instantânea ocupa 1–2 células na
   grade de 0,1° (§5f, Decisão 1) —, mas quem esperar "onde está o óleo no
   dia 7" está lendo outra coisa: "por onde o óleo passou até o dia 7".

## 7. Como rodar

- Env conda **`opendrift`** (miniforge, Python 3.14). O python do PATH **não**
  serve. No Windows, o numpy precisa estar linkado ao **OpenBLAS** — com MKL o
  processo morre nativamente (exit `0xC06D007F`, sem output).
- Sempre a partir da raiz do repositório.
- App: `python -m streamlit run main/app.py`
- Testes: `python -m pytest main/tests -o addopts=""` (**118 testes**, todos
  passando em 2026-08-26; `-o addopts=""` neutraliza as opções de pytest do
  OpenDrift upstream).

## 8. O que está aberto (agenda v8)

Entregues: footprint (v6 item 1, em 2026-08-25, `CAMADA_IA.md` §5f) e a
exposição da camada no app (v7 item 1, em 2026-08-26, §5g). O que sobra, em
ordem de custo:

1. **Consertar a pluma 2D** — os bins do núcleo são normalizados pelo
   deslocamento previsto e ficam menores que a célula em horizontes curtos
   (§5f, Resultado 2). Barato, e é a hipótese com mais margem. **Próximo
   passo recomendado.**
2. **Descobrir por que o espalhamento não é previsível** (MAE 1,28–1,38 km
   para todo modelo, inclusive climatologia): ou não há sinal nas features,
   ou o alvo está mal definido. Não exige simulação nova.
3. **Deploy do app** — plataforma segue em aberto (`PERGUNTAS_ABERTAS.md`),
   e o polimento de UI foi decidido para depois de ciência+ML.
4. **Mais locais de semeadura** (grade de pontos, não só os 6 campos) — ataca
   diretamente a limitação #1 da §6. Exige horas de simulação nova.
5. **Ondas ERA5 reais** — os scripts existem, `waves_cf.nc` nunca foi gerado;
   o Stokes drift hoje é parametrizado do vento (§6.7).
6. **D+14** — exige runs de 336 h; o arquivo atual vai até 168 h. Medir antes
   de prometer: em D+7 o ganho sobre climatologia no local conhecido já é nulo.

## 9. Mapa dos documentos

| Arquivo | O que é | Vigente? |
|---|---|---|
| `docs/auditoria/ESTADO_ATUAL.md` | este arquivo | ✅ |
| `docs/auditoria/CAMADA_IA.md` | registro completo da camada de ML, §5a–§5e | ✅ |
| `docs/auditoria/REGENERACAO.md` | registro da regeração de 2026-07-30 | ✅ (histórico factual) |
| `docs/auditoria/PERGUNTAS_ABERTAS.md` | decisões do autor, registradas | ✅ |
| `docs/auditoria/DIAGNOSTICO.md` | achados da auditoria | 📷 foto de 2026-07-29 — bugs já corrigidos |
| `docs/auditoria/MAPA_DO_PROJETO.md` | inventário arquivo a arquivo | 📷 foto de 2026-07-29 — anterior a `main/ml/` |
| `docs/auditoria/ARQUITETURA.md` | fluxo fim-a-fim | 📷 foto de 2026-07-29 — anterior a `main/ml/` |
| `docs/auditoria/PIPELINE_CIENTIFICO.md` | config efetiva do OpenOil | 📷 foto de 2026-07-29 — valores pré-fix |
| `docs/auditoria/PLANO_DE_ACAO.md` | plano da Fase 8 | 📷 concluído |
| `main/CLAUDE.md` | contexto portátil para o Claude (inglês) | ✅ |
| `main/README.md` | README do projeto (inglês) | ✅ |
| `CLAUDE.md` (raiz) | contexto do repositório | ✅ |
