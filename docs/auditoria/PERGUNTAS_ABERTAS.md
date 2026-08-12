# PERGUNTAS ABERTAS → DECISÕES REGISTRADAS

> Perguntas levantadas pela auditoria em 2026-07-29 e **respondidas pelo autor na mesma data**.
> Este arquivo agora é o registro das decisões. Itens sem resposta definitiva estão marcados ⏳.

## Ciência / metodologia

| # | Pergunta | Decisão do autor | Operacionalização |
|---|---|---|---|
| 1 | Cenário de derrame (hoje 1 m³ por default herdado) | **Definir um default; tipo de vazamento não é o foco** (foco = técnicas de IA na análise) | Fase 8: expor `spill_m3` em `run_simulation`; proposta do auditor: **default 10 m³ instantâneo**, declarado na UI/docs e configurável |
| 2 | Nova caixa de forçantes | **Aprovada: lon −45..−36, lat −27..−19** | Fase 8 itens 1.5/1.7 |
| 3 | Regra API↔óleo contradita | **A regra do cabeçalho está certa; corrigir a classificação** | Roncador (18°), Frade (18°) e o campo substituto passam a `GENERIC MEDIUM CRUDE` conforme 15–22° → MEDIUM |
| 4 | Fonte das coordenadas dos campos | **Validar em fontes confiáveis e usar no modelo** | Pesquisa feita (ver §Validação abaixo); Roncador corrigido para o valor referenciado; demais: derivar do shapefile oficial da ANP na Fase 8 |
| 5 | Jubarte é Bacia do Espírito Santo | **Trocar por um campo de Campos** | Substituto proposto: **Papa-Terra** (sul de Campos, óleo pesado 14–17,4° API, lâmina 1.190 m) — coordenadas via ANP na Fase 8 |
| 6 | Mais anos além de 2025 | **Podem entrar se relevante** | Relevante para ML (holdout) — ver #15 |
| 7 | Espalhamento estocástico (defaults 0,05/0,5 m/s) | **A critério do auditor** | Decisão do auditor: **manter os defaults e declará-los** como mecanismo de espalhamento na documentação do método (calibração fica como trabalho futuro) |
| 8 | Convergência de partículas/membros nunca testada | **Livre para fazer** | Fase 8: experimento rápido (risk grid com 5 vs 10 membros; 200 vs 500 partículas) antes de fixar números na metodologia |

## Produto / engenharia

| # | Pergunta | Decisão |
|---|---|---|
| 9 | Outputs no git (202 MB) | **Manter no git** |
| 10 | Papel do app | **Entregável importante**; UI recebe atenção depois que o núcleo científico+IA estiver pronto |
| 11 | Plataforma de deploy | **Flexível** — não fixo em Streamlit Cloud |
| 12 | Ondas ERA5 reais | **Entram como alternativa** — toggle na UI (ondas on/off) |
| 13 | Padronizar idioma | **Sim** — código/UI/docs de projeto em inglês (os docs desta auditoria permanecem em PT como registro) |

## Futuro (camada de IA)

| # | Pergunta | Decisão |
|---|---|---|
| 14 | Alvo da pesquisa de IA | **(a) surrogate de transporte de patch E (b) estatísticas-resumo por cenário** — (b) exigirá novo plano de amostragem (LHS sobre posição/volume/data) |
| 15 | Ano de holdout | Autor indiferente → decisão do auditor: **2024 reservado como teste** (baixar 2024 junto com a regeração; nunca usar no treino) |
| 16 | Métrica de sucesso do surrogate | **Pesquisado — recomendação abaixo** (§Métrica) |

## Arqueologia

| # | Pergunta | Decisão |
|---|---|---|
| 17 | `openoil_smoketest.nc` órfão | **Remover na limpeza** (aprovado) |
| 18 | `rebuild_all.log` | Ninguém depende dele como registro — manifests são a fonte de verdade |

## Decisões posteriores (registradas na data em que foram tomadas)

| # | Data | Questão | Decisão do autor |
|---|---|---|---|
| 19 | 2026-08-07 | Horizonte de previsão da camada de cenário | **Até D+7.** D+14 adiado — exigiria runs de 336 h. Consequência: novo arquivo `training168_*` (240 runs/ano, 168 h); os arquivos de 120 h ficam intactos como registro de §5a–5d |
| 20 | 2026-08-11 | Versionar os 756 MB dos arquivos de 168 h no git | **Sim, versionar tudo** — mantém a decisão #9. Reconfirmada depois de se verificar que os arquivos são regeráveis (a escolha é de conveniência, não de necessidade); `.git` vai a ~2,7 GB |
| 21 | 2026-08-11 | Semente de RNG nas simulações | **Expor como parâmetro opcional.** `run_simulation(random_seed=0)`; o default 0 é o que gerou todo o arquivo existente, então a reprodutibilidade deixa de ser acidente do default da biblioteca e passa a ser propriedade declarada e testada |

Decisões metodológicas tomadas pelo auditor dentro do escopo já aprovado
(registradas em `CAMADA_IA.md` §5e, sem necessidade de decisão do autor):
controle linear (RidgeCV) obrigatório em toda afirmação de ganho do ML;
`dist_km` derivado de `|(dx, dy)|` em vez de alvo independente; HGB com
`loss='absolute_error'` para casar com a mediana que se reporta; envelope de
incerteza calibrado por conformal (CQR) com calibração em **ano inteiro
deixado de fora**, nunca split aleatório.

---

## §Validação das coordenadas (pesquisa de 2026-07-29)

O que fontes públicas confirmam numericamente vs o que está em `fields_config.py`:

| Campo | Projeto (lon, lat) | Fonte externa | Veredito |
|---|---|---|---|
| Peregrino | −41,2593, −23,3183 · 100 m · 13° | [Wikipedia](https://en.wikipedia.org/wiki/Peregrino_oil_field): 85 km offshore, SW Campos, 100–119 m, 13° API | ✅ consistente (sem lat/lon numérico público; posição bate com FPSO) |
| Marlim | −40,60, −22,60 · 720 m · 20° | [Wikipedia](https://en.wikipedia.org/wiki/Marlim): ~110 km offshore NE Campos, 650–1.050 m, 17–21° | ✅ consistente |
| Roncador | −39,80, **−22,40** · 1.800 m · 18° | [Wikipedia](https://en.wikipedia.org/wiki/Roncador_Field): **−39,781, −21,977**; 1.500–1.900 m; API 18–31 por módulo | ⚠ **lat difere ~47 km** → corrigir para o valor referenciado na Fase 8 |
| Frade | −41,00, −22,10 · 1.100 m · 18° | [Offshore Technology](https://www.offshore-technology.com/projects/fradefieldcamposbasi/): norte de Campos, 1.128 m | ✅ profundidade bate; sem lat/lon numérico público |
| Albacora | −40,20, −22,00 · 300 m · 19° | [NS Energy/ports](https://ports.marinelink.com/oilrigs/rig/albacora): ~110 km offshore, P-31 a 330 m | ✅ consistente; sem lat/lon numérico público |
| Papa-Terra (novo) | — a definir | [NS Energy](https://www.nsenergybusiness.com/projects/papa-terra-field/), [JPT/SPE](https://jpt.spe.org/heavy-oil-papa-terra-project-uses-innovative-solutions-deep-water): sul de Campos, ~110 km SE do RJ, 1.190 m, **14–17,4° API**, bloco BC-20 | ⏳ coordenada numérica não publicada em fonte aberta |

**Ação definitiva (Fase 8):** baixar o shapefile de campos de produção do geoserver público da
**ANP** (GeoMaps/EPE) e extrair o centróide de cada campo — vira a fonte citável única de todas
as 6 coordenadas, resolvendo Roncador, Frade, Albacora e Papa-Terra de uma vez.

## §Métrica recomendada para o surrogate (pesquisa de 2026-07-29)

Recomendação em 3 níveis, do padrão da literatura para o específico:

1. **Trajetória/centróide — Liu–Weisberg Skill Score** (separação lagrangiana cumulativa
   normalizada pelo comprimento da trajetória): é *a* métrica padrão da área desde o Deepwater
   Horizon, adimensional [0–1], robusta a regimes de corrente fracos/fortes
   ([Liu & Weisberg 2011, JGR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JC006837);
   [análise de sensibilidade e recomendações, Frontiers 2021](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.630388/full)).
2. **Forma da mancha — sobreposição espacial** (IoU ou FSS na grade 0,1°) para o alvo (a),
   comparando a mancha prevista vs a do OpenDrift no mesmo instante
   ([medidas de desempenho para dispersão de óleo](https://www.researchgate.net/publication/354773945_Performance_Measures_for_Validation_of_Oil_Spill_Dispersion_Models_Based_on_Satellite_and_Coastal_Data)).
3. **Skill relativo a baselines**: reportar sempre como ganho sobre persistência e advecção
   pura — um SS alto sem baseline não prova nada.

Para o alvo (b) (estatísticas-resumo): MAE/erro absoluto em fração encalhada e nas
probabilidades por célula (Brier score para P(célula atingida)).
