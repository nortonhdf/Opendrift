# PIPELINE CIENTÍFICO — configuração, unidades, premissas, catálogo

> Tudo abaixo foi verificado empiricamente em 2026-07-29 (arquivos abertos, modelo instanciado,
> configs interrogadas), não inferido de nomes ou READMEs.

## 1. Configuração efetiva do OpenOil (como `run_simulation` monta, caso batch)

| Config | Valor efetivo | Origem | Nota |
|---|---|---|---|
| Modelo | `OpenOil(weathering_model="noaa")` | `run_open_oil.py:211` | |
| `drift:vertical_mixing` | **True** | default OpenOil | ⚠ contradiz docs do projeto (🔴 2); decisão: desligar |
| `processes:dispersion` | True | default | entranhamento ativo |
| `processes:evaporation` / `emulsification` | True / True | default | |
| `processes:biodegradation` | False | default | budget `mass_biodegraded` ≡ 0 |
| `drift:stokes_drift` | False nos batches; True + tabularizado no app c/ waves | `run_open_oil.py:223-230` | fetch 25.000 m |
| `drift:horizontal_diffusivity` | 0 | default | espalhamento vem só das incertezas ↓ |
| `drift:current_uncertainty` | 0,05 m/s | default | não documentado no projeto |
| `drift:wind_uncertainty` | 0,5 m/s | default | idem |
| `drift:advection_scheme` | `euler` | default | RK4 recomendado |
| `general:coastline_action` | `stranding` | default | encalhe desativa a partícula |
| `general:seafloor_action` | `lift_to_seafloor` | default | |
| `seed:wind_drift_factor` | 0,03 | default | 3% do vento na superfície |
| `seed:m3_per_hour` | **1** | default | ⚠ derrame de 1 m³ (963 kg p/ heavy crude) — nunca declarado (🟠 6) |
| `seed:droplet_size_distribution` | `uniform` | default | |
| Fallbacks | T=10 °C, S=34; correntes/vento/land = None | default | ⚠ weathering a 10 °C (🟠 3); None → `missing_data` na borda (🟠 4) |
| Semeadura | ponto (radius=1 m), z=0, instantânea | `run_open_oil.py:257-264` | 500 (cenários) / 200 (ensemble) / 100–2000 (app) |
| Passos | dt=600 s, output=1800 s, duração 120 h | `run_open_oil.py:40-42` | 241 timesteps de saída |
| Óleos | `GENERIC HEAVY CRUDE` / `GENERIC MEDIUM CRUDE` por campo; default `GENERIC BUNKER C` | `fields_config.py` | os 3 existem por match exato no catálogo ADIOS (1.280 óleos) |

## 2. Unidades e referenciais — checklist verificado

| Item | Situação |
|---|---|
| m/s vs nós | ✅ tudo m/s (`units` conferidos nos NetCDF; magnitudes plausíveis: correntes ≤1,5, vento ≤16,6 m/s) |
| Sentido do vento | ✅ ERA5 `u10/v10` são componentes "para onde vai"; renomeação não inverte nada. A armadilha "from direction" não existe neste pipeline (só entraria com `mean_wave_direction` das ondas — cadeia nunca usada) |
| Graus vs metros | ✅ posições em graus; conversões km no app usam 111 km/° com cos(lat) — aproximação razoável na escala usada |
| CRS | ✅ lat/lon regulares WGS84-like em toda a cadeia; sem projeções intermediárias |
| Lon [−180,180] vs [0,360] | ✅ tudo em [−180,180] |
| UTC vs local | ✅ ERA5/CMEMS são UTC; datetimes naïve do código = UTC consistente |
| Latitude decrescente no ERA5 | ✅ reader do OpenDrift lida; verificado nos runs |

## 3. Forçantes

| Arquivo | Fonte | Grade | Tempo | Domínio |
|---|---|---|---|---|
| `currents.nc` | CMEMS `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` (camada 0–1 m, `isel(depth=0)`) | 1/12° (43×43) | diário, 365 dias de 2025 | −42,5..−39,0 / −24,5..−21,0 |
| `wind_cf.nc` | ERA5 single-levels u10/v10 | 0,25° (15×15) | horário, 8.760 h de 2025 | idem |
| ondas | — | — | — | **inexistente** (Stokes só parametrizado do vento) |
| SST | — | — | — | **inexistente** → fallback 10 °C (🟠 3) |

Limitações estruturais a declarar em qualquer defesa: correntes **diárias** (sem maré nem
inércia sub-diária), **superficiais** (uma camada), um **único ano** (2025) representando
"estações" por 4 meses; deriva de vento 3% fixa.

## 4. Catálogo das simulações pré-computadas

| Produto | Nº | Parâmetros varridos | Formato |
|---|---|---|---|
| `scenarios/` | 48 | 6 campos × {jan,apr,jul,oct}(dia 15) × wind {on,off}; 500 part. | `.nc` (lon,lat,status,z,massas,densidade,water_fraction,viscosity) + `_budget.npz` + `.png` |
| `ensemble/` | 240 | 6 × 4 × 10 datas (dias 1–28, passo 3); wind on; 200 part. | idem |
| `risk_grids/` | 24 | agregação por (campo, mês) | `.npz`: `prob_any`, `prob_final`, lons, lats, n_members |
| `beaching/` | 24 | idem | `.npz`: `strand_grid`, fração, percentis — **INVÁLIDOS até correção do 🔴 1** |

Amostragem: **grade regular** (não LHS/aleatória). Adequada para consulta; estreita para
interpolação/ML (6 pontos espaciais fixos, 1 volume, 1 ano).

Integridade (varredura de 2026-07-29): 288/288 legíveis, 288/288 sidecars presentes,
manifests 100% consistentes com o disco nos dois sentidos.

## 5. Semântica de `status` nos NetCDF (essencial para qualquer pós-processamento)

O código numérico de status **varia por arquivo**; o mapeamento correto está nos attrs
`flag_values`/`flag_meanings` da variável `status`. Mapeamentos observados nos 288 arquivos:

| `flag_meanings` | Arquivos (scen + ens) |
|---|---|
| `active` | 41 + 181 |
| `active missing_data` | 7 + 48 |
| `active stranded` | 0 + 10 |
| `active stranded missing_data` | 0 + 1 |

Regra: `active` é sempre 0; **os demais dependem da ordem de ocorrência no run**. Nunca
hardcodar (causa do 🔴 1). `missing_data` = partícula saiu da cobertura das forçantes.

## 6. Comportamento nas fronteiras

- **Costa:** `stranding` — partícula desativa no toque; posição final = ponto de encalhe.
- **Borda do domínio de forçantes:** fallback None → desativação `missing_data` no passo
  seguinte à saída. 15,9% do ensemble e 14,1% dos cenários terminam assim (medido).
- **Fundo:** irrelevante na prática (z≈0; `lift_to_seafloor`).

## 7. Desempenho e reprodutibilidade (medidos)

- Run 24 h/100 part.: 9–19 s. Cenário completo 120 h/500 part.: ~59 s (log do rebuild).
- Rebuild completo: 94,5 min (48+240 runs + grids), sequencial.
- Lookup pré-computado: 0,02 s → ganho ~3.000× sobre rodar ao vivo.
- Determinismo: dois runs de config idêntica → desvio 0,00 km (bit-a-bit). A regeração completa
  reproduz os produtos **se** o env for o mesmo (`environment.yml` hoje diverge do env real — 🟡 10).

## 8. Veredito

Os parâmetros de transporte são defensáveis para estudo acadêmico regional; unidades e
referenciais estão corretos. Os pontos indefensáveis eram **defaults silenciosos** (mixing
vertical, SST 10 °C, 1 m³, incertezas estocásticas) e o **pós-processamento de status** — todos
mapeados no DIAGNOSTICO com correção aprovada.
