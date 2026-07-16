# MOCRE-1 — Inventario canónico de variables (autogenerado)

> Generado por `src/inventory.py` desde código+configs+datos reales. NO editar a mano — se regenera. Este fichero es un snapshot de conveniencia; la verdad viva es `python3 src/inventory.py`.

## Universo de discurso
**Uno solo, compartido:** `[-1,1]` (espacio tanh), partición `ANOMALY_SETS` de 5 términos (`muy_bajo/bajo/normal/alto/muy_alto`), en `common.py`. Todas las variables dinámicas usan la MISMA partición; lo único que varía por variable es el mapeo señal_cruda→[-1,1] vía `tau_X`.

## Tabla de variables

| # | Estado | Clú | μ_ev | Variable | Canal | Tipo | τ | Datos | Saturación | Nota |
|---|--------|-----|------|----------|-------|------|---|-------|-----------|------|
| 1 | OK | R | — | Actividad sísmica regional | `a_regional_rate` | dynamic | — | 9/9 | limpio | MOCRE-2: anomalía Poisson, bbox regional, causal |
| 2 | OK | R | — | Expansión espacial regional | `a_regional_spread` | dynamic | — | 9/9 | OK-PAWNEE-PRAGUE=8%, ITALY_APENNINES=8%, JAPAN_NANKAI=100% | MOCRE-2: ocupación espacial 30d normalizada por actividad |
| 3 | OK | R | — | Caída de b-value | `a_bvalue_drop` | dynamic | — | 9/9 | SAF-MOJAVE=14%, CSZ-CENTRAL-OR=22%, AAK-SHUMAGIN=11%, OK-PAWNEE-PRAGUE=5%, ITALY_APENNINES=9% | MOCRE-2: diferencia MLE reciente vs baseline |
| 4 | OK | R | — | Concentración hacia la falla | `a_fault_concentration` | dynamic | — | 9/9 | limpio | MOCRE-2: cambio de distancia radial media |
| 5 | OK | R | — | Aceleración de magnitud regional | `a_mag_accel` | dynamic | — | 8/9 | limpio | MOCRE-2: magnitud media reciente vs baseline |
| 6 | OK | A | 0.50 | Microsismicidad (tasa M<3) | `a_seis` | dynamic | 4.0 | 9/9 | limpio |  |
| 7 | OK | A | 0.40 | Foreshocks | `a_foreshock` | dynamic | 4.1 | 7/9 | limpio | algorithmic (7d/30d accel), no new data |
| 8 | OK | A | 0.55 | Enjambres sísmicos | `a_swarm` | dynamic | (default in code) | 8/9 | limpio | + E_state EWMA; algorithmic, no new data |
| 9 | WALL | A | 0.47 | Cambios Vp/Vs | `—` | absent | — | — | — | VERIFIED: no published time series exists; dv/v pivot needs Julia+GB waveform cross-correlation infra -- a signal-processing project, not data integration |
| 10 | OK | A | 0.35 | Ruido sísmico ambiental | `a_noise` | dynamic | (default in code) | 6/9 | OK-PAWNEE-PRAGUE=15% | NEW 4-jul (3-jul 'blocked' was a bad-metric error): MUSTANG sample_rms, 6 segs, level-proxy not dv/v, water tier |
| 11 | OK | B | 0.85 | GNSS/GPS interseísmico | `a_gnss` | dynamic | 7.5 | 6/9 | AAK-SHUMAGIN=9% | tau recalibrated 4-jul 2.0->7.5 |
| 12 | OK | B | 0.85 | InSAR | `a_insar` | dynamic | (default in code) | 5/9 | SJC-ANZA=13%, SAF-PARKFIELD=19%, SAF-MOJAVE=13%, CSZ-CENTRAL-OR=7%, AAK-SHUMAGIN=39% | 5/9; LOWEST tier, atmo-noise-limited |
| 13 | OK | B | 0.68 | Inclinómetros/extensómetros (strain) | `a_strain` | dynamic | 6.3 | 4/9 | SJC-ANZA=5%, SAF-PARKFIELD=6% | 4 segments w/ real PB net; tau recalibrated 4-jul 3.0->6.3 |
| 14 | OK | B | 0.60 | Slow Slip Events | `a_sse` | dynamic | 10.9 | 5/9 | limpio | + K_state EWMA; gated OFF for regime=induced (4-jul); tau recal 3.0->10.9 |
| 15 | OK | C | 0.50 | Acumulación de esfuerzo | `C_state` | state | — | — | — | internal accumulator from recurrence + GNSS loading + ΔCFS |
| 16 | OK | C | 0.70 | Transferencia Coulomb (ΔCFS) | `C_state` | state | — | — | — | stress_graph.py point-source approx, feeds C_state |
| 17 | OK | C | 0.90 | Presión de fluidos subterráneos | `a_fluid` | dynamic | 3.0 | 1/9 | OK-PAWNEE-PRAGUE=38% | + F_state; only regime=induced (Oklahoma OCC wells) |
| 18 | PRIOR | C | 0.50 | Acumulación esfuerzo — orientación (World Stress Map) | `—` | static | — | — | — | NEW 4-jul: static SHmax/regime prior per segment (data/stress_map/), NOT a temporal channel; diagnostic in stress_graph.py, not yet in G |
| 19 | WALL | D | 0.25 | Radón | `—` | absent | — | — | — | VERIFIED wall 4-jul: WQP grab-samples end 2014-2018, IRON=Italy-only frozen 2021, Shumagin=0. No continuous open series exists for these segments -- needs field hardware |
| 20 | WALL | D | 0.30 | Helio/metano/hidrógeno | `—` | absent | — | — | — | VERIFIED wall 4-jul: only a static point-compilation (Zenodo HEDB), 0 continuous series; California springs sampled once in 1983/1997/2013, Shumagin/Oklahoma=0 |
| 21 | WAIT | D | 0.35 | Nivel de aguas subterráneas | `a_water` | dynamic | (default in code) | 0/9 | — | code done, USGS OGC API rate-limited, no data harvested |
| 22 | N/A | D | 0.70 | Gases volcánicos (SO2, CO2) | `—` | absent | — | — | — | volcanic context only -- none of the 6 segments |
| 23 | LOWVAL | E | 0.30 | Campo gravitatorio | `—` | absent | — | — | — | VERIFIED 4-jul: GRACE-FO mascon open+current but ~300km res groups Anza+Mojave into 1 cell, monthly cadence, measures water-equiv not tectonic stress -- scale/mechanism mismatch |
| 24 | OK | E | 0.25 | TEC ionosférico | `a_tec` | dynamic | (default in code) | 6/9 | limpio | NEW 4-jul: JPL GIM, 6 segments; real data, contested mechanism |
| 25 | LOWVAL | E | 0.20 | Temperatura superficial (satélite) | `—` | absent | — | — | — | VERIFIED 4-jul: MODIS LST open+current but mu_ev=0.20 lowest, mechanism non-reproducible, Shumagin ~97% cloud-blind -- lowest value/effort of the catalog |
| 26 | LOWVAL | E | 0.20 | Variaciones electromagnéticas | `—` | absent | — | — | — | VERIFIED 4-jul: only INTERMAGNET geomag observatories 100-860km away (measure global field, not local EM); QuakeFinder frozen 2010+closed. Touches only 2/9 segments marginally |
| 27 | OK | F | 0.55 | Historial sísmico | `—` | static | — | — | — | config T_rec_years / years_since_last_major, feeds C_state initial condition |
| 28 | OK | F | 0.75 | Geometría de fallas | `—` | static | — | — | — | real CA-GIS traces + PCA strike; feeds dist_to_trace + stress_graph |
| 29 | OK | + | — | Carga oceánica no-tidal | `a_ocean` | dynamic | (default in code) | 6/9 | limpio | NEW 4-jul, NOT in the audited 25; EOST/ECCO2, water-tier confidence (unaudited mechanism) |

Leyenda estado: `OK`=activa · `PRIOR`=prior estático (no canal temporal) · `WAIT`=código listo sin datos (solo cosecha) · `WALL`=muro real verificado (no existe fuente/infra viable) · `LOWVAL`=accesible pero baja relación valor/esfuerzo (verificado) · `N/A`=no aplica a estos segmentos.

## Recuento
- **20 variables activas** de 23 del catálogo de 25 + 6 adiciones fuera de catálogo (carga oceánica + 5 regionales MOCRE-2).
- **16 canales de anomalía `a_X`** con datos reales en disco.
- 5 estados persistentes: `A_state, C_state, K_state, E_state, F_state`.

## Qué NO está listo (señalizado honestamente)

**Conteos corregidos:** `a_seis` y `a_regional_rate` usan anomalía Poisson causal. `a_swarm` y `a_foreshock` son razones de ventanas activas y conservan normalización robusta con gate de actividad mínima.

**Residuos menores recalibrables** (distribución del segmento difiere del target uniforme, un τ por-segmento lo bajaría): `a_gnss` Shumagin ~9%, `a_insar` Shumagin ~18% (además muestra pequeña, n=9 bins), `a_strain` SJC/Parkfield ~5-6%.

**Muros REALES verificados por descarga real 4-jul (no corazonada — no existe fuente/infra viable, ningún esfuerzo de ingeniería lo crea):**
- Radón (μ_ev 0.25): solo grab-samples que paran en 2014-2018, red continua única (IRON) es Italia congelada 2021, Shumagin sin dato. Requiere hardware de campo propio.
- Helio/metano/H2 (μ_ev 0.30): solo compilado estático de puntos, 0 series continuas; California muestreada 1 vez en 1983/1997/2013.
- Vp/Vs (μ_ev 0.475): no existe serie publicada; el pivote dv/v es un proyecto de procesamiento de señal (Julia + cross-correlation waveform GB-scale), no integración de datos.

**Accesibles pero baja relación valor/esfuerzo (verificado 4-jul — se dejan sin construir por decisión, no por muro):**
- Campo gravitatorio (μ_ev 0.30): GRACE-FO abierto y actual, pero ~300km de resolución agrupa Anza+Mojave en una celda, cadencia mensual, mide agua no esfuerzo tectónico — desajuste de escala y mecanismo.
- Temperatura superficial (μ_ev 0.20, el más bajo): MODIS LST abierto, pero mecanismo no reproducible y Shumagin ~97% ciego por nube.
- Variaciones EM (μ_ev 0.20, el más disputado): solo observatorios geomagnéticos a 100-860km (campo global, no EM local); QuakeFinder congelado 2010 y cerrado. Roza 2/9 segmentos.

**Solo cosecha (código listo, alcanzable):** nivel de aguas subterráneas (`a_water`, API USGS rate-limited — puro tiempo de cosecha, no código).

**N/A a estos segmentos:** gases volcánicos (μ_ev 0.70, solo contexto volcánico — ninguno de los 9 es un volcán).
