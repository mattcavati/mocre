# MOCRE-2 — Inferencia fuzzy del estado crítico

## Hipótesis

El terremoto no se observa antes de ocurrir, pero un proceso preparatorio puede dejar evidencia
parcial en clústeres físicos distintos. MOCRE-2 no aprende directamente `día → terremoto`; infiere
el estado latente `normal → preparación → crítico` y solo después permitiría entrenar IA3.

## Arquitectura ejecutada

1. **IA1 por clúster:** regional, sismicidad local, deformación, mecánica/fluidos, hidrogeología y
   teledetección. Cada clúster produce nivel alto, quietud, tendencia, persistencia, coherencia y
   cobertura. Missingness limita la regla y nunca equivale a neutralidad.
2. **Coincidencia fuzzy:** t-norma `min` entre clústeres; dentro de un clúster se usa noisy-OR para
   acumular evidencia sin que `max` oculte los demás sensores.
3. **IA2 recurrente:** ataque rápido al aparecer evidencia y relajación lenta; solo usa valores
   disponibles hasta el día de emisión.
4. **Consecuentes aprendidos:** pesos no negativos con regularización, fit cronológico; el umbral
   se fija usando exclusivamente controles de calibración.
5. **IA3 bloqueada:** ocurrencia, magnitud, cuándo y dónde no se entrenan/publican hasta que el
   estado crítico supere todos los criterios de promoción.

## Nuevas observaciones regionales

- anomalía Poisson de actividad regional;
- expansión espacial aproximada mediante ocupación de celdas;
- caída causal de b-value;
- concentración de actividad hacia la falla;
- aceleración de magnitud;
- confirmación entre actividad regional, sismicidad local, carga y deformación.

## Unidad experimental

Los datos diarios no se cuentan como observaciones independientes. Se declusterizan mainshocks a
45 días y cada terremoto pesa uno aunque contribuya con lead-times de 7, 14 y 30 días. Cada evento
se compara con tres controles de la misma región/temporada separados al menos 180 días de cualquier
target. Split por orden de eventos: 60% fit, 20% calibración, 20% test.

## Contrato de promoción

- ≥20 terremotos independientes en test;
- AUC episódico ≥0.70;
- mediana leave-one-region-out ≥0.60;
- falsas alarmas en controles ≤5%;
- sensibilidad por evento ≥70%;
- falsificación desplazando fechas un año AUC ≤0.58.

El modelo solo sale de sombra si cumple simultáneamente todo el contrato definido en
`config/critical_state_contract.json`.

## Resultado actual

Dataset: 90 terremotos independientes y 531 muestras ponderadas. La primera versión local obtuvo
AUC test 0.455. Al incorporar propagación regional, b-value, concentración y aceleración, la
generalización entre regiones mejoró (mediana leave-region-out 0.64–0.66), pero el test temporal
continuó por debajo del azar (AUC 0.40–0.43), sensibilidad 16.7% y falsas alarmas 11–12.5%.

Por tanto, `promoted=false` y `status=shadow_research`. El resultado indica que las señales
catalogales regionales cambian entre regiones de forma transferible, pero no se mantienen en los
eventos futuros de cada región. La siguiente fuente de información necesaria es observación física
de mayor resolución: dv/v desde formas de onda, strain continuo y deformación espacial invertida;
no más optimización sobre las mismas columnas.

## Prospectividad

`emit_critical_shadow.py` escribe el estado, cobertura y reglas dominantes por segmento. Cada salida
se archiva de forma inmutable en `out/prospective-critical/`. El mapa lo carga dinámicamente y lo
etiqueta como índice fuzzy, nunca como probabilidad o alerta. El timer diario actualiza catálogos
regionales, reconstruye estados causales y publica el JSON atómicamente.
