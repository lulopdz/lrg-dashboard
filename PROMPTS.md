# Prompts para seguir el trabajo

Prompts listos para pegar en una sesión nueva de Claude Code sobre este repo (`lulopdz/lrg-dashboard`).
Una sesión nueva no recuerda las anteriores: cada prompt trae el contexto que necesita y apunta a
las secciones del README donde está el resto. Si la tarea también usa los datos de
[ontario-grid](https://github.com/lulopdz/ontario-grid), agrega ese repo a la sesión.

Estado al 2026-11-01: en producción desde el 31-oct, 5 picks por día (★1–★5), features de la
rampa de la tarde, el veto del clasificador solo en load, y el modelo RT ajustado sobre RT
recortado a ±$300. En el walk-forward completo, `picks_5` sumó $15,894 en los días con las
entradas de las 9:00 (las cuatro zonas) y $96,217 en el resto.

## Reglas que valen para todos

Pégalas al final de cualquier prompt de modelos si la sesión no las sigue sola:

> Antes de cambiar un modelo, lee en el README "Evaluar un cambio". La métrica es el P&L de
> `picks_5` y `top5_fc` (strategies.py), no el MAE ni el % de acierto. Prueba primero con el banco
> rápido (`src/forecast/bench.py`) y confirma lo que gane con `walkforward.py --all-days` en las
> cuatro zonas (~80 min) contra el código de `main`. Un cambio entra solo si mejora en al menos 3
> de 4 zonas, tanto en los días con las entradas de las 9:00 como en el resto. CI corre Python
> 3.11: nada de f-strings con comillas iguales anidadas. Trabaja en una rama y abre un PR; no
> reescribas historia ya subida.

## 1. Revisión del rendimiento en vivo (cada 2–4 semanas)

> Revisa cómo rinden en vivo los modelos que entraron a producción el 2026-10-31 (rampa de la
> tarde, picks sin veto en gen, RT recortado a ±$300, 5 picks por día). Corre
> `python src/forecast/scorecard.py --zone all` y mira la sección `strategies`, segmento `live`,
> vintage `pre_dam`, solo los días desde 2026-11-01: P&L de `picks` (los ★ que se publicaron),
> $/hora, acierto, peor día y drawdown, por zona. Compáralo con lo que prometía el walk-forward
> (picks_5 ≈ $15,894 en 106 días con entradas de las 9:00: ~$150/día entre las cuatro zonas,
> ~$37/día por zona) y con los baselines `always_gen` y `follow_sign` de los mismos días. Dime si va en
> línea, mejor o peor, con el IC90, y si algo merece investigarse. No cambies código.

## 2. Fase 1, punto 4: un solo modelo para las cuatro zonas

> Fase 1, punto 4 del plan: probar un modelo de RT (y del clasificador P(DART > 0)) entrenado
> con las cuatro zonas juntas y la zona como feature, en lugar de uno por zona. La idea: el DART
> de las zonas se mueve junto (OTTAWA–TORONTO 0.87, TORONTO–NIAGARA 0.94, NORTHWEST ~0.6) y el
> modelo vería 4 veces más ejemplos de los picos raros que deciden el P&L. El banco
> (`src/forecast/bench.py`) hoy corre zona por zona: agrégale un modo que arme las grillas de
> todas las zonas, entrene una vez por bloque y prediga cada zona. Compara contra `current` con
> el veredicto de 3 de 4 zonas; si gana con claridad, impleméntalo en predict_rtm/predict_spread
> y confírmalo con el walk-forward completo.

## 3. Fase 1, punto 5: quitar los lags del DART y del RT

> Fase 1, punto 5: el DART casi no tiene memoria (correlación con la misma hora del día anterior
> 0.05, con el promedio diario 0.17), así que los lags (`*_lag_2d`, `*_lag_7d`, `*_roll_7d`,
> `*_roll_28d`, `*_today_early_*`) probablemente solo meten ruido en el modelo RT y en el
> clasificador. Agrega al banco variantes que los quiten (todos, y por grupos) y compáralas con
> `current`. Lo que gane en 3 de 4 zonas, confírmalo con el walk-forward completo.

## 4. Fase 2: datos nuevos de ontario-grid como features

> Fase 2: usar como features los datos que `lulopdz/ontario-grid` archiva cada hora desde octubre
> de 2026 (agrega ese repo a la sesión; los CSV están en su `data/`, documentados en su README).
> Candidatos: posiciones virtuales del resto del mercado (`virtual_transactions/`), salidas de
> transmisión planificadas (`transmission_outages.csv`, con `first_seen`), pronóstico eólico y
> solar de IESO (`vg_forecast/`) y producción por generador (`generator_output/`).
> Ojo con el momento en que cada dato existe: la oferta pre-DAM se hace a las 9:00 del día
> anterior. Las virtuales del día D salen con el DAM (~12:35 del día anterior), así que a las
> 9:00 solo se conocen las de D-1. El vg_forecast archivado es el último antes de empezar el día
> (~23:57), posterior a las 9:00: para usarlo hay que archivar primero en ontario-grid la versión
> vigente a las 9:00. Una salida de transmisión solo cuenta si su `first_seen` es anterior a la
> corrida. La historia empieza en ago–oct de 2026, así que evalúa solo sobre ese período y dime
> cuántos días hay; si son muy pocos para concluir, dilo y propón esperar.

## 5. Fase 3: cuántos MW poner en cada pick

> Fase 3: hoy cada pick es 1 MW. Propón y evalúa reglas para repartir el colateral entre los
> ★1–★5 según la confianza (|pronóstico|, P(DART > 0), lado gen o load: en el walk-forward las
> posiciones load rindieron ~$1.3/h contra ~$7.2/h de gen). Usa el archivo de forecasts
> reconstruido (walk-forward `--all-days`) para medir P&L, peor día y drawdown de cada regla con
> el mismo presupuesto total de MW. No cambies producción sin mostrarme la comparación.

## 6. Hueco de RT del 2026-01-28 HE16

> Falta en `data/ieso_rtm_prices.csv` la hora 2026-01-28 15:00 (-05:00) en las nueve zonas. IESO
> ya no tiene ese reporte; hay que traerlo de GridStatus. Necesitas `GRIDSTATUS_API_KEY` en el
> entorno y acceso de red a `api.gridstatus.io`. Haz una sola consulta para esa hora (no
> `update_rtm.py --hours`, que pediría miles de horas), inserta solo las 9 filas sin reescribir el
> resto del CSV, y abre un PR.

## 7. Si algo se rompe

> El workflow `<nombre>` de lrg-dashboard falló (o el sitio no se actualizó). Revisa los logs del
> último run, reproduce el error en local con Python 3.11 y corrígelo en una rama con PR. Si la
> causa es externa (IESO o GridStatus caídos, cupo de GridStatus agotado), dímelo en lugar de
> cambiar código.
