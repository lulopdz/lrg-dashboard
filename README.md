# LRG Dashboard (Ontario IESO)

Este proyecto genera un dashboard interactivo web, un simulador y un registro de operaciones (portfolio) para analizar, predecir y hacer seguimiento de resultados en el mercado eléctrico de Ontario (IESO). El sistema monitorea el **Day-Ahead Market (DAM)**, el **Real-Time Market (RTM)**, el *spread* entre ambos, además de recopilar y visualizar datos climáticos, de demanda y de generación eólica.

## Características

- **Datos de Mercado**: Descarga automatizada de precios DAM y RTM directamente de IESO a través de la librería `gridstatusio`.
- **Datos Climáticos y de Red**: Recopilación de pronósticos de demanda, generación eólica y clima mediante `openmeteo_requests`.
- **Modelos Predictivos (Machine Learning)**: `scikit-learn` para predecir precios DAM y RTM del día siguiente, y para el spread, siempre como **DART = DA − RT**, el pronóstico puntual con sus dos días similares más una **señal** por hora: P(DART > 0), P(DART > +$40) y P(DART < −$40) contra sus tasas base, con nivel de convicción ganado en backtest (acierto y $/h), en dos corridas por día: pre-DAM (9:00) y post-DAM (tarde).
- **Trading Simulator**: Backtest interactivo -- elige una fecha pasada, ve solo la información disponible en ese momento, y evalúa tus apuestas Long/Flat/Short contra el spread real.
- **Portfolio**: Registro mensual de las participaciones reales enviadas a IESO (a partir de los reportes XML en `data/reports/`), con el PnL de cada caso y un resumen Ganadas/Perdidas/Sin exposición.
- **Visualización Interactiva**: Gráficos y tablas dinámicas generados con `plotly`, compilados en un formato HTML estático sin necesidad de un backend activo.
- **Automatización**: Pipelines de GitHub Actions para ejecución programada. Refresca datos diarios, entrena modelos y publica directamente en GitHub Pages.

## Estructura del Proyecto

```text
lrg-dashboard/
├── .github/
│   └── workflows/          <-- daily.yml (9:00, pipeline completo), forecast_pm.yml (post-DAM de la tarde),
│                               refresh_rtm.yml (botón Refresh Real-Time) y smoke.yml (prueba en cada push de código)
├── data/                   <-- CSVs guardados con datos históricos, predicciones y metadatos
│   ├── *_forecast_history.csv <-- Archivo de forecasts: una versión por día objetivo y vintage (pre_dam / post_dam)
│   ├── *_TORONTO.* / *_NIAGARA.* <-- Lo mismo para las otras zonas (la de OTTAWA, la del dashboard, no lleva sufijo)
│   ├── forecast_inputs_dayahead.csv <-- Entradas (carga, viento, clima, adecuación) tal como se veían a las 9:00 del día anterior
│   ├── forecast_scorecard.json <-- Evaluación out-of-sample del archivo de forecasts (scorecard.py)
│   └── reports/            <-- Reportes XML de participación (IESO DAScheduledEnergy2), fuente del Portfolio
├── docs/                   <-- Salida de GitHub Pages (no se commitea: se publica como artefacto)
│   ├── index.html          <-- Dashboard principal generado por Plotly/Python
│   ├── simulator.html      <-- Herramienta del simulador
│   └── portfolio.html      <-- Registro de operaciones y PnL mensual
├── src/                    <-- Código Python, en tres etapas: entra el dato, se predice, se dibuja
│   ├── ingest/             <-- Descarga desde las APIs y deja CSVs en data/
│   │   ├── update_*.py     <-- Un script por fuente vía GridStatus u Open-Meteo (DAM, RTM, clima, carga, viento, adecuación)
│   │   ├── update_prices_ieso.py <-- DAM y RT desde los reportes públicos de IESO: mismos valores, 0 requests del cupo
│   │   ├── update_common.py <-- Descarga y fusión incremental que comparten los update_*
│   │   ├── check_freshness.py <-- Falla si el RT quedó viejo (job "verificar")
│   │   └── parse_reports.py <-- Convierte los reportes XML en data/reports/ a data/historical_pnl.csv
│   ├── forecast/           <-- Modelos que proyectan el día siguiente
│   │   ├── forecast_common.py <-- Features, entrenamiento, *backtest* y archivo que comparten los predict_*
│   │   ├── predict_*.py    <-- DAM y RTM (regresión); predict_spread.py hace el pronóstico de DART y la señal (clasificadores)
│   │   ├── run_forecasts.py <-- Corre los tres modelos para todas las zonas en un proceso (lo que llaman los workflows)
│   │   ├── scorecard.py    <-- Puntúa el archivo contra lo que efectivamente se liquidó
│   │   ├── walkforward.py  <-- Reconstruye días pasados con el código actual, para comparar cambios (--jobs N en paralelo)
│   │   ├── vintages.py     <-- Archivo de entradas day-ahead (--rebuild lo reconstruye desde git)
│   │   └── backfill_history.py <-- One-off ya ejecutado; no volver a correrlo (borra las columnas vintage/p10/p90)
│   └── web/                <-- Construye el HTML que se publica en docs/
│       ├── theme.py        <-- Colores y estilo compartidos
│       ├── dashboard_data.py <-- Carga los CSVs de data/ y los deja listos para graficar
│       ├── dashboard_figures.py <-- Construye las figuras de Plotly
│       └── generar_*.py    <-- Ensamblan y exportan index.html, simulator.html y portfolio.html
├── ops/                    <-- Apps Script que dispara el run de las 9:00 y el botón de refresh (ver ops/README.md)
├── requirements.txt        <-- Dependencias requeridas
└── README.md               <-- Este archivo de documentación
```

## Pipelines de Automatización (GitHub Actions)

El dashboard se actualiza bajo tres modalidades; todas comitean los datos nuevos y publican en *GitHub Pages*:

1. **Actualización diaria (`daily.yml`)**: a las 9:00 hora de Ottawa, un solo job corre el pipeline completo en orden: DAM, RT, clima, carga, viento, adecuación, portafolio y P&L (`parse_reports.py`), los forecasts del vintage **pre-DAM** para OTTAWA, TORONTO y NIAGARA (`run_forecasts.py`, más el scorecard) y la regeneración del sitio. Pre-DAM quiere decir que el DAM de mañana todavía no salió y RT/spread usan el DAM pronosticado: es el vintage con el que se puede ofertar antes de las 10:00. El disparo de las 9:00 en punto lo hace un Apps Script externo (ver `ops/README.md`); el cron de GitHub (`0 15 * * *` UTC, 11:00 EDT / 10:00 EST) es solo respaldo y se salta las descargas si ese día ya hubo un run disparado a mano o por el Apps Script. Si fallan los forecasts, el sitio se publica igual con los datos frescos; el job `verificar` pone el run en rojo si el RT quedó con más de 4 h de atraso o si fallaron los forecasts, para que un fallo no pase desapercibido detrás de `continue-on-error`.
2. **Forecast post-DAM (`forecast_pm.yml`)**: por la tarde (cron 19:30 UTC, con un reintento a las 23:30 UTC que no hace nada si el primero completó), una vez que IESO publicó el DAM de mañana (~12:35 EST). Trae ese DAM y el RT de hoy de los reportes públicos de IESO (0 requests del cupo) y repite RT y spread con el DAM real: el vintage **post-DAM**, para decidir exposición en tiempo real.
3. **Refresh Real-Time (`refresh_rtm.yml`)**: bajo demanda, desde el botón *Refresh Real-Time* del dashboard (pestañas RT y Spread) y de la página Portfolio. Trae los últimos intervalos RT publicados (la hora en curso se descarta hasta que tenga sus 12 intervalos de 5 minutos) y el DAM de mañana desde IESO si ya salió, recalcula spreads y el P&L del portafolio y republica. Con la variable `REFRESH_RT_URL` configurada el botón dispara el workflow con un clic; sin ella abre la página de Actions.

Un `push` a `main` (un reporte nuevo, un cambio de código) también corre `daily.yml`, pero sin descargas: solo recalcula el P&L y regenera el sitio. Si un cambio de código necesita datos frescos, se lanza a mano con *Run workflow*. Los push que tocan código corren además `smoke.yml`: compila, puntúa el archivo, reconstruye un día con los tres modelos y genera las tres páginas, sin escribir en `data/`.

**Cupo de API.** GridStatus da 250 requests por mes por cuenta. Una corrida diaria completa gasta hasta 5 (DAM, RT, carga, viento, adecuación; el DAM se salta si ya está al día) y cada clic de *Refresh Real-Time* 1 (el DAM viene gratis de IESO). La corrida post-DAM no gasta nada. Con una descarga diaria y ~2 clics por día se llega a fin de mes; el 20/09/2026 el cupo se agotó porque `daily.yml` corría 3 o 4 veces al día. Si vuelve a pasar, el paso RT muestra `Error 403: API requests limit reached` y el job `verificar` sale en rojo.

Los tres flujos que escriben comparten un grupo de concurrency y clonan la punta de `main` al arrancar (`ref: main`), así que se encolan en vez de pisarse. GitHub solo mantiene un run pendiente por grupo: uno nuevo cancela al que ya esperaba (por eso `forecast_pm.yml` tiene el cron de reintento).

## Evaluar un cambio en los modelos

El backtest de 21 días que imprime cada `predict_*.py` entrena y evalúa con la versión más reciente de las entradas, así que es optimista. Para decidir si un cambio mejora algo:

```bash
python src/forecast/walkforward.py --out /tmp/base --jobs 4     # antes del cambio (o con el código de main)
python src/forecast/walkforward.py --out /tmp/cand --jobs 4     # con el cambio
python src/forecast/scorecard.py --dir /tmp/base --compare /tmp/cand
```

`scorecard.py` sin argumentos puntúa el archivo live (`data/*_history.csv`): lo que el modelo realmente dijo cada día contra lo que se liquidó. Es la cifra que muestra el dashboard como *Out-of-sample record*.

## Configuración y Uso Local

Para correr el proyecto en tu propia máquina:

1. **Crear y activar entorno virtual**:
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar Variable de Entorno**:
   El proyecto usa la API de GridStatus, por lo que necesitas declarar la llave en tu entorno:
   - **Windows (PowerShell)**: `$env:GRIDSTATUS_API_KEY="tu_llave_aqui"`
   - **Mac/Linux**: `export GRIDSTATUS_API_KEY="tu_llave_aqui"`

4. **Actualizar datos (Opcional)**:
   Puedes extraer nueva información corriendo los módulos de recolección:
   ```bash
   python src/ingest/update_data.py
   python src/ingest/update_rtm.py
   python src/ingest/update_weather.py
   python src/ingest/update_load_forecast.py
   python src/ingest/update_wind_forecast.py
   ```
   Y generar las predicciones del día siguiente (requiere los datos anteriores ya actualizados). Ojo: escribe en `data/` y archiva el día como si fuera la corrida live, así que en local conviene reconstruir un día pasado a otra carpeta en su lugar:
   ```bash
   python src/forecast/run_forecasts.py                      # live: los tres modelos, todas las zonas, y el scorecard
   python src/forecast/predict_rtm.py --target-date 2026-09-20 --out-dir /tmp/prueba   # un día pasado, sin tocar data/
   ```

5. **Procesar reportes de participación (Opcional)**:
   Si agregaste un nuevo reporte XML de IESO a `data/reports/`, procésalo para actualizar el PnL histórico:
   ```bash
   python src/ingest/parse_reports.py
   ```

6. **Generar el Dashboard**:
   ```bash
   python src/web/generar_web.py
   python src/web/generar_simulator.py
   python src/web/generar_portfolio.py
   ```
   Una vez terminado, abre los archivos generados (`docs/index.html`, `docs/simulator.html` y `docs/portfolio.html`) en tu navegador para ver la interfaz actualizada localmente.