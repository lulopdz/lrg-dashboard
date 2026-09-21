# LRG Dashboard (Ontario IESO)

Este proyecto genera un dashboard interactivo web, un simulador y un registro de operaciones (portfolio) para analizar, predecir y hacer seguimiento de resultados en el mercado eléctrico de Ontario (IESO). El sistema monitorea el **Day-Ahead Market (DAM)**, el **Real-Time Market (RTM)**, el *spread* entre ambos, además de recopilar y visualizar datos climáticos, de demanda y de generación eólica.

## Características

- **Datos de Mercado**: Descarga automatizada de precios DAM y RTM directamente de IESO a través de la librería `gridstatusio`.
- **Datos Climáticos y de Red**: Recopilación de pronósticos de demanda, generación eólica y clima mediante `openmeteo_requests`.
- **Modelos Predictivos (Machine Learning)**: `scikit-learn` para predecir precios DAM y RTM del día siguiente, y para el spread, siempre como **DART = DA − RT**, el pronóstico puntual con sus dos días similares más una **señal** por hora: P(DART > 0), P(DART > +$40) y P(DART < −$40) contra sus tasas base, con nivel de convicción ganado en backtest (acierto y $/h).
- **Trading Simulator**: Backtest interactivo -- elige una fecha pasada, ve solo la información disponible en ese momento, y evalúa tus apuestas Long/Flat/Short contra el spread real.
- **Portfolio**: Registro mensual de las participaciones reales enviadas a IESO (a partir de los reportes XML en `data/reports/`), con el PnL de cada caso y un resumen Ganadas/Perdidas/Sin exposición.
- **Visualización Interactiva**: Gráficos y tablas dinámicas generados con `plotly`, compilados en un formato HTML estático sin necesidad de un backend activo.
- **Automatización**: Pipelines de GitHub Actions para ejecución programada. Refresca datos diarios, entrena modelos y publica directamente en GitHub Pages.

## Estructura del Proyecto

```text
lrg-dashboard/
├── .github/
│   └── workflows/          <-- daily.yml (9:00 Ottawa, pipeline completo) y refresh_rtm.yml (botón Refresh Real-Time)
├── data/                   <-- CSVs guardados con datos históricos, predicciones y metadatos
│   ├── *_forecast_history.csv <-- Archivo de forecasts, una versión por día objetivo (backfill: src/forecast/backfill_history.py)
│   └── reports/            <-- Reportes XML de participación (IESO DAScheduledEnergy2), fuente del Portfolio
├── docs/                   <-- Carpeta raíz para GitHub Pages
│   ├── index.html          <-- Dashboard principal generado por Plotly/Python
│   ├── simulator.html      <-- Herramienta del simulador
│   └── portfolio.html      <-- Registro de operaciones y PnL mensual
├── src/                    <-- Código Python, en tres etapas: entra el dato, se predice, se dibuja
│   ├── ingest/             <-- Descarga desde las APIs y deja CSVs en data/
│   │   ├── update_*.py     <-- Un script por fuente (DAM, RTM, clima, carga, viento, adecuación)
│   │   ├── update_common.py <-- Descarga y fusión incremental que comparten los update_*
│   │   └── parse_reports.py <-- Convierte los reportes XML en data/reports/ a data/historical_pnl.csv
│   ├── forecast/           <-- Modelos que proyectan el día siguiente
│   │   ├── forecast_common.py <-- Features, entrenamiento y *backtest* que comparten los predict_*
│   │   └── predict_*.py    <-- DAM y RTM (regresión); predict_spread.py es el clasificador de señal (--backfill reconstruye el histórico walk-forward)
│   └── web/                <-- Construye el HTML que se publica en docs/
│       ├── theme.py        <-- Colores y estilo compartidos
│       ├── dashboard_data.py <-- Carga los CSVs de data/ y los deja listos para graficar
│       ├── dashboard_figures.py <-- Construye las figuras de Plotly
│       └── generar_*.py    <-- Ensamblan y exportan index.html, simulator.html y portfolio.html
├── requirements.txt        <-- Dependencias requeridas
└── README.md               <-- Este archivo de documentación
```

## Pipelines de Automatización (GitHub Actions)

El dashboard se actualiza bajo dos modalidades, ambas comitean los datos nuevos y publican en *GitHub Pages*:

1. **Actualización diaria (`daily.yml`)**: a las 9:00 hora de Ottawa, un solo job corre el pipeline completo en orden: DAM, RT, clima, carga, viento, adecuación, portafolio y P&L (`parse_reports.py`), los forecasts DAM y RT y la señal de spread y la regeneración del sitio. El disparo de las 9:00 en punto lo hace un Apps Script externo (ver `ops/README.md`); el cron de GitHub (`0 15 * * *` UTC, 11:00 EDT / 10:00 EST) es solo respaldo y se salta las descargas si ese día ya hubo un run disparado a mano o por el Apps Script. Al final, un job `verificar` pone el run en rojo si el RT quedó con más de 4 h de atraso, para que un fallo de GridStatus no pase desapercibido detrás de `continue-on-error`.
2. **Refresh Real-Time (`refresh_rtm.yml`)**: bajo demanda, desde el botón *Refresh Real-Time* del dashboard (pestañas RT y Spread) y de la página Portfolio. Trae los últimos intervalos RT publicados (y el DAM solo si ya pasaron las 13:30 EST y el de mañana todavía no está guardado), recalcula spreads y el P&L del portafolio y republica. Con la variable `REFRESH_RT_URL` configurada el botón dispara el workflow con un clic; sin ella abre la página de Actions.

Un `push` a `main` (un reporte nuevo, un cambio de código) también corre `daily.yml`, pero sin descargas: solo recalcula el P&L y regenera el sitio. Si un cambio de código necesita datos frescos, se lanza a mano con *Run workflow*.

**Cupo de API.** GridStatus da 250 requests por mes por cuenta. Una corrida diaria completa gasta hasta 5 (DAM, RT, carga, viento, adecuación; el DAM se salta si ya está al día) y cada clic de *Refresh Real-Time* 1 o 2. Con una descarga diaria y ~2 clics por día se llega a fin de mes; el 20/09/2026 el cupo se agotó porque `daily.yml` corría 3 o 4 veces al día. Si vuelve a pasar, el paso RT muestra `Error 403: API requests limit reached` y el job `verificar` sale en rojo.

Los dos flujos comparten un grupo de concurrency y clonan la punta de `main` al arrancar (`ref: main`), así que se encolan en vez de pisarse. GitHub solo mantiene un run pendiente por grupo: uno nuevo cancela al que ya esperaba.

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
   Y generar las predicciones del día siguiente (requiere los datos anteriores ya actualizados):
   ```bash
   python src/forecast/predict_dam.py
   python src/forecast/predict_rtm.py
   python src/forecast/predict_spread.py
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