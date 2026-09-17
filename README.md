# LRG Dashboard (Ontario IESO)

Este proyecto genera un dashboard interactivo web, un simulador y un registro de operaciones (portfolio) para analizar, predecir y hacer seguimiento de resultados en el mercado eléctrico de Ontario (IESO). El sistema monitorea el **Day-Ahead Market (DAM)**, el **Real-Time Market (RTM)**, el *spread* entre ambos, además de recopilar y visualizar datos climáticos, de demanda y de generación eólica.

## Características

- **Datos de Mercado**: Descarga automatizada de precios DAM y RTM directamente de IESO a través de la librería `gridstatusio`.
- **Datos Climáticos y de Red**: Recopilación de pronósticos de demanda, generación eólica y clima mediante `openmeteo_requests`.
- **Modelos Predictivos (Machine Learning)**: Uso de algoritmos con `scikit-learn` para predecir precios DAM, RTM y spreads del día siguiente.
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
│   │   └── predict_*.py    <-- Un script por serie (DAM, RTM, spread)
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

1. **Actualización diaria (`daily.yml`)**: a las 9:00 hora de Ottawa, un solo job corre el pipeline completo en orden: DAM, RT, clima, carga, viento, adecuación, portafolio y P&L (`parse_reports.py`), los tres forecasts (DAM, RT, spread) y la regeneración del sitio. El disparo de las 9:00 en punto lo hace un Apps Script externo (ver `ops/README.md`); el cron de GitHub (`23 7 * * *` UTC) es solo respaldo, porque GitHub lo entrega con horas de retraso y no sigue el horario de Ottawa.
2. **Refresh Real-Time (`refresh_rtm.yml`)**: bajo demanda, desde el botón *Refresh Real-Time* del dashboard (pestañas RT y Spread) y de la página Portfolio. Trae los últimos intervalos RT publicados (y el DAM, por si ya salió el de mañana), recalcula spreads y el P&L del portafolio y republica. Con la variable `REFRESH_RT_URL` configurada el botón dispara el workflow con un clic; sin ella abre la página de Actions.

Un `push` de código también corre `daily.yml`. Los dos flujos comparten un grupo de concurrency y clonan la punta de `main` al arrancar (`ref: main`), así que se encolan en vez de pisarse. GitHub solo mantiene un run pendiente por grupo: uno nuevo cancela al que ya esperaba.

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