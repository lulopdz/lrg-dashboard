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
│   └── workflows/          <-- Tareas programadas (dashboard.yml, predict.yml, refresh_rtm.yml)
├── data/                   <-- CSVs guardados con datos históricos, predicciones y metadatos
│   └── reports/            <-- Reportes XML de participación (IESO DAScheduledEnergy2), fuente del Portfolio
├── docs/                   <-- Carpeta raíz para GitHub Pages
│   ├── index.html          <-- Dashboard principal generado por Plotly/Python
│   ├── simulator.html      <-- Herramienta del simulador
│   └── portfolio.html      <-- Registro de operaciones y PnL mensual
├── src/                    <-- Scripts principales de Python
│   ├── dashboard_*.py      <-- Lógica para procesar datos y construir las gráficas
│   ├── generar_*.py        <-- Scripts que ensamblan y exportan los archivos HTML de la web
│   ├── predict_*.py        <-- Scripts para generar predicciones y *backtests*
│   ├── parse_reports.py    <-- Convierte los reportes XML en data/reports/ a data/historical_pnl.csv
│   └── update_*.py         <-- Scripts para la descarga de datos desde las distintas APIs
├── requirements.txt        <-- Dependencias requeridas
└── README.md               <-- Este archivo de documentación
```

## Pipelines de Automatización (GitHub Actions)

El proyecto se ejecuta de forma autónoma gracias a los siguientes flujos de trabajo configurados para comitear los datos nuevos y publicar en *GitHub Pages*:

1. **Actualizar Dashboard DAM Diariamente (`dashboard.yml`)**: Ejecutado todos los días a las 10:00 UTC (6:00 AM EDT). Descarga los datos DAM, clima, carga y viento; procesa los reportes de participación (`parse_reports.py`) y regenera el HTML interactivo, el simulador y el portfolio.
2. **Refrescar RTM (`refresh_rtm.yml`)**: Ejecutado junto con el DAM a las 10:00 UTC. Actualiza los valores reales del mercado RTM y reconstruye la web. Puede lanzarse manualmente para forzar una actualización puntual.
3. **Predicciones DAM/RTM/Spread (`predict.yml`)**: Ejecutado todos los días a las 10:30 UTC. Aprovecha que los flujos anteriores ya han renovado las variables para lanzar los modelos de IA sobre datos frescos y proyectar el día de mañana.

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
   python src/update_data.py
   python src/update_rtm.py
   python src/update_weather.py
   python src/update_load_forecast.py
   python src/update_wind_forecast.py
   ```
   Y generar las predicciones del día siguiente (requiere los datos anteriores ya actualizados):
   ```bash
   python src/predict_dam.py
   python src/predict_rtm.py
   python src/predict_spread.py
   ```

5. **Procesar reportes de participación (Opcional)**:
   Si agregaste un nuevo reporte XML de IESO a `data/reports/`, procésalo para actualizar el PnL histórico:
   ```bash
   python src/parse_reports.py
   ```

6. **Generar el Dashboard**:
   ```bash
   python src/generar_web.py
   python src/generar_simulator.py
   python src/generar_portfolio.py
   ```
   Una vez terminado, abre los archivos generados (`docs/index.html`, `docs/simulator.html` y `docs/portfolio.html`) en tu navegador para ver la interfaz actualizada localmente.