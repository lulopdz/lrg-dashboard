import pandas as pd
import plotly.express as px
import os

# 1. Cargar los datos históricos
df = pd.read_csv('data/ieso_dam_prices.csv')

# 2. Graficar el precio (lmp) en el tiempo, una línea por zona
fig = px.line(df, x='interval_start_local', y='lmp', color='location',
              title='Dashboard DAM - Reporte Diario',
              template='plotly_dark')  # Un estilo oscuro se ve más profesional

# 3. Asegurarnos de que la carpeta 'docs' exista
os.makedirs('docs', exist_ok=True)

# 4. Exportar el gráfico como una página web HTML estática
fig.write_html('docs/index.html')
print("¡Dashboard generado exitosamente en docs/index.html!")
