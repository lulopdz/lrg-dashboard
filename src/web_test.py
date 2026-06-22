import pandas as pd
import plotly.express as px
import os

# 1. Cargar los datos históricos
# Nota: Si tus columnas tienen otro nombre (ej. 'Fecha' en vez de 'Date'), cámbialas abajo.
df = pd.read_csv('data/historico/ieso_dam_prices.csv')

# Para este ejemplo, asumo que tus columnas se llaman 'Date' y 'Price'
# Si se llaman diferente, solo modifica los valores de x e y aquí abajo:
fig = px.line(df, x=df.columns[0], y=df.columns[1], 
              title='Dashboard DAM - Reporte Diario',
              template='plotly_dark') # Un estilo oscuro se ve más profesional

# 2. Asegurarnos de que la carpeta 'docs' exista
os.makedirs('docs', exist_ok=True)

# 3. Exportar el gráfico como una página web HTML estática
fig.write_html('docs/index.html')
print("¡Dashboard generado exitosamente en docs/index.html!")