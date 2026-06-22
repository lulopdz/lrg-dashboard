# lrg-dashboard

lrg-dam-dashboard/
├── .github/
│   └── workflows/          <-- Aquí vivirá el archivo .yml que automatiza el reporte diario
├── data/
│   ├── historico/          <-- ¡Aquí debes guardar los datos que ya tomaste antes! (Ej: .csv)
│   └── procesado/          <-- Datos limpios o temporales que tu código genere
├── docs/                   <-- Esta será la carpeta pública para la web
│   └── index.html          <-- El dashboard final que generará Plotly
├── src/                    <-- Todos tus scripts de Python van aquí
│   ├── descargar_datos.py  <-- Script para GridStatusIO
│   ├── calcular_spreads.py <-- Lógica de tu algoritmo
│   └── generar_web.py      <-- El que junta todo y crea el index.html
├── venv/                   <-- Tu entorno virtual (ya lo creaste)
├── .gitignore              <-- ARCHIVO VITAL (te explico abajo)
├── requirements.txt        <-- Tus librerías (ya lo creaste)
└── README.md               <-- Explicación de tu proyecto