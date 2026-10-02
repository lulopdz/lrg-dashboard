@echo off
rem Doble clic: sube los reportes nuevos de data\reports\ al portfolio (ver subir_reportes.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0subir_reportes.ps1"
echo.
pause
