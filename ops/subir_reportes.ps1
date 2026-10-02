# Sube los reportes nuevos de IESO al portfolio. Se corre con doble clic en subir_reportes.bat.
#
# Los XML de data/reports/ nunca se suben (son confidenciales y la carpeta está en .gitignore):
# parse_reports.py --positions-only los suma a data/portfolio_positions.csv y solo ese archivo
# se comitea. El push dispara daily.yml, que recalcula el P&L y republica el sitio en ~3 min.
# El bot nunca toca portfolio_positions.csv (en Actions no hay XML), así que este commit y los
# del bot no pueden chocar.

$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$positions = 'data/portfolio_positions.csv'
$site = 'https://lulopdz.github.io/lrg-dashboard/portfolio.html'

function Fail($msg) {
    Write-Host "`nERROR: $msg" -ForegroundColor Red
    exit 1
}

# Git escribe su progreso en stderr; en PowerShell 5.1 eso no es un error, solo cuenta el exit code.
function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { Fail "git $($args -join ' ') falló (código $LASTEXITCODE)" }
}

$branch = git rev-parse --abbrev-ref HEAD
if ($branch -ne 'main') { Fail "estás en la rama '$branch'; cambia a main y vuelve a correrlo" }

$xmls = @(Get-ChildItem 'data/reports' -Filter '*.xml' -ErrorAction SilentlyContinue)
if ($xmls.Count -eq 0) { Fail 'no hay XML en data/reports/' }

Write-Host '1/3 Trayendo lo último de GitHub...'
# Drive cambia las fechas de los archivos y git los marca como modificados sin que lo estén, y
# eso frena el pull. Volver a agregar los que no tienen diff real refresca el índice sin cambiar
# contenido; un cambio de verdad (código que estés editando) queda intacto y lo cubre --autostash.
$real = @(git diff --name-only)
git status --porcelain | Where-Object { $_ -like ' M *' } | ForEach-Object { $_.Substring(3) } |
    Where-Object { $real -notcontains $_ } | ForEach-Object { & git add -- $_ }
Invoke-Git pull -q --rebase --autostash origin main

Write-Host "2/3 Leyendo $($xmls.Count) reportes de data/reports/..."
& python src/ingest/parse_reports.py --positions-only
if ($LASTEXITCODE -ne 0) { Fail 'parse_reports.py falló' }

Invoke-Git add -- $positions
& git diff --cached --quiet -- $positions
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nNo hay reportes nuevos: el portfolio ya está al día." -ForegroundColor Green
    exit 0
}

$dates = @(git diff --cached -U0 -- $positions | Select-String '^\+(\d{4}-\d{2}-\d{2}),' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Sort-Object -Unique)
if ($dates.Count -eq 0) { $msg = 'Reportes de participación (reemisión)' }
elseif ($dates.Count -le 5) { $msg = "Reportes $($dates -join ', ')" }
else { $msg = "Reportes $($dates[0]) a $($dates[-1]) ($($dates.Count) días)" }

Write-Host "3/3 Subiendo: $msg"
Invoke-Git commit -q -m $msg -- $positions
& git push -q origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    # El bot pusheó entre el pull y el push: se rebasa encima (no toca este archivo) y se reintenta.
    Write-Host 'GitHub tenía un commit nuevo del bot; reintentando encima de ese...' -ForegroundColor Yellow
    Invoke-Git pull -q --rebase --autostash origin main
    Invoke-Git push -q origin HEAD:main
}

Write-Host "`nListo. El portfolio se republica en ~3 min: $site" -ForegroundColor Green
