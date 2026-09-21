# ops: disparadores externos

GitHub Actions no puede ni correr a las 9:00 hora de Ottawa en punto (cron solo en UTC, y entre el 11 y el 17/09/2026 llegó 3 a 6 h tarde cada día) ni exponer un botón que dispare un workflow sin pasar por la página de Actions. `refresh_trigger.gs` resuelve las dos cosas desde una cuenta de Google.

## Despliegue (una vez, ~10 min)

1. **Token**: GitHub > Settings > Developer settings > Fine-grained tokens > Generate. Repo: `lrg-dashboard`. Permiso: *Actions: Read and write*. Copiar el token.
2. **Script**: [script.google.com](https://script.google.com) > Nuevo proyecto > pegar `refresh_trigger.gs`.
   - Project settings > Time zone: **America/Toronto**.
   - Project settings > Script properties: `GITHUB_TOKEN` = el token.
3. **9:00 diario**: en el editor, ejecutar `armDaily` una vez (autorizar cuando lo pida). Verificar en *Triggers* que aparece `runDaily` para mañana a las 9:00. Se re-arma solo cada día.
4. **Botón**: Deploy > New deployment > Web app. Execute as: *Me*. Who has access: *Anyone*. Copiar la URL `.../exec`.
5. **Variable**: GitHub > repo > Settings > Secrets and variables > Actions > Variables > `REFRESH_RT_URL` = esa URL. El sitio la incorpora en la siguiente regeneración; hasta entonces el botón abre la página de Actions.

## Notas

- La URL del botón queda en el HTML público: quien la tenga puede pedir un refresh. Solo dispara `refresh_rtm.yml`, que no acepta parámetros y se encola con el grupo de concurrency, así que lo peor es una corrida de más, que cuesta 1 o 2 de los 250 requests mensuales de GridStatus; si aparecen refreshes que nadie pidió, redeployar el web app para cambiar la URL.
- Si el token expira, `dispatch` devuelve 401/403 y el botón lo muestra. Renovarlo en Script properties.
- El cron `0 15 * * *` de `daily.yml` queda como respaldo por si el trigger de Apps Script falla: corre después de las 9:00 y, si ese día ya hubo un run por `workflow_dispatch`, no descarga nada (cada descarga completa son 5 de los 250 requests mensuales de GridStatus).
