// Google Apps Script: the two things GitHub can't do on its own for lrg-dashboard.
//   1. Fire daily.yml at 9:00 sharp, Ottawa time (GitHub cron is UTC only and arrives hours late).
//   2. Serve a URL the site's "Refresh Real-Time" button can hit to fire refresh_rtm.yml.
// Setup: see ops/README.md. Script properties: GITHUB_TOKEN (fine-grained PAT, Actions: write).

const OWNER = 'lulopdz';
const REPO = 'lrg-dashboard';
const DAILY_HOUR = 9;  // project timezone must be America/Toronto (File > Project settings)

function dispatch(workflow) {
  const token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  const res = UrlFetchApp.fetch(
    `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${workflow}/dispatches`, {
      method: 'post',
      contentType: 'application/json',
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' },
      payload: JSON.stringify({ ref: 'main' }),
      muteHttpExceptions: true,
    });
  return res.getResponseCode();  // 204 = accepted
}

// Button endpoint (deployed as a web app, "Anyone"). GET so a plain link works too.
function doGet() {
  const code = dispatch('refresh_rtm.yml');
  return ContentService.createTextOutput(code === 204 ? 'Refresh Real-Time requested' : `GitHub said ${code}`);
}

// 9:00 run. Exact-minute triggers only exist as one-shots, so each run re-arms tomorrow's
// inside finally so a transient dispatch error never breaks the chain.
// A rejected dispatch (e.g. 401/403 from an expired token) throws, so the execution shows as
// Failed and Google emails the owner; without it the run would look Completed.
function runDaily() {
  try {
    const code = dispatch('daily.yml');
    if (code !== 204) throw new Error(`GitHub dispatch of daily.yml failed: HTTP ${code}`);
  } finally {
    armDaily();
  }
}

function armDaily() {
  ScriptApp.getProjectTriggers()
    .filter(t => t.getHandlerFunction() === 'runDaily')
    .forEach(t => ScriptApp.deleteTrigger(t));
  const next = new Date();
  next.setHours(DAILY_HOUR, 0, 0, 0);
  if (next <= new Date()) next.setDate(next.getDate() + 1);
  ScriptApp.newTrigger('runDaily').timeBased().at(next).create();
}

// Self-healing backup: run armBackupOnce() once in the editor to add an 8:00 AM daily check
// that re-arms runDaily if Google ever drops a one-shot trigger (e.g. around DST changes).
function ensureDailyArmed() {
  // Always recreate today's 9:00 AM one-shot (recovers even if a stale un-fired trigger sits in Triggers).
  armDaily();
}

function armBackupOnce() {
  ScriptApp.getProjectTriggers()
    .filter(t => t.getHandlerFunction() === 'ensureDailyArmed')
    .forEach(t => ScriptApp.deleteTrigger(t));
  ScriptApp.newTrigger('ensureDailyArmed')
    .timeBased()
    .everyDays(1)
    .atHour(DAILY_HOUR - 1)
    .create();
  armDaily();
}
