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

// 9:00 run. Exact-minute triggers only exist as one-shots, so each run re-arms tomorrow's.
function runDaily() {
  dispatch('daily.yml');
  armDaily();
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
