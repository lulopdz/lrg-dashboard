"""The Refresh buttons shared by index.html and portfolio.html.

Real-Time is the one-click one: with REFRESH_RT_URL set (repo variable pointing at the
ops/refresh_trigger.gs web app) the click dispatches refresh_rtm.yml from the page itself.
Without it, and for every other button, the link opens the workflow's GitHub Actions page.
No pandas here on purpose: generar_portfolio.py imports this and must stay light."""
import os

GITHUB_OWNER = 'lulopdz'
GITHUB_REPO = 'lrg-dashboard'
DAILY_WORKFLOW = 'daily.yml'
RT_WORKFLOW = 'refresh_rtm.yml'
REFRESH_RT_URL = os.environ.get('REFRESH_RT_URL', '').strip()


def actions_url(workflow):
    return f'https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/{workflow}'


def refresh_target(workflow, label):
    """{href, label, direct}; direct=True means a click fires the workflow without leaving the page."""
    if workflow == RT_WORKFLOW and REFRESH_RT_URL:
        return {'href': REFRESH_RT_URL, 'label': label, 'direct': True}
    return {'href': actions_url(workflow), 'label': label, 'direct': False}


# Page JS: wireRefresh(btn, target) sets a button up; direct targets GET the trigger URL
# (no-cors: the Apps Script reply is opaque, the request still lands) and show progress text.
REFRESH_JS = """
function wireRefresh(btn, target) {
  btn.href = target.href;
  btn.textContent = target.label;
  btn.title = target.direct ? 'Fetches the latest RT intervals, recomputes spreads and P&L, republishes in ~3 min' : 'Opens GitHub Actions';
  btn.onclick = target.direct ? (e) => { e.preventDefault(); fireRefresh(btn, target); } : null;
}
function fireRefresh(btn, target) {
  btn.textContent = 'Requesting...';
  fetch(target.href, {mode: 'no-cors', cache: 'no-store'})
    .then(() => { btn.textContent = 'Refresh requested, site updates in ~3 min'; })
    .catch(() => { window.open(target.href, '_blank'); btn.textContent = target.label; });
  setTimeout(() => { btn.textContent = target.label; }, 90000);
}
"""
