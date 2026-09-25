"""Password gate shared by index.html, simulator.html and portfolio.html.

Two locks: 'site' covers every page, and 'portfolio' sits on top of it for portfolio.html.
Only SHA-256 hashes live here, never the passwords. It's a casual lock, not real security:
the pages are static, so the data is still in the HTML source for anyone who opens it.
To change a password:
    python -c "import hashlib;print(hashlib.sha256(b'new-password').hexdigest())"
Changing a hash locks everyone out of that lock again (the unlock is remembered per browser
by hash). No pandas here on purpose, same as refresh.py."""
import json

from theme import COLORS

SITE_LOCK = {'key': 'site', 'title': 'LRG Dashboard',
             'hash': '68b79ba7c1c56c1875d5bef864987aec43cf959fec59912cc13d2d00e64e943d'}
PORTFOLIO_LOCK = {'key': 'portfolio', 'title': 'Portfolio',
                  'hash': '30f92e940574066f7df0f8a4d362c68d1723adf3e348af462664ac1032c5d1d0'}

# An opaque overlay on top of the page rather than display:none on the content, so the
# Plotly charts still size against the real width underneath.
_GATE_CSS = f"""
  #lock {{ position:fixed; inset:0; z-index:1000; background:{COLORS['ring']};
    display:flex; align-items:center; justify-content:center;
    font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  #lock form {{ background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:24px;
    display:flex; flex-direction:column; gap:12px; width:260px; }}
  #lock .lock-title {{ color:#eee; font-size:16px; font-weight:600; }}
  #lock input {{ background:#111; color:#eee; border:1px solid #444; border-radius:4px; padding:8px 10px; font-size:14px; }}
  #lock button {{ background:{COLORS['dam']}; color:#fff; border:0; border-radius:4px; padding:8px; font-size:13px; cursor:pointer; }}
  #lock .lock-error {{ color:{COLORS['negative']}; font-size:12px; min-height:14px; }}
  #lock a {{ color:#aaa; font-size:13px; text-decoration:none; }}
"""


def gate_html(locks, back_link=False):
    """<style> + overlay + script, placed right after <body>. locks are asked for in order,
    skipping any this browser already unlocked; the overlay goes away once all are open."""
    back = '<a href="index.html">&larr; Back to dashboard</a>' if back_link else ''
    return f"""<style>{_GATE_CSS}</style>
<div id="lock">
  <form onsubmit="gateSubmit(event)">
    <div id="lock-title" class="lock-title"></div>
    <input id="lock-pw" type="password" placeholder="Password" autocomplete="current-password">
    <button type="submit">Enter</button>
    <div id="lock-error" class="lock-error"></div>
    {back}
  </form>
</div>
<script>
const GATE_LOCKS = {json.dumps(locks)};
function gateStored(lock) {{
    try {{ return localStorage.getItem('lrg-unlocked-' + lock.key) === lock.hash; }} catch (err) {{ return false; }}
}}
function gateNext() {{
    const lock = GATE_LOCKS.find(l => !gateStored(l));
    if (!lock) {{ document.getElementById('lock').remove(); return; }}
    document.getElementById('lock-title').textContent = lock.title;
    const pw = document.getElementById('lock-pw');
    pw.value = '';
    pw.focus();
}}
async function gateSubmit(e) {{
    e.preventDefault();
    const lock = GATE_LOCKS.find(l => !gateStored(l));
    const bytes = new TextEncoder().encode(document.getElementById('lock-pw').value);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hex = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
    if (hex !== lock.hash) {{
        document.getElementById('lock-error').textContent = 'Wrong password';
        return;
    }}
    document.getElementById('lock-error').textContent = '';
    try {{ localStorage.setItem('lrg-unlocked-' + lock.key, lock.hash); }} catch (err) {{
        // No storage (private mode etc.): still let this page view through.
        GATE_LOCKS.splice(GATE_LOCKS.indexOf(lock), 1);
    }}
    gateNext();
}}
gateNext();
</script>
"""
