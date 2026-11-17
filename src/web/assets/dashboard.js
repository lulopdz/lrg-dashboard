// Every chart of a tab that isn't open is drawn the first time the tab is: the figure scripts
// call lazyPlot instead of Plotly.newPlot (swapped in at the end of generar_web.py). Drawing
// all ~30 charts at load, hidden tabs included, was most of the page's startup time.
const PENDING_PLOTS = {};
// A chart in a predict tab's hidden zone panel waits the same way: drawn into display:none it
// would come out at Plotly's default width.
function inHiddenPanel(el) {
  const panel = el.closest('.zone-panel');
  return !!panel && !panel.classList.contains('active');
}
function lazyPlot(id, data, layout, config) {
  const el = document.getElementById(id);
  const tab = el && el.closest('.tab-content');
  if (!tab || (tab.classList.contains('active') && !inHiddenPanel(el))) return Plotly['newPlot'](id, data, layout, config);  // not Plotly.newPlot: see the replace at the end
  PENDING_PLOTS[id] = [data, layout, config];
}
function drawPendingPlots(tabEl) {
  const ids = Object.keys(PENDING_PLOTS).filter(id => { const el = document.getElementById(id); return tabEl.contains(el) && !inHiddenPanel(el); });
  ids.forEach(id => { const [d, l, c] = PENDING_PLOTS[id]; delete PENDING_PLOTS[id]; Plotly['newPlot'](id, d, l, c); });
  return ids;
}
function isDrawn(divId) {
  const gd = document.getElementById(divId);
  return !!(gd && gd.data);
}

// The shown zone's archive and live target day for a predict tab (null/undefined elsewhere).
function zoneHistory(tab) {
  if (tab === 'spread-signal') return SIGNAL_HISTORY[forecastZone] || {};
  const prefix = FORECAST_TABS[tab];
  return prefix ? ((FORECAST_HISTORY[prefix] || {})[forecastZone] || {}) : null;
}
function zoneLatest(tab) {
  if (tab === 'spread-signal') return (SIGNAL_BASE[forecastZone] || {}).latest;
  return (FORECAST_LATEST[FORECAST_TABS[tab]] || {})[forecastZone];
}
// Let the Day bar reach the archive's oldest day (forecast archives outlast the 30-day window).
function setDateBounds(hist) {
  const dateInput = document.getElementById('global-date');
  const days = hist ? Object.keys(hist).sort() : [];
  dateInput.min = days.length ? days[0] : DAY_RANGE[0];
  dateInput.max = days.length && days[days.length - 1] > DAY_RANGE[1] ? days[days.length - 1] : DAY_RANGE[1];
}
// Show a predict tab's panel for forecastZone; returns the ids of the charts drawn just now.
function showZonePanel(tab) {
  const tabEl = document.getElementById('tab-' + tab);
  if (!tabEl) return [];
  let shown = null;
  tabEl.querySelectorAll('.zone-panel').forEach(p => {
    const on = p.dataset.zone === forecastZone;
    p.classList.toggle('active', on);
    if (on) shown = p;
  });
  return shown && tabEl.classList.contains('active') ? drawPendingPlots(shown) : [];
}

function showTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const tabEl = document.getElementById('tab-' + name);
  tabEl.classList.add('active');
  btn.classList.add('active');
  currentTab = name;
  if (tabEl.querySelector('.zone-panel')) showZonePanel(name);
  const drawn = drawPendingPlots(tabEl);

  const dateInput = document.getElementById('global-date');
  const startDate = zoneLatest(name) || TAB_DATES[name];
  if (startDate && dateInput) dateInput.value = startDate;
  if (dateInput) setDateBounds(zoneHistory(name));
  if (FORECAST_TABS[name]) applyForecastDate(name);
  if (name === 'spread-signal') applySignalDate();

  const refreshBtn = document.getElementById('tab-refresh-btn');
  const r = TAB_REFRESH[name];
  if (r) {
    wireRefresh(refreshBtn, r);
    refreshBtn.style.display = 'inline-block';
  } else {
    refreshBtn.style.display = 'none';
  }

  const zoneGroup = document.getElementById('tab-zone-group');
  const z = TAB_ZONES[name];
  if (z) {
    document.getElementById('tab-zone-label').textContent = z.label + ':';
    const sel = document.getElementById('tab-zone-select');
    sel.innerHTML = z.options.map(o =>
      '<option value="' + o + '"' + (o === (z.panels ? forecastZone : z.default) ? ' selected' : '') + '>' + o + '</option>'
    ).join('');
    zoneGroup.style.display = '';
  } else {
    zoneGroup.style.display = 'none';
  }

  updateSupplyTiles();
  renderWeatherContext();
  // Charts drawn just now start on their baked default; bring them to the Day bar and zone.
  drawn.forEach(id => { if (FIG_CONFIGS[id]) applyFigSelection(id); });
  if (drawn.length) applyZoneChange();
}

const FIG_CONFIGS = {};
const TABLE_CONFIGS = {};

function registerFig(divId, zonesList, datesList, tracesPerCombo, titlePrefix, yAxisTitles, dateSelId, showTitle, zoneSelId, compactZoneData, defaultZoneIdx, compactKind) {
  FIG_CONFIGS[divId] = {
    zones: zonesList, dates: datesList, tracesPerCombo: tracesPerCombo, titlePrefix: titlePrefix,
    yAxisTitles: yAxisTitles || null, dateSelId: dateSelId || 'global-date', showTitle: showTitle !== false,
    zoneSelId: zoneSelId || (divId + '-zone'),
    compactZoneData: compactZoneData || null, defaultZoneIdx: defaultZoneIdx != null ? defaultZoneIdx : 0,
    compactKind: compactKind || 'reference'
  };
}

function registerTable(divId, zonesList) {
  TABLE_CONFIGS[divId] = zonesList;
}

function addDays(dateStr, n) {
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

function zoneReferenceSeries(zoneData, date) {
  // Mirrors dashboard_figures.py's _reference_series in JS: the selected day, the day before
  // it, and the trailing 7-day average leading up to it, computed from a compact
  // {date: [24 hourly values]} map instead of a pre-baked trace.
  const prevDate = addDays(date, -1);
  const hours = Array.from({length: 24}, (_, i) => i + 1);
  const dayVals = zoneData[date] || new Array(24).fill(null);
  const prevVals = zoneData[prevDate] || new Array(24).fill(null);
  const sum = new Array(24).fill(0), count = new Array(24).fill(0);
  for (let i = 1; i <= 7; i++) {
    const row = zoneData[addDays(date, -i)];
    if (!row) continue;
    for (let h = 0; h < 24; h++) { if (row[h] != null) { sum[h] += row[h]; count[h]++; } }
  }
  const avgVals = sum.map((s, h) => count[h] ? s / count[h] : null);
  return {hours, dayVals, prevVals, avgVals, prevDate};
}

// Charts compacted by compact_combos keep one trace group; COMBO_DATA[divId][i] holds the
// [x, y, name(, bar colors)] of each trace for combination i (zone-major, then day). x is 0
// when it is just hours 1..n.
function showCombo(divId, comboIdx, offset) {
  const c = (COMBO_DATA[divId] || [])[comboIdx];
  if (!c) return;
  const idx = c.map((_, i) => offset + i);
  const xOf = t => t[0] === 0 ? t[1].map((_, i) => i + 1) : t[0];
  Plotly.restyle(divId, {x: c.map(xOf), y: c.map(t => t[1]), name: c.map(t => t[2]), visible: c.map(() => true)}, idx);
  const colored = c.map((t, i) => t[3] ? i : -1).filter(i => i >= 0);
  if (colored.length) Plotly.restyle(divId, {'marker.color': colored.map(i => c[i][3])}, colored.map(i => offset + i));
}

// Traces ahead of the 3 dynamic ones: one group once compacted, else one group per day.
function bakedCount(divId, cfg) {
  return COMBO_DATA[divId] ? cfg.tracesPerCombo : cfg.dates.length * cfg.tracesPerCombo;
}

function showCompactZone(divId, cfg, zoneLabel, date) {
  // Recomputes the 3 'dynamic' placeholder traces build_hourly_fig(compact_zones=True) added
  // after the baked ones, and hides every baked trace so only the freshly-computed zone shows.
  const zoneData = COMPACT_ZONE_DATA[cfg.compactZoneData] && COMPACT_ZONE_DATA[cfg.compactZoneData][zoneLabel];
  if (!zoneData) return;
  const bakedTotal = bakedCount(divId, cfg);
  const dynamicStart = bakedTotal;
  const {hours, dayVals, prevVals, avgVals, prevDate} = zoneReferenceSeries(zoneData, date);
  Plotly.restyle(divId, {visible: new Array(bakedTotal).fill(false)},
                  Array.from({length: bakedTotal}, (_, i) => i));
  Plotly.restyle(divId, {
    x: [hours, hours, hours],
    y: [avgVals, prevVals, dayVals],
    name: ['7d Average', prevDate, date],
    visible: [true, true, true]
  }, [dynamicStart, dynamicStart + 1, dynamicStart + 2]);
}

function showCompactSpreadZone(divId, cfg, zoneLabel, date) {
  // Spread's dynamic placeholders are a different shape than the reference-series ones
  // (DAM line, RTM line, spread bar -- not avg/prev/day of one series), and Spread doesn't
  // carry its own compact data: it reads the DAM/RTM compact data already embedded for
  // those tabs (see build_spread_detail_fig's compact_zones docstring) and derives the
  // spread client-side instead of shipping a third copy of the same numbers.
  const damData = COMPACT_ZONE_DATA['dam'] && COMPACT_ZONE_DATA['dam'][zoneLabel];
  const rtmData = COMPACT_ZONE_DATA['rtm'] && COMPACT_ZONE_DATA['rtm'][zoneLabel];
  if (!damData || !rtmData) return;
  const bakedTotal = bakedCount(divId, cfg);
  const dynamicStart = bakedTotal;
  const hours = Array.from({length: 24}, (_, i) => i + 1);
  const damVals = damData[date] || new Array(24).fill(null);
  const rtmVals = rtmData[date] || new Array(24).fill(null);
  const spreadVals = hours.map((_, i) => (damVals[i] == null || rtmVals[i] == null) ? null : damVals[i] - rtmVals[i]);
  const barColors = spreadVals.map(v => v == null ? '/*__COLOR_MUTED__*/' : (v >= 0 ? '/*__COLOR_POSITIVE__*/' : '/*__COLOR_NEGATIVE__*/'));
  Plotly.restyle(divId, {visible: new Array(bakedTotal).fill(false)},
                  Array.from({length: bakedTotal}, (_, i) => i));
  Plotly.restyle(divId, {
    x: [hours, hours, hours], y: [damVals, rtmVals, spreadVals], visible: [true, true, true]
  }, [dynamicStart, dynamicStart + 1, dynamicStart + 2]);
  Plotly.restyle(divId, {'marker.color': [barColors]}, [dynamicStart + 2]);
}

function showBakedZone(divId, cfg, dateIdx) {
  // Restores the default zone's own pre-baked trace for the selected day, and hides the 3
  // dynamic placeholder traces in case a non-default zone had them showing.
  if (COMBO_DATA[divId]) {
    showCombo(divId, dateIdx, 0);
    const n = cfg.tracesPerCombo;
    Plotly.restyle(divId, {visible: [false, false, false]}, [n, n + 1, n + 2]);
    return;
  }
  const bakedTotal = cfg.dates.length * cfg.tracesPerCombo;
  const visible = new Array(bakedTotal).fill(false);
  const base = dateIdx * cfg.tracesPerCombo;
  for (let k = 0; k < cfg.tracesPerCombo; k++) visible[base + k] = true;
  Plotly.restyle(divId, {visible: visible}, Array.from({length: bakedTotal}, (_, i) => i));
  Plotly.restyle(divId, {visible: [false, false, false]}, [bakedTotal, bakedTotal + 1, bakedTotal + 2]);
}

function applyFigSelection(divId) {
  if (!isDrawn(divId)) return;  // not drawn yet: showTab applies the selection when it is
  const cfg = FIG_CONFIGS[divId];
  const dateSel = document.getElementById(cfg.dateSelId);
  const dateIdx = cfg.dates.indexOf(dateSel.value);
  // Small-multiple tiles (e.g. the Weather Forecast grid) have exactly one 'zone' -- their
  // own div, registered as a single-item list -- and no per-tile selector to read; only
  // figures with an actual dropdown (zones.length > 1) look one up.
  let zoneIdx = 0;
  let zoneLabel = cfg.zones[0];
  if (cfg.zones.length > 1) {
    const zoneSel = document.getElementById(cfg.zoneSelId);
    if (!zoneSel) return;
    zoneIdx = cfg.zones.indexOf(zoneSel.value);
    zoneLabel = zoneSel.value;
  }
  if (zoneIdx === -1 || dateIdx === -1) return;

  if (cfg.compactZoneData) {
    if (zoneIdx === cfg.defaultZoneIdx) {
      showBakedZone(divId, cfg, dateIdx);
    } else if (cfg.compactKind === 'spread') {
      showCompactSpreadZone(divId, cfg, zoneLabel, dateSel.value);
    } else {
      showCompactZone(divId, cfg, zoneLabel, dateSel.value);
    }
    return; // DAM/RTM/Spread titles are already hidden (show_title=False) and have no per-zone y-axis unit
  }

  if (COMBO_DATA[divId]) {
    showCombo(divId, zoneIdx * cfg.dates.length + dateIdx, 0);
  } else {
    const total = cfg.zones.length * cfg.dates.length * cfg.tracesPerCombo;
    const visible = new Array(total).fill(false);
    const base = (zoneIdx * cfg.dates.length + dateIdx) * cfg.tracesPerCombo;
    for (let k = 0; k < cfg.tracesPerCombo; k++) visible[base + k] = true;
    Plotly.restyle(divId, {visible: visible});
  }
  const relayout = {};
  if (cfg.showTitle) relayout.title = cfg.titlePrefix + ' - ' + zoneLabel + ' (' + dateSel.value + ')';
  if (cfg.yAxisTitles) relayout['yaxis.title'] = cfg.yAxisTitles[zoneIdx];
  Plotly.relayout(divId, relayout);
}

// The Supply Mix tiles are per-day like the charts beside them, so they have to be repainted
// whenever the Day picker moves -- otherwise they sit showing another day's numbers with
// nothing on screen saying so.
function updateSupplyTiles() {
  const day = SUPPLY_TILE_DATA[document.getElementById('global-date').value];
  SUPPLY_TILE_LABELS.forEach((_, i) => {
    const tile = document.getElementById('supply-tile-' + i);
    if (!tile) return;
    const entry = day ? day[i] : null;
    tile.style.display = entry ? '' : 'none';
    if (!entry) return;
    tile.querySelector('.stat-value').textContent = entry.value;
    tile.querySelector('.stat-sub').textContent = entry.sub || '';
  });
}

// Weather tab hour-by-hour table (tabs/weather.weather_context_json): HE1-24 across, one row per
// metric. Level rows are shaded along their own 90-day range, change/spread rows by size, and an
// unusual value (top 20%) is outlined. Neutral colours on purpose: it is context, not a verdict.
// The first row is the day's picks for the zone last chosen on the forecast tabs.
const WX_PALETTE = {temp: null, temp_tor: null, wind: [26, 188, 156], solar: [241, 196, 15]};
function wxColor(r, v) {
  const sc = WEATHER_CONTEXT.scale[r.key] || [0, 1];
  if (r.kind === 'level') {
    const t = Math.max(0, Math.min(1, (v - sc[0]) / ((sc[1] - sc[0]) || 1)));
    const base = WX_PALETTE[r.key];
    if (!base) {  // temperature: cold blue -> warm orange
      const c = [Math.round(60 + 170 * t), Math.round(120 + 20 * t), Math.round(210 - 160 * t)];
      return 'rgba(' + c.join(',') + ',' + (0.18 + 0.32 * Math.abs(t - 0.5) * 2).toFixed(2) + ')';
    }
    return 'rgba(' + base.join(',') + ',' + (0.06 + 0.40 * t).toFixed(2) + ')';
  }
  const t = Math.max(0, Math.min(1, Math.abs(v) / (sc[1] || 1)));
  return 'rgba(120,150,190,' + (0.04 + 0.42 * t).toFixed(2) + ')';
}
function wxFmt(r, v) {
  if (v == null) return '—';
  const d = r.key === 'solar' ? 0 : 1;
  const n = Math.abs(v).toFixed(d);
  if (r.kind !== 'change') return (v < 0 ? '−' : '') + n;
  return (Number(n) === 0 ? '' : v > 0 ? '+' : '−') + n;
}
function wxDay(r, vals) {
  const xs = vals.filter(v => v != null);
  if (!xs.length) return [null, ''];
  if (r.kind === 'level') return [xs.reduce((a, b) => a + b, 0) / xs.length, 'day average'];
  if (r.kind === 'change') {
    const m = xs.reduce((a, b) => Math.abs(b) > Math.abs(a) ? b : a, 0);
    return [m, 'largest of the day'];
  }
  return [Math.max(...xs), 'widest of the day'];
}
function renderWeatherContext() {
  const table = document.getElementById('wx-table');
  if (!table || !WEATHER_CONTEXT.rows) return;
  const date = document.getElementById('global-date').value;
  const day = WEATHER_CONTEXT.days[date];
  const title = document.getElementById('wx-title');
  if (title) title.textContent = 'Hour by hour \u00b7 ' + date;
  if (!day) {
    table.innerHTML = '<tbody><tr><td class="wx-empty">No weather data for ' + date + '</td></tr></tbody>';
    return;
  }
  const H = Array.from({length: 24}, (_, i) => i + 1);
  const picks = (((SIGNAL_HISTORY[forecastZone] || {})[date]) || {}).pick || [];
  const fc = ((((FORECAST_HISTORY.spread || {})[forecastZone]) || {})[date] || {}).predicted || [];
  let html = '<thead><tr><th class="wx-label">HE</th>' + H.map(h => '<th data-h="' + h + '">' + h + '</th>').join('')
    + '<th class="wx-day">Day</th></tr></thead><tbody>';
  html += '<tr class="wx-picks"><td class="wx-label">Picks <span class="site">' + forecastZone + '</span></td>'
    + H.map((h, i) => {
      const rank = picks[i] || 0, f = fc[i];
      if (!rank) return '<td data-h="' + h + '"></td>';
      const side = f > 0 ? 'gen' : 'load';
      return '<td data-h="' + h + '" title="HE' + h + ': pick \u2605' + rank + ', virtual ' + side + ', DART forecast $'
        + (f == null ? '\u2014' : f.toFixed(1)) + '"><span class="wx-star ' + (f > 0 ? 'pos' : 'neg') + '">\u2605' + rank + '</span></td>';
    }).join('')
    + '<td class="wx-day" title="Picks that day">' + picks.filter(Boolean).length + '</td></tr>';
  let group = null;
  WEATHER_CONTEXT.rows.forEach(r => {
    if (r.group !== group) {
      group = r.group;
      html += '<tr class="wx-group"><td colspan="26">' + group + '</td></tr>';
    }
    const vals = day[r.key] || [];
    const thr = WEATHER_CONTEXT.unusual[r.key];
    const cells = H.map((h, i) => {
      const v = vals[i];
      if (v == null) return '<td data-h="' + h + '" class="wx-na">\u2014</td>';
      const odd = thr != null && Math.abs(v) >= thr;
      const tip = 'HE' + h + ' \u00b7 ' + r.label + ' (' + r.site + '): ' + wxFmt(r, v) + ' ' + r.unit
        + (odd ? ' \u2014 unusual, top 20% of the last 90 days' : '');
      return '<td data-h="' + h + '" class="' + (odd ? 'wx-hi' : '') + '" style="background:' + wxColor(r, v)
        + '" title="' + tip + '">' + wxFmt(r, v) + '</td>';
    }).join('');
    const [dv, how] = wxDay(r, vals);
    const dOdd = dv != null && thr != null && Math.abs(dv) >= thr;
    html += '<tr><td class="wx-label" title="' + r.help + '">' + r.label + ' <span class="site">' + r.site + ' \u00b7 ' + r.unit
      + '</span></td>' + cells + '<td class="wx-day' + (dOdd ? ' wx-hi' : '') + '" title="' + how + '">' + wxFmt(r, dv) + '</td></tr>';
  });
  table.innerHTML = html + '</tbody>';
  if (!table.dataset.hover) {  // highlight the hovered hour's column, so a column reads as one hour
    table.dataset.hover = '1';
    table.addEventListener('mouseover', e => {
      const td = e.target.closest('[data-h]');
      table.querySelectorAll('.wx-col').forEach(c => c.classList.remove('wx-col'));
      if (td) table.querySelectorAll('[data-h="' + td.dataset.h + '"]').forEach(c => c.classList.add('wx-col'));
    });
    table.addEventListener('mouseleave', () => table.querySelectorAll('.wx-col').forEach(c => c.classList.remove('wx-col')));
  }
}

function applyAllFigs() {
  Object.keys(FIG_CONFIGS).forEach(applyFigSelection);
  updateSupplyTiles();
  renderWeatherContext();
  if (FORECAST_TABS[currentTab]) applyForecastDate(currentTab);
  if (currentTab === 'spread-signal') applySignalDate();
}


// Hour-by-hour table for the Spread Signal tab, rebuilt from the archive for whichever day
// the Day bar shows. Columns past the DART forecast only appear for a day that has cleared.
function renderSignalTable(date) {
  const day = zoneHistory('spread-signal')[date];
  const body = document.getElementById(fz('spread-signal-rows'));
  const table = body && body.closest('table');
  if (!day || !body) return;
  const fc = ((((FORECAST_HISTORY.spread || {})[forecastZone]) || {})[date] || {}).predicted || [];
  const act = day.actual;
  table.classList.toggle('live', !act);
  const sign = v => v > 0 ? 'pos' : (v < 0 ? 'neg' : '');
  const money = (v, d = 1) => {
    if (v == null) return '—';
    const r = Number(v.toFixed(d));  // sign of the rounded value, so -0.3 at 0 decimals reads $0, not $−0
    return '$' + (r < 0 ? '−' : r > 0 ? '+' : '') + Math.abs(r).toFixed(d);
  };
  // ★ = the day's picks (picks.py, archived per day), traded on the point forecast's sign. A past
  // day is scored on that same side for every hour; the picks in full, the rest dimmed.
  const rows = day.p_pos.map((p, i) => {
    const rank = (day.pick || [])[i] || 0;
    const f = fc[i];
    const side = f > 0 ? 'gen' : 'load';
    let past = '';
    if (act) {
      const a = act[i];
      const pnl = a == null || f == null || f === 0 ? null : (f > 0 ? a : -a);
      const result = a == null ? '<span class="tag tag-pending">pending</span>'
        : pnl == null ? '—'
        : '<span class="' + (rank ? 'tag ' + (pnl > 0 ? 'tag-right' : 'tag-wrong') : 'dim') + '">' + (pnl > 0 ? '✓ ' : '✗ ') + money(pnl, 0) + '</span>';
      past = '<td class="num ' + (a == null ? '' : sign(a)) + '">' + money(a) + '</td><td class="c">' + result + '</td>';
    }
    return '<tr class="' + (rank ? 'top' : '') + '">'
      + '<td>HE' + (i + 1) + '</td>'
      + '<td class="c stars" title="' + (rank ? '#' + rank + ' hour to trade, virtual ' + side : '') + '">'
      + (rank ? '★' + rank + ' <span class="' + (f > 0 ? 'pos' : 'neg') + '">' + side + '</span>' : '') + '</td>'
      + '<td class="num ' + (f == null ? '' : sign(f)) + '">' + money(f) + '</td>'
      + '<td class="num">' + (p == null ? '—' : p.toFixed(0) + '%') + '</td>' + past + '</tr>';
  });
  body.innerHTML = rows.join('');
  const title = document.getElementById(fz('spread-signal-table-title'));
  if (title) title.textContent = 'Hour by hour' + (act ? ' (' + date + ', scored)' : '');
}

// Spread Signal tab, driven by the Day bar: the bar chart (build_signal_bars: 0 bars, 1 no-call
// ticks, 2 outcome marks) and the hour-by-hour table are rebuilt from the archive.
const BAR_COLOR = /*__TIER_COLORS_JSON__*/;
function applySignalDate() {
  const divId = fz('spread-signal-fig');
  const gd = document.getElementById(divId);
  const note = document.getElementById(fz('spread-signal-note'));
  const title = document.getElementById(fz('spread-signal-title'));
  const hist = zoneHistory('spread-signal');
  const latest = zoneLatest('spread-signal');
  if (!gd || !gd.layout || !Object.keys(hist).length) return;
  const date = document.getElementById('global-date').value;
  const day = hist[date];
  if (!day) {
    const days = Object.keys(hist).sort();
    note.textContent = 'No signal archived for ' + date + ' (archive spans ' + days[0] + ' to ' + days[days.length - 1] + ')';
    return;
  }
  const H = Array.from({length: 24}, (_, i) => i + 1);
  const rated = day.tier.map(t => t !== 'low');
  const fc = ((FORECAST_HISTORY.spread || {})[forecastZone] || {})[date];
  const signed = H.map((h, i) => fc && fc.predicted[i] != null ? fc.predicted[i] : 0);
  const act = day.actual;
  // A call is right when sign(DART) matches it; 1 MW P&L is +DART on DART > 0, -DART on DART < 0.
  const scored = H.map((h, i) => (act && act[i] != null && rated[i]) ? {h, i, dart: act[i], pos: day.call[i] === 'DART > 0', hit: (act[i] > 0) === (day.call[i] === 'DART > 0'), tier: day.tier[i]} : null).filter(Boolean);
  Plotly.restyle(divId, {
    y: [signed], 'marker.color': [day.call.map((c, i) => BAR_COLOR[c + '|' + day.tier[i]] || BAR_COLOR.none)],
    text: [signed.map(v => (v >= 0 ? '+' : '') + v.toFixed(1))],
    hovertext: [H.map((h, i) => 'HE' + h + ': DART fc $' + (signed[i] >= 0 ? '+' : '') + signed[i].toFixed(1) + ' · ' + (rated[i] ? day.call[i] + ' · ' + day.tier[i] + ' · hit ' + (day.confidence[i] || 0).toFixed(0) + '% · edge $' + (day.edge[i] || 0).toFixed(1) + '/h' : 'no call'))],
  }, [0]);
  const top = Math.max(5, ...signed.map(Math.abs)) * 1.35;
  Plotly.restyle(divId, {x: [scored.map(s => s.h)], y: [scored.map(s => signed[s.i] >= 0 ? -top * 0.12 : top * 0.12)],
                         text: [scored.map(s => s.hit ? '✓' : '✗')], textfont: [{size: 14, color: scored.map(s => s.hit ? '#eee' : '#999')}]}, [1]);
  Plotly.relayout(divId, {'yaxis.range': [-top, top]});
  if (fc) restyleForecastCurve(fz('spread-signal-curve'), fc, date === latest);
  title.textContent = 'Spread Signal (DART = DA − RT) - ' + forecastZone + ' (' + date + ')';
  renderSignalTable(date);
  // The day's picks (picks.py): outlined bars and one card each, scored once the day clears.
  const pick = day.pick || new Array(24).fill(0);
  Plotly.restyle(divId, {'marker.line.width': [pick.map(r => r ? 3 : 0)], 'marker.line.color': '#f1c40f'}, [0]);
  const picked = H.map((h, i) => ({h, i, r: pick[i]})).filter(p => p.r && fc && fc.predicted[p.i] != null).sort((a, b) => a.r - b.r);
  const box = document.getElementById(fz('spread-signal-picks'));
  const pickPnl = picked.map(p => (act && act[p.i] != null) ? (fc.predicted[p.i] > 0 ? act[p.i] : -act[p.i]) : null);
  if (box) box.innerHTML = picked.length ? picked.map((p, k) => {
    const v = fc.predicted[p.i], gen = v > 0, res = pickPnl[k];
    const sub = res == null ? 'DART fc ' + (v >= 0 ? '+' : '−') + '$' + Math.abs(v).toFixed(1) + ' · P(DART > 0) ' + day.p_pos[p.i].toFixed(0) + '%'
      : 'fc ' + (v >= 0 ? '+' : '−') + '$' + Math.abs(v).toFixed(1) + ' · actual ' + (act[p.i] >= 0 ? '+' : '−') + '$' + Math.abs(act[p.i]).toFixed(1)
        + ' · <span class="' + (res > 0 ? 'pos' : 'neg') + '">' + (res > 0 ? '✓ ' : '✗ ') + (res >= 0 ? '+' : '−') + '$' + Math.abs(res).toFixed(0) + '</span>';
    return '<div class="pick ' + (gen ? 'gen' : 'load') + '"><div class="pick-head">★' + p.r + ' · HE' + p.h + '</div>'
      + '<div class="pick-side ' + (gen ? 'pos' : 'neg') + '">Virtual ' + (gen ? 'gen' : 'load') + '</div><div class="pick-sub">' + sub + '</div></div>';
  }).join('') : '<div class="none">No top hours this day: no hour where the forecast is $5 or more and the classifier agrees on its side.</div>';
  let text = (day.backfilled ? 'Walk-forward reconstruction (built ' + day.generated + ')' : 'Generated ' + day.generated);
  const cleared = act ? act.filter(v => v != null).length : 0;
  if (cleared) {
    const pnl = s => s.reduce((a, x) => a + (x.pos ? x.dart : -x.dart), 0);
    const high = scored.filter(s => s.tier === 'high');
    text += ' · ' + cleared + ' hours cleared, ' + scored.length + ' with a call: ' + scored.filter(s => s.hit).length + '/' + scored.length + ' right, 1 MW P&L $' + pnl(scored).toFixed(0);
    if (high.length) text += ' ($' + pnl(high).toFixed(0) + ' on ' + high.length + ' high, ' + high.filter(s => s.hit).length + ' right)';
    const done = pickPnl.filter(v => v != null);
    if (done.length) text += ' · top ' + done.length + ': ' + done.filter(v => v > 0).length + '/' + done.length + ' right, $' + done.reduce((a, v) => a + v, 0).toFixed(0);
  } else if (date === latest) {
    text += ' · pick a past day in the Day bar to see that signal scored against what cleared';
  }
  note.textContent = text;
}


// Swap a forecast tab's chart to the archived forecast for the Day bar's date. Trace order
// is fixed by build_forecast_fig: Similar #1, Similar #2, band low, band high, Predicted,
// Actual. The stat tiles and "Why this day?" always describe the live forecast.

// Restyle a forecast curve (build_forecast_fig's fixed trace order: Similar #1, Similar #2,
// band low, band high, Predicted, Actual) to an archived day. Shared by the DAM/RTM forecast
// tabs and the Spread Signal's curve.
function restyleForecastCurve(divId, day, isLive) {
  const gd = document.getElementById(divId);
  if (!gd || !gd.layout) return;
  if (!forecastMarks[divId]) forecastMarks[divId] = {shapes: gd.layout.shapes || [], annotations: gd.layout.annotations || []};
  const actual = day.actual || new Array(24).fill(null);
  const lo = day.band_lo, hi = day.band_hi;
  Plotly.restyle(divId, {
    y: [day.analog, day.analog_2, lo, hi, day.predicted, actual],
    name: ['Similar #1 (' + day.analog_date + ')', 'Similar #2 (' + day.analog_date_2 + ')', '', day.band_name, 'Predicted', 'Actual'],
    showlegend: [true, day.analog_2.some(v => v != null), false, true, true, !!day.actual],
  });
  // Past days keep the zero line (add_hline: xref 'x domain') but drop the "most confident
  // hour" vline (add_vline: yref 'y domain') and its label: that stat belongs to the live model.
  const marks = forecastMarks[divId];
  Plotly.relayout(divId, isLive ? marks : {shapes: marks.shapes.filter(s => s.yref !== 'y domain'), annotations: []});
}

function applyForecastDate(tab) {
  const hist = zoneHistory(tab);
  const divId = fz(tab + '-forecast');
  const gd = document.getElementById(divId);
  const note = document.getElementById(fz(tab + '-note'));
  const title = document.getElementById(fz(tab + '-title'));
  if (!hist || !Object.keys(hist).length || !gd || !gd.layout) return;
  const date = document.getElementById('global-date').value;
  const day = hist[date];
  const label = title.textContent.split(' Forecast')[0];
  if (!day) {
    const days = Object.keys(hist).sort();
    note.textContent = 'No forecast archived for ' + date + ' (archive spans ' + days[0] + ' to ' + days[days.length - 1] + ')';
    return;
  }
  const isLive = date === zoneLatest(tab);
  restyleForecastCurve(divId, day, isLive);
  const actual = day.actual || new Array(24).fill(null);
  title.textContent = label + ' Forecast - ' + forecastZone + ' (' + date + ')';
  const pairs = day.predicted.map((p, i) => [p, actual[i]]).filter(([p, a]) => p != null && a != null);
  let text = (day.backfilled ? 'Walk-forward reconstruction (built ' + day.generated + ')' : 'Generated ' + day.generated);
  if (pairs.length) {
    const mae = pairs.reduce((s, [p, a]) => s + Math.abs(p - a), 0) / pairs.length;
    text += ' · realized MAE $' + mae.toFixed(1) + ' over ' + pairs.length + ' cleared hours';
  }
  note.textContent = text + (isLive ? ' · pick a past day in the Day bar to see that forecast against what actually cleared' : '');
}

function applyZoneChange() {
  const z = TAB_ZONES[currentTab];
  if (!z) return;
  if (z.panels) {
    forecastZone = document.getElementById('tab-zone-select').value;
    showZonePanel(currentTab);
    // Stay on the Day bar's day if this zone has it archived, else jump to its live forecast.
    const hist = zoneHistory(currentTab), dateInput = document.getElementById('global-date');
    setDateBounds(hist);
    if (!hist[dateInput.value]) dateInput.value = zoneLatest(currentTab) || dateInput.value;
    if (FORECAST_TABS[currentTab]) applyForecastDate(currentTab);
    if (currentTab === 'spread-signal') applySignalDate();
    return;
  }
  if (z.hourlyDiv) applyFigSelection(z.hourlyDiv);
  if (z.tableDiv && TABLE_CONFIGS[z.tableDiv] && isDrawn(z.tableDiv)) {
    const sel = document.getElementById('tab-zone-select');
    const zones = TABLE_CONFIGS[z.tableDiv];
    const idx = zones.indexOf(sel.value);
    if (idx === -1) return;
    Plotly.restyle(z.tableDiv, {visible: zones.map((_, i) => i === idx)});
  }
}

function copyTableTSV(btn, plotlyDivId) {
  // Reads straight off the already-rendered heatmap trace (z/x/y) instead of a second copy
  // of the same numbers embedded separately -- the table figures are built with plain-list
  // z/x/y (see build_table_fig) specifically so this works without a typed-array decode step.
  const gd = document.getElementById(plotlyDivId);
  if (!gd || !gd.data) return;
  const trace = gd.data.find(t => t.visible === true || t.visible === undefined);
  if (!trace) return;
  const sep = '\t';
  const header = 'Date' + sep + trace.x.map(h => 'H' + h).join(sep);
  const rows = trace.y.map((date, i) =>
    date + sep + trace.z[i].map(v => v == null ? '' : v).join(sep)
  );
  navigator.clipboard.writeText([header, ...rows].join('\n'))
    .then(() => { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1800); })
    .catch(() => { btn.textContent = 'Failed';  setTimeout(() => btn.textContent = 'Copy', 1800); });
}

// This script block runs before the body, so the tiles it fills don't exist yet -- they start
// as empty shells and get their first paint once the DOM is up.
document.addEventListener('DOMContentLoaded', () => { updateSupplyTiles(); renderWeatherContext(); if (SIGNAL_BASE[forecastZone]) renderSignalTable(SIGNAL_BASE[forecastZone].latest); });
