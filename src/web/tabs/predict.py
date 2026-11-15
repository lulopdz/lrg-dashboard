"""Predict tabs (DAM Forecast, RTM Forecast, Spread Signal) HTML and metadata builders."""

import json
import os

import pandas as pd

from dashboard_data import (
    DEFAULT_ZONE,
    FORECAST_ZONES,
    forecast_history,
    signal_history,
    zone_suffix,
)
from dashboard_figures import (
    build_analog_comparison_fig,
    build_forecast_fig,
    build_signal_bars,
)
from tabs.common import hour_ranges, missing_cols_label, vintage_label, zid, zone_panels
from theme import TIER_COLORS

TOP_LABEL = 'top 5 (fc sign)'  # the day's picks (picks.py) in the archive record table


def build_signal_tab(tab_id='spread-signal'):
    """The Spread Signal tab, one panel per forecast zone."""
    return zone_panels({z: build_signal_panel(z, tab_id) for z in FORECAST_ZONES}, tab_id, 'Spread Signal')


def build_signal_panel(zone, tab_id):
    """One zone's Spread Signal, all in DART = DA - RT: KPI tiles, one bar per hour, and the
    hour-by-hour table front and center (predict_spread.py's classifiers)."""
    sfx = zone_suffix(zone)
    csv_path, meta_path = f'data/spread_signal{sfx}.csv', f'data/spread_signal_meta{sfx}.json'
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return ''
    sig = pd.read_csv(csv_path)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    forecast, fmeta = None, None
    if os.path.exists(f'data/spread_forecast{sfx}.csv') and os.path.exists(f'data/spread_forecast_meta{sfx}.json'):
        forecast = pd.read_csv(f'data/spread_forecast{sfx}.csv')
        with open(f'data/spread_forecast_meta{sfx}.json', encoding='utf-8') as f:
            fmeta = json.load(f)
    fig = build_signal_bars(sig, forecast)
    curve_html = ''
    if forecast is not None:
        # Since 2026-09-23 the point forecast is the difference of the two price forecasts
        # (predict_spread.derived_forecast); older metas are the DART booster's.
        method = fmeta.get('method')
        how = (f"{method}, the same run's forecasts, with the RT model's P10-P90 carried over"
               if method else "a model trained on DART itself")
        curve = build_forecast_fig(forecast, fmeta, series_label='Spread')
        curve_html = f"""
<h3>DART forecast vs. the two most similar days</h3>
{curve.to_html(full_html=False, include_plotlyjs=False, div_id=zid(f'{tab_id}-curve', zone))}
<p class="caveat">The point forecast of DART is {how}. On the archive it beat a model trained on DART directly:
the sign right 62% of hours against 56%, and 1 MW following that sign made $8.4k against $5.7k over 68 days. The
two similar days are the RT model's (closest on forecast load, wind and weather), shown with the DART that cleared
on them. On a past day the actual DART is overlaid.</p>"""
    bt = meta.get('backtest') or {}

    generated = pd.Timestamp(meta['generated_at']).tz_convert('-05:00').strftime('%Y-%m-%d %H:%M EST')
    missing = meta.get('missing_input_hours') or 0
    note = f'Generated {generated}{vintage_label(meta)}'
    if missing:
        note += f' · {missing} of 24 target hours had incomplete inputs{missing_cols_label(meta)}'
    note += ' · pick a past day in the Day bar to see that signal scored against what cleared'

    high = sig[sig['tier'] == 'high']
    medium = sig[sig['tier'] == 'medium']

    def sign_cls(call):
        return 'pos' if call == 'DART > 0' else 'neg'

    def calls_html(df):
        return ' · '.join(
            f"{hour_ranges(g['hour'])} <span class=\"{sign_cls(call)}\">{call}</span>" for call, g in df.groupby('call')) or 'none'

    def money(v):
        return f"${v:,.0f}" if v is not None else 'n/a'

    def pct(v):
        return f"{v:.0f}%" if v is not None else 'n/a'

    def pnl_cls(v):
        return 'pos' if (v or 0) > 0 else 'neg'

    NO_CALL = '<span class="rel">no call</span>'

    def bin_row(b):
        rated = b['tier'] != 'low'
        call_html = f"<span class=\"{sign_cls(b['call'])}\">{b['call']}</span>" if rated else '<span class="rel">—</span>'
        return (f"<tr class=\"{'model-row' if b['tier'] == 'high' else ''}\"><td>{b['lo']:.0%} – {b['hi']:.0%}</td><td>{b['n']}</td>"
                f"<td class=\"{pnl_cls(b.get('e_dart'))}\">${b.get('e_dart', 0):+.1f} ± {b.get('se', 0):.1f}</td>"
                f"<td>{call_html}</td><td>{pct(b['hit_rate'])}</td>"
                f"<td>{b['tier'] if rated else NO_CALL}</td></tr>")

    bin_rows = ''.join(bin_row(b) for b in bt.get('bins', []))

    # Walk-forward record: every archived day scored on what cleared (each day's signal was
    # produced with only earlier data, so this is out-of-sample end to end).
    # Live days (the signal as it was published) apart from walk-forward reconstructions, which
    # see the freshest version of every input rather than the 9:00 one and so flatter the signal.
    wf = {}
    wf_days = {'live': 0, 'reconstructed': 0}
    fc_days = forecast_history['spread'].get(zone, {})
    for d, day in signal_history.get(zone, {}).items():
        if not day['actual']:
            continue
        source = 'reconstructed' if day['backfilled'] else 'live'
        wf_days[source] += 1
        fc = (fc_days.get(d) or {}).get('predicted') or [None] * 24
        for i, dart in enumerate(day['actual']):
            if dart is None:
                continue
            wf.setdefault((source, day['tier'][i]), []).append(dart if day['call'][i] == 'DART > 0' else -dart)
            if day['pick'][i] and fc[i] is not None:
                wf.setdefault((source, TOP_LABEL), []).append(dart if fc[i] > 0 else -dart)
    wf_rows = ''.join(
        f"<tr class=\"{'model-row' if t == 'high' else ''}\"><td>{source}</td><td>{t}</td><td>{len(v)}</td>"
        f"<td>{sum(1 for x in v if x > 0) / len(v) * 100:.0f}%</td><td class=\"{pnl_cls(sum(v))}\">${sum(v) / len(v):.1f}/h</td>"
        f"<td class=\"{pnl_cls(sum(v))}\">${sum(v):,.0f}</td></tr>"
        for (source, t), v in sorted(wf.items(), key=lambda kv: (kv[0][0] != 'live', [TOP_LABEL, 'high', 'medium', 'low'].index(kv[0][1])))
        if v and t != 'low')  # no call in 'low'; the page scores the same way
    eval_days = bt.get('eval_days') or bt.get('days', '?')  # metas before the split scored the whole window

    return f"""
<h2 id="{zid(f'{tab_id}-title', zone)}">Spread Signal (DART = DA − RT) - {meta.get('zone')} ({meta.get('target_date')})</h2>
<p class="caveat" id="{zid(f'{tab_id}-note', zone)}">{note}</p>
<div class="picks-box" id="{zid(f'{tab_id}-picks', zone)}"></div>
{fig.to_html(full_html=False, include_plotlyjs=False, div_id=zid(f'{tab_id}-fig', zone))}
<p class="caveat"><span class="swatch" style="background:{TIER_COLORS[('DART > 0', 'high')]}"></span>DART &gt; 0 high
<span class="swatch" style="background:{TIER_COLORS[('DART > 0', 'medium')]}"></span>DART &gt; 0 medium
<span class="swatch" style="background:{TIER_COLORS[('DART < 0', 'high')]}"></span>DART &lt; 0 high
<span class="swatch" style="background:{TIER_COLORS[('DART < 0', 'medium')]}"></span>DART &lt; 0 medium
<span class="swatch" style="background:{TIER_COLORS['none']}"></span>no call</p>
<p class="caveat">One bar per hour, its height the DART point forecast ($/MWh) and its color the call:
<span class="pos">green</span> for DART &gt; 0 (DA above RT, a virtual gen pays), <span class="neg">red</span> for DART &lt; 0
(RT above DA, a virtual load pays); bright = high conviction, darker shade = medium, grey = no call. A red bar pointing up
means the point forecast is slightly positive while calls like this one have paid on the negative side.</p>
{curve_html}
<h3 id="{zid(f'{tab_id}-table-title', zone)}">Hour by hour</h3>
<table class="naive-table signal-table">
  <thead><tr><th>Hour</th><th class="c stars-col">★</th><th class="num">DART fc</th><th class="num">P(DART &gt; 0)</th>
    <th class="num past-only">Actual DART</th><th class="c past-only">Result</th></tr></thead>
  <tbody id="{zid(f'{tab_id}-rows', zone)}"></tbody>
</table>
<p class="caveat"><strong>★ Top 5</strong>: the day's five hours to trade, ranked ★1 to ★5 (with less collateral, trade
from ★1 down), each on the side of the DART forecast (<span class="pos">gen</span> above zero, <span class="neg">load</span>
below): of the hours where the P(DART &gt; 0) classifier agrees with that side, the five with the largest forecast, none
under $5 either way (so some days have fewer). On 106 days rebuilt walk-forward with the 9:00 inputs: $4.8–7.9/h at 1 MW,
right 64–72% of the time in all four zones; days before 2026-10-31 were archived with three. Live record in the archive
table below. <strong>Result</strong> (past days): 1 MW P&amp;L on the forecast's side, dimmed outside the picks.</p>
<details class="stats-block">
  <summary>Model record: last {eval_days} days out of sample and the archive</summary>
  <div class="stat-row">
    <div class="stat-tile highlight"><div class="stat-label">High-conviction hours</div><div class="stat-value">{len(high)}</div>
      <div class="stat-sub">{calls_html(high)}</div></div>
    <div class="stat-tile"><div class="stat-label">Medium</div><div class="stat-value">{len(medium)}</div>
      <div class="stat-sub">{calls_html(medium)}</div></div>
    <div class="stat-tile"><div class="stat-label">Calls right (last {eval_days}d, out of sample)</div>
      <div class="stat-value">{pct(bt.get('hit_rate_high'))} <span class="rel">high</span> · {pct(bt.get('hit_rate_rated'))} <span class="rel">high+medium</span></div>
      <div class="stat-sub">DART &gt; 0 happens {pct(bt.get('base_rate_pos'))} of hours; a DART &lt; 0 call can be right less often and still pay</div></div>
    <div class="stat-tile"><div class="stat-label">1 MW P&amp;L if followed (last {eval_days}d)</div>
      <div class="stat-value"><span class="{pnl_cls(bt.get('pnl_high'))}">{money(bt.get('pnl_high'))}</span> <span class="rel">high, {bt.get('n_high', 0)}h</span> · <span class="{pnl_cls(bt.get('pnl_rated'))}">{money(bt.get('pnl_rated'))}</span> <span class="rel">high+medium, {bt.get('n_rated', 0)}h</span></div>
      <div class="stat-sub">always gen {money(bt.get('always_gen_pnl'))} · always load {money(bt.get('always_load_pnl'))} · perfect {money(bt.get('optimal_pnl'))}</div></div>
  </div>
  <div class="summary-split">
    <div class="summary-half">
      <h3>How the tiers are earned (last {bt.get('days', '?')}d, {bt.get('bins_hours') or bt.get('n_hours', '?')}h)</h3>
      <table class="naive-table">
        <thead><tr><th>P(DART&gt;0) range</th><th>Hours</th><th>E[DART] $/h</th><th>Call</th><th>Call right</th><th>Tier</th></tr></thead>
        <tbody>{bin_rows}</tbody>
      </table>
      <p class="caveat">DART &gt; 0 is the normal state of this market (every hour of the day, 58–75% of the time), so "the
      likelier side" is almost always DART &gt; 0 and is not the question. The call for each probability range is the side
      that has <em>paid</em> there: the sign of the mean DART over the window (E[DART], ± its standard error); the edge is
      its size. High = edge ≥ $5/h and the call right ≥ 60%; medium = edge ≥ $1/h; an edge under 2 standard errors, or
      fewer than 24 hours, is noise: no call. P(DART &gt; 0) is the classifier's own number and runs overconfident, which
      is why a range is judged by what it paid rather than by the probability. The call/tier for each range is fit on the
      whole window; the tiles above fit it on the older half and score it on the newer {eval_days} days.</p>
    </div>
    <div class="summary-half">
      <h3>Archive record ({wf_days['live']} live days, {wf_days['reconstructed']} reconstructed, scored on what cleared)</h3>
      <table class="naive-table">
        <thead><tr><th>Days</th><th>Tier</th><th>Hours</th><th>Hit rate</th><th>Edge</th><th>1 MW P&amp;L</th></tr></thead>
        <tbody>{wf_rows}</tbody>
      </table>
      <p class="caveat">Each archived day's signal was produced with only earlier data. <em>Live</em> days are the
      signal as published at 9:00; <em>reconstructed</em> days were rebuilt walk-forward and saw the latest version of
      the load/wind/weather forecasts rather than the 9:00 one, so they flatter the signal.</p>
    </div>
  </div>
</details>
"""


def scorecard_html(prefix, zone):
    """The out-of-sample record from data/forecast_scorecard{_ZONE}.json (src/forecast/scorecard.py):
    every archived forecast scored against what cleared, per vintage. Unlike the backtest
    tiles above it, this is what the model actually did on the days it was live for."""
    path = f'data/forecast_scorecard{zone_suffix(zone)}.json'
    if not os.path.exists(path):
        return ''
    with open(path, encoding='utf-8') as f:
        card = json.load(f)
    rows = []
    for vintage, r in (card.get('prices') or {}).get(prefix, {}).items():
        label = 'pre-DAM (9:00 run)' if vintage.startswith('pre_dam') else 'post-DAM (afternoon run)'
        if vintage.endswith('_reconstructed'):
            label += ', walk-forward reconstruction'
        extra = ''
        if prefix == 'rtm' and r.get('baseline_rt_eq_dam_real_mae') is not None:
            extra = f" · RT = real DAM would score ${r['baseline_rt_eq_dam_real_mae']:.1f}"
        if prefix == 'spread' and r.get('sign_hit_rate') is not None:
            extra = f" · sign right {r['sign_hit_rate']:.0f}% (always DART&gt;0: {r['base_rate_pos']:.0f}%)"
        band = ''
        if r.get('band_p10_p90_coverage') is not None:
            # Bands only exist since 2026-09-22, so say over how many days (older scorecards lack it).
            over = f" over the {r['band_n_days']} days with a band" if r.get('band_n_days') else ''
            band = f" · P10-P90 covered {r['band_p10_p90_coverage']:.0f}% of hours{over}"
        rows.append(f"<li><b>{label}</b>: {r['n_days']} days archived, MAE ${r['mae']:.1f}, bias ${r['bias']:+.1f}{band}{extra}</li>")
    if not rows:
        return ''
    return ('<div class="caveat"><b>Out-of-sample record</b> (archived forecasts vs. what cleared; '
            'the honest number, updated daily):<ul>' + ''.join(rows) + '</ul></div>')


def build_forecast_tab(prefix, tab_id, series_label):
    """A forecast tab (DAM/RTM Forecast), one panel per forecast zone. Its refresh link lives in
    the shared day-bar (see TAB_REFRESH), not inside the tab."""
    return zone_panels({z: build_forecast_panel(prefix, tab_id, series_label, z) for z in FORECAST_ZONES},
                       tab_id, f'{series_label} Forecast')


def build_forecast_panel(prefix, tab_id, series_label, zone):
    """One zone's forecast (predicted curve + backtest stats + similar-day comparison), or ''
    when that zone's predict_*.py hasn't written its files yet."""
    csv_path = f'data/{prefix}_forecast{zone_suffix(zone)}.csv'
    meta_path = f'data/{prefix}_forecast_meta{zone_suffix(zone)}.json'
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return ''

    forecast = pd.read_csv(csv_path)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    fig = build_forecast_fig(forecast, meta, series_label=series_label)
    analog_fig = build_analog_comparison_fig(meta)

    backtest = meta.get('backtest') or {}
    model_mae = f"${backtest['model_mae']:.1f}" if backtest else 'n/a'
    naive_mae = f"${backtest['naive_mae']:.1f}" if backtest else 'n/a'
    backtest_days_label = round(backtest['n_test_hours'] / 24) if backtest else 'n/a'

    # Freshness line: a run that arrives late looks identical to a fresh one otherwise.
    generated = pd.Timestamp(meta['generated_at']).tz_convert('-05:00').strftime('%Y-%m-%d %H:%M EST')
    missing = meta.get('missing_input_hours') or 0
    note_id = zid(f'{tab_id}-note', zone)
    freshness_html = f'<p class="caveat" id="{note_id}">Generated {generated}{vintage_label(meta)}'
    if missing:
        freshness_html += f' · {missing} of 24 target hours had incomplete inputs{missing_cols_label(meta)}'
    freshness_html += ' · pick a past day in the Day bar to see that forecast against what actually cleared</p>'

    recommended = meta.get('recommended_hour') or {}
    confident_hour_label = (
        f"Hour {recommended['hour']} (±${recommended['expected_error']:.1f})" if recommended.get('hour') else 'n/a'
    )

    analog_section_html = f"""
<h3>Why this day? Tomorrow's forecast vs. the closest historical day{'s' if meta.get('analog_comparison_2') else ''}</h3>
{analog_fig.to_html(full_html=False, include_plotlyjs=False, div_id=zid(f'{tab_id}-analog', zone))}""" if analog_fig else ''

    # Directional track record (spread only -- DAM/RTM prices have no Long/Short to call):
    # the model's own backtest predictions scored exactly like the Trading Simulator scores a
    # human's calls (see forecast_common.directional_backtest). "Would I have trusted this
    # model?" is a more direct question than the $ MAE above answers on its own.
    track = meta.get('directional_backtest')
    track_section_html = ''
    if track:
        pnl_cls = 'pos' if track['total_pnl'] > 0 else ('neg' if track['total_pnl'] < 0 else '')
        # The naive rows are the honest bar: the spread is positive most hours, so always
        # taking one side wins a lot of them. Beating zero is not the achievement -- beating
        # these is. Which naive side wins flips between periods, so both are shown.
        beats = max(track['naive_gen_pct'] or 0, track['naive_load_pct'] or 0)
        verdict = ('beats both naive strategies' if (track['pct_of_optimal'] or 0) > beats
                   else 'does NOT beat picking one side and holding it')
        track_section_html = f"""
<div class="summary-split">
  <div class="summary-half">
    <h3>If you'd followed this model's own calls (last {track['n_hours']}h)</h3>
    <div class="stat-row">
      <div class="stat-tile"><div class="stat-label">Record</div><div class="stat-value">{track['correct']}-{track['n_hours'] - track['correct']}</div>
        <div class="stat-sub">{track['win_rate']:.0f}% win rate</div></div>
      <div class="stat-tile highlight"><div class="stat-label">P&amp;L vs. optimal</div>
        <div class="stat-value {pnl_cls}">${track['total_pnl']:,.0f} / ${track['optimal_pnl']:,.0f}</div>
        <div class="stat-sub">{track['pct_of_optimal']:.0f}% of optimal (perfect hindsight)</div></div>
    </div>
  </div>
  <div class="summary-half">
    <h3>Versus never changing your mind</h3>
    <table class="naive-table">
      <thead><tr><th>Strategy</th><th>Win rate</th><th>P&amp;L</th><th>% of optimal</th></tr></thead>
      <tbody>
        <tr class="model-row"><td>This model</td><td>{track['win_rate']:.0f}%</td>
          <td>${track['total_pnl']:,.0f}</td><td>{track['pct_of_optimal']:.0f}%</td></tr>
        <tr><td>Always Virtual Gen</td><td>{track['naive_gen_win_rate']:.0f}%</td>
          <td>${track['naive_gen_pnl']:,.0f}</td><td>{track['naive_gen_pct']:.0f}%</td></tr>
        <tr><td>Always Virtual Load</td><td>{track['naive_load_win_rate']:.0f}%</td>
          <td>${track['naive_load_pnl']:,.0f}</td><td>{track['naive_load_pct']:.0f}%</td></tr>
      </tbody>
    </table>
    <p class="caveat">Here the model {verdict}. Win rate misleads: the spread is positive most hours,
    so always going Virtual Gen wins more of them &mdash; but the losing hours are the big ones. Which
    naive side wins flips between periods, so one window is weak evidence.</p>
  </div>
</div>"""

    return f"""
<h2 id="{zid(f'{tab_id}-title', zone)}">{series_label} Forecast - {meta.get('zone')} ({meta.get('target_date')})</h2>
{freshness_html}
<div class="stat-row">
  <div class="stat-tile"><div class="stat-label">Model MAE ({backtest_days_label}d backtest)</div><div class="stat-value">{model_mae}</div></div>
  <div class="stat-tile"><div class="stat-label">Naive baseline MAE</div><div class="stat-value">{naive_mae}</div></div>
  <div class="stat-tile"><div class="stat-label">Most confident hour</div><div class="stat-value">{confident_hour_label}</div></div>
</div>
{scorecard_html(prefix, zone)}
{track_section_html}
{fig.to_html(full_html=False, include_plotlyjs=False, div_id=zid(f'{tab_id}-forecast', zone))}
{analog_section_html}
"""


def build_predict_tabs():
    """Build DAM Forecast, RTM Forecast, and Spread Signal tabs plus JS archive/target metadata."""
    dam_forecast_tab_button, dam_forecast_tab_html = build_forecast_tab('dam', 'forecast', 'DAM')
    rtm_forecast_tab_button, rtm_forecast_tab_html = build_forecast_tab('rtm', 'rtm-forecast', 'RTM')
    spread_forecast_tab_button, spread_forecast_tab_html = build_signal_tab()

    # Mark whichever predict tab appears first as the group's visual start (extra left margin,
    # see .tab-group-start) -- any of the three may be absent if its predict_*.py hasn't run yet.
    _predict_buttons = [dam_forecast_tab_button, rtm_forecast_tab_button, spread_forecast_tab_button]
    for _i, _btn in enumerate(_predict_buttons):
        if _btn:
            _predict_buttons[_i] = _btn.replace('class="tab-btn group-predict"', 'class="tab-btn group-predict tab-group-start"', 1)
            break
    dam_forecast_tab_button, rtm_forecast_tab_button, spread_forecast_tab_button = _predict_buttons

    # Forecast tabs open on the live forecast's own target day, and let the Day bar reach back
    # to the oldest archived forecast (older than the TABLE_DAYS window the other tabs use). Both
    # are per zone ({prefix: {zone: date}}): the page JS reads the shown zone's.
    forecast_tabs = {'forecast': 'dam', 'rtm-forecast': 'rtm'}
    forecast_latest = {}
    tab_dates = {}
    for _tab, _prefix in forecast_tabs.items():
        for _zone in FORECAST_ZONES:
            _meta_path = f'data/{_prefix}_forecast_meta{zone_suffix(_zone)}.json'
            if os.path.exists(_meta_path):
                with open(_meta_path, encoding='utf-8') as _f:
                    forecast_latest.setdefault(_prefix, {})[_zone] = json.load(_f)['target_date']
        if forecast_latest.get(_prefix):
            tab_dates[_tab] = next(iter(forecast_latest[_prefix].values()))

    # The signal's live target day and big-hour threshold, per zone.
    signal_base = {}
    for _zone in FORECAST_ZONES:
        _meta_path = f'data/spread_signal_meta{zone_suffix(_zone)}.json'
        if not os.path.exists(_meta_path):
            continue
        with open(_meta_path, encoding='utf-8') as _f:
            _m = json.load(_f)
        _t = _m['big_threshold']
        signal_base[_zone] = dict(_m.get('base_rates') or {}, big_t=_t, latest=_m['target_date'])
    if signal_base:
        tab_dates['spread-signal'] = next(iter(signal_base.values()))['latest']

    return {
        'dam_forecast_tab_button': dam_forecast_tab_button,
        'dam_forecast_tab_html': dam_forecast_tab_html,
        'rtm_forecast_tab_button': rtm_forecast_tab_button,
        'rtm_forecast_tab_html': rtm_forecast_tab_html,
        'spread_forecast_tab_button': spread_forecast_tab_button,
        'spread_forecast_tab_html': spread_forecast_tab_html,
        'tab_dates': tab_dates,
        'FORECAST_TABS_JSON': json.dumps(forecast_tabs),
        'FORECAST_LATEST_JSON': json.dumps(forecast_latest),
        'FORECAST_HISTORY_JSON': json.dumps(forecast_history),
        'SIGNAL_HISTORY_JSON': json.dumps(signal_history),
        'SIGNAL_BASE_JSON': json.dumps(signal_base),
        'FORECAST_ZONE_JSON': json.dumps(FORECAST_ZONES[0] if FORECAST_ZONES else DEFAULT_ZONE),
        'TIER_COLORS_JSON': json.dumps({(k if isinstance(k, str) else f'{k[0]}|{k[1]}'): v for k, v in TIER_COLORS.items()}),
    }
