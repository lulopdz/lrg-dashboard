"""Shared presentation constants: chart colors and figure sizes.

Deliberately free of any data loading or pandas import, so a page that only needs the look
(generar_portfolio.py, which reads its own small CSV) can pull the palette without importing
dashboard_data.py -- that module reads ~20MB of price/weather CSVs at import time, which is
pure waste for a caller that just wants a hex value."""

# One source of truth for the hex values that carry meaning -- series identity, direction,
# tab-group accents -- across dashboard_figures.py, generar_web.py and generar_portfolio.py
# (and, loosely, generar_simulator.py, which mirrors these by eye since it's a deliberately
# separate page). Structural UI chrome (panel backgrounds, borders, body text) stays local to
# each file's own CSS since it isn't data-carrying and doesn't need to change in lockstep
# with a chart color.
COLORS = {
    'dam': '#3498db',              # DAM / "today" line, market-group tab accent
    'rtm': '#e67e22',              # RTM line
    'predicted': '#9b59b6',        # ML forecast "Predicted" line, predict-group tab accent
    'forecast': '#14b8a6',         # weather/load/wind group tab accent
    'positive': '#2ecc71',         # positive spread / correct call / confident-hour marker
    'negative': '#e74c3c',         # negative spread / incorrect call
    'avg': '#6b7280',              # 7d-average reference line (polished)
    'avg_legacy': '#888',          # 7d-average / other muted secondary reference line
    'prev_day': '#e8a33d',         # previous-day reference line (polished)
    'prev_day_legacy': '#f1c40f',  # previous-day reference line (non-polished)
    'muted': '#666',               # spike/crosshair lines, hlines, secondary reference lines
    'grid': '#242424',             # gridlines
    'ring': '#111',                # marker ring / chart surface color
}

TABLE_BUCKET_SIZE = 100  # $/MWh step size for the discrete table color scales
TABLE_ROW_HEIGHT = 26    # px per date row in the hourly heatmap tables -- height scales with the row count instead of being fixed
PROFILE_HEIGHT = 380     # px for every single-panel line chart (hourly profile + forecast tabs) -- one shared value so they read as one page
SPREAD_HEIGHT = 520      # px for the Spread tab's 2-panel chart -- taller than a single panel needs
