"""
Macro Cycle Dashboard — 20-Year History
========================================
13 macro-cycle charts with descriptive headings and US Interest Rate overlay.

SETUP:  pip install pandas fredapi yfinance matplotlib requests
FRED KEY: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("fredapi not found. Run: pip install fredapi")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("yfinance not found. Run: pip install yfinance")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
FRED_API_KEY = "60a3081989b43df4947f991a37fa7c72"

START = "2004-01-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")

# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
BG          = "#0d1117"
PANEL_BG    = "#161b27"
GRID_CLR    = "#2a3347"
TEXT_CLR    = "#e2e8f4"
CAPTION_BG  = "#1a2035"
CAPTION_CLR = "#9ba8bf"
LINE_MAIN   = "#4f9cf9"   # blue   — primary series
LINE_VEL    = "#ffa726"   # amber  — velocity / secondary
LINE_THR    = "#ef5350"   # red    — risk / US rates / secondary
LINE_ALT    = "#26c6a0"   # teal   — tertiary
LINE_RATE   = "#ef5350"   # red used for US Interest Rates twin axis
SHADE_UP    = "#26c6a0"
SHADE_DN    = "#ef5350"
RECESSION   = "#ff6b6b"

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def fred_get(series_id, fred_obj, retries=3, delay=1.5):
    for attempt in range(retries):
        try:
            s = fred_obj.get_series(series_id, observation_start=START, observation_end=END)
            s.name = series_id
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)
            return s.dropna()
        except Exception as e:
            msg = str(e)
            if attempt < retries - 1 and ("Server Error" in msg or "timeout" in msg.lower()):
                time.sleep(delay)
                continue
            print(f"  [FRED] Could not fetch {series_id}: {e}")
            return pd.Series(dtype=float, name=series_id)

def yf_close(ticker, name=None):
    if not YF_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df["Close"]
        s = s.squeeze()
        s.name = name or ticker
        return s.resample("ME").last().dropna()
    except Exception as e:
        print(f"  [yfinance] Could not fetch {ticker}: {e}")
        return pd.Series(dtype=float)

def resample_monthly(s):
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().dropna()

def velocity(series, window=3):
    return series.diff(window)

def pct_change_yoy(series):
    return series.pct_change(12) * 100

def ma(series, n):
    return series.rolling(n).mean()

def safe_div(a, b):
    result = a / b
    return result.replace([np.inf, -np.inf], np.nan).dropna()

def annotate_last(ax, series, color, fmt="{:.2f}"):
    try:
        s = series.dropna()
        if s.empty:
            return
        ax.annotate(fmt.format(s.iloc[-1]),
                    xy=(s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color,
                    fontsize=8.5, fontweight="bold", va="center", clip_on=False)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
#  DATA PULL
# ═══════════════════════════════════════════════════════════════════════════
print("Connecting to FRED ...")
if not FRED_AVAILABLE:
    raise SystemExit("Install fredapi first: pip install fredapi")

f = Fred(api_key=FRED_API_KEY)
print("Fetching FRED series ...")

# ── US Interest Rates (used as overlay on most charts) ─────────────────────
fedfunds = resample_monthly(fred_get("FEDFUNDS", f))   # Federal Funds Rate

# ── Country CLIs (Chart #1 — % above 100) ─────────────────────────────────
CLI_MAP = {
    "USA":       "USALOLITOAASTSAM",
    "Germany":   "DEULOLITOAASTSAM",
    "Japan":     "JPNLOLITOAASTSAM",
    "UK":        "GBRLOLITOAASTSAM",
    "France":    "FRALOLITOAASTSAM",
    "Italy":     "ITALOLITOAASTSAM",
    "Canada":    "CANLOLITOAASTSAM",
    "China":     "CHNLOLITOAASTSAM",
    "Korea":     "KORLOLITOAASTSAM",
    "Australia": "AUSLOLITOAASTSAM",
    "India":     "INDLOLITOAASTSAM",
}
cli_raw = {}
for country, sid in CLI_MAP.items():
    s = resample_monthly(fred_get(sid, f))
    if not s.empty:
        cli_raw[country] = s

# ── OECD aggregate + China CLI (Chart #2) ─────────────────────────────────
cli_g20 = resample_monthly(fred_get("OECDLOLITOAASTSAM", f))
cli_chn = cli_raw.get("China", pd.Series(dtype=float, name="China_CLI"))

# ── US Manufacturing + New Orders (Charts #3 and #4) ──────────────────────
indpro  = resample_monthly(fred_get("INDPRO",     f))   # Actual Factory Output
ism_no  = resample_monthly(fred_get("AMTMNO",     f))   # New Orders ($M) — demand pipeline
ism_inv = resample_monthly(fred_get("MNFCTRIRSA", f))   # Inventories

# ── China interest rates (Charts #5 and #6) ───────────────────────────────
cn_long  = resample_monthly(fred_get("INTDSRCNM193N",   f))  # lending rate (LT proxy)
cn_short = resample_monthly(fred_get("IR3TIB01CNM156N", f))  # 3M interbank

# ── Credit spreads (Chart #7) ─────────────────────────────────────────────
baa = resample_monthly(fred_get("BAA", f))
aaa = resample_monthly(fred_get("AAA", f))

# ── DXY (Chart #8) ────────────────────────────────────────────────────────
print("Fetching yfinance data ...")
dxy = yf_close("DX-Y.NYB", "DXY")

# ── Germany IP + New Orders (Chart #9) ────────────────────────────────────
de_ip = resample_monthly(fred_get("DEUPROINDMISMEI",    f))
de_no = resample_monthly(fred_get("DEUODMNTO01IXOBSAM", f))
if de_no.empty:
    de_no = de_ip.copy()

# ── US Money / M2 / CPI (Charts #10 and #11) ──────────────────────────────
m2      = resample_monthly(fred_get("M2SL",     f))
cpi     = resample_monthly(fred_get("CPIAUCSL", f))
gdp_nom = resample_monthly(fred_get("GDP",      f)).reindex(m2.index).ffill()

# ── Housing / Mortgage (Chart #12) ────────────────────────────────────────
nahb   = resample_monthly(fred_get("HOUST",        f))   # housing starts proxy
mort30 = resample_monthly(fred_get("MORTGAGE30US", f))

# ── Bond yields (Chart #13) ───────────────────────────────────────────────
us10y  = resample_monthly(fred_get("GS10",           f))
de_10y = resample_monthly(fred_get("IRLTLT01DEM156N", f))

print("Data fetch complete. Building derived series ...\n")

# ═══════════════════════════════════════════════════════════════════════════
#  DERIVED SERIES
# ═══════════════════════════════════════════════════════════════════════════

# Chart 1 — % of nations with CLI > 100
if cli_raw:
    cli_df  = pd.DataFrame(cli_raw).resample("ME").last()
    valid   = cli_df.notna().sum(axis=1) >= 4
    cli_pct = (cli_df[valid] > 100).sum(axis=1) / cli_df[valid].notna().sum(axis=1) * 100
    cli_pct.name = "pct_above_100"
else:
    cli_pct = pd.Series(dtype=float, name="pct_above_100")
cli_pct_ma  = ma(cli_pct, 3)

# Chart 2 — China vs G20 CLI (absolute levels)
if cli_g20.empty and cli_raw:
    cli_g20 = pd.DataFrame(cli_raw).mean(axis=1)
    cli_g20.name = "G20_proxy"
cli_vel_diff = (cli_chn - cli_g20).dropna()   # shows gap between China and world

# Chart 3 — New orders YoY vs actual output YoY
no_yoy     = pct_change_yoy(ism_no)     # new orders growth = forward demand
indpro_yoy = pct_change_yoy(indpro)     # factory output growth = coincident

# Chart 4 — New Orders / Inventories oscillator
no_inv     = safe_div(ism_no, ism_inv)
no_inv_ma  = no_inv.rolling(3).mean()
no_inv_vel = velocity(no_inv_ma, 3)

# Chart 5 — Chinese yield curve
cn_curve     = (cn_long - cn_short).dropna()
cn_curve_vel = velocity(cn_curve, 3)

# Chart 6 — Chinese rate velocity (multi-horizon)
cn_vel3  = velocity(cn_long,  3)
cn_vel6  = velocity(cn_long,  6)
cn_vel12 = velocity(cn_long, 12)

# Chart 7 — Credit spread
credit_spread     = (baa - aaa).dropna()
credit_spread_vel = velocity(credit_spread, 3)

# Chart 8 — Dollar liquidity
dxy_vel = velocity(dxy, 3) * -1   # positive = USD weakening = liquidity expanding

# Chart 9 — German IP vs New Orders spread
de_ip_yoy = pct_change_yoy(de_ip)
de_no_yoy = pct_change_yoy(de_no)
de_spread = (de_no_yoy - de_ip_yoy).dropna()   # positive = orders ahead of output

# Chart 10 — US money creation
m2_yoy        = pct_change_yoy(m2)
cpi_yoy       = pct_change_yoy(cpi)
m2_lead       = m2_yoy.copy()
m2_lead.index = m2_lead.index + pd.DateOffset(months=18)

# Chart 11 — M2 velocity
m2_velocity = safe_div(gdp_nom * 1e9, m2 * 1e9)

# Chart 12 — Housing vs mortgage rate velocity
mort_vel = velocity(mort30, 1)   # positive = rates rising = hurts housing

# Chart 13 — Bond yield vs 48-month MA
us10y_ma48   = ma(us10y, 48)
us10y_vs_ma  = (us10y - us10y_ma48).dropna()
de_10y_ma48  = ma(de_10y, 48)
de_10y_vs_ma = (de_10y - de_10y_ma48).dropna()

# ═══════════════════════════════════════════════════════════════════════════
#  CHART DEFINITIONS  (13 charts)
# ═══════════════════════════════════════════════════════════════════════════
charts = [

    # ── 1 ─────────────────────────────────────────────────────────────────
    {
        "title": "#1   OECD CLI's — % of Nations Above Trend",
        "sub":   "Global Growth Pulse:  Rising = more countries expanding  |  Falling = slowdown spreading worldwide",
        "series": [
            (cli_pct,    "% of Countries Growing  (above 50% = world majority expanding)", LINE_MAIN, 2.4, "-"),
            (cli_pct_ma, "3-Month Moving Average",                                          LINE_ALT,  1.4, "--"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — higher = Fed braking global growth", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [(50, "50% Threshold — Half of World at Trend", "#666677")],
        "shade_above":  (cli_pct, 50, SHADE_UP),
        "shade_below":  (cli_pct, 50, SHADE_DN),
        "ylabel":       "% of Nations",
        "caption": (
            "BLUE  '%% of Countries Growing' — above 50%% = majority of world economy expanding\n"
            "RED   'US Interest Rates' (right axis) — higher = Fed is braking global growth\n"
            "GREEN zone = global expansion dominant   RED zone = widespread slowdown   Grey bars = NBER recessions"
        ),
    },

    # ── 2 ─────────────────────────────────────────────────────────────────
    {
        "title": "#2   OECD China / G20 CLI Ratio",
        "sub":   "China vs World Speed:  Rising = China outpacing global growth  |  Falling = China dragging behind",
        "series": [
            (cli_chn,      "China Growth Momentum — leading indicator for commodity demand",  LINE_MAIN, 2.4, "-"),
            (cli_g20,      "G20 Growth Momentum — benchmark for rest of world",               LINE_THR,  2.0, "--"),
            (cli_vel_diff, "Gap (China minus G20) — positive = China outperforming",          LINE_VEL,  1.2, ":"),
        ],
        "hlines":  [(100, "Above-Trend Threshold (100)", "#666677")],
        "ylabel":  "CLI Index / Gap",
        "caption": (
            "BLUE  'China Growth Momentum' — when above 100 and rising, commodity demand accelerates\n"
            "RED   'G20 Growth Momentum' — benchmark; when China (blue) is above G20 (red) = China outpacing world\n"
            "AMBER 'Gap' — positive gap = China accelerating faster, leads EM assets and commodity prices"
        ),
    },

    # ── 3 ─────────────────────────────────────────────────────────────────
    {
        "title": "#3   US ISM — Manufacturing Indicator",
        "sub":   "US Factory Reality Check:  Rising = industry healthy  |  Falling orders below output = contraction coming",
        "series": [
            (no_yoy,     "New Orders Growth YoY %% — what businesses are ordering (forward-looking)", LINE_MAIN, 2.4, "-"),
            (indpro_yoy, "Actual Factory Output YoY %% — what factories are producing (coincident)",  LINE_THR,  2.0, "--"),
        ],
        "hlines":  [(0, "Zero Growth (contraction below this line)", "#666677")],
        "shade_above": (indpro_yoy, 0, SHADE_UP),
        "shade_below": (indpro_yoy, 0, SHADE_DN),
        "ylabel":  "Year-over-Year %",
        "caption": (
            "BLUE  'New Orders Growth' — what businesses think is coming; leads output by 1-3 months\n"
            "RED   'Actual Factory Output' — what factories are really producing right now\n"
            "Gap widening (blue above red) = production set to catch up = expansion.  Red zone = output contraction"
        ),
    },

    # ── 4 ─────────────────────────────────────────────────────────────────
    {
        "title": "#4   US ISM New Orders / Inventories Ratio  (Oscillator & Velocity)",
        "sub":   "US Demand Pipeline:  Rising = factories need to produce more  |  Falling = warehouses full, cuts coming",
        "series": [
            (no_inv_ma,  "Demand vs Stock Ratio (3M MA) — high = shelves empty, orders flooding in", LINE_MAIN, 2.4, "-"),
            (no_inv,     "Raw Ratio",                                                                  LINE_ALT,  0.9, "-"),
            (no_inv_vel, "Rate of Change — how fast the ratio is shifting",                            LINE_VEL,  1.4, "--"),
        ],
        "twin_series": [(fedfunds, "US Interest Rates — context for credit conditions", LINE_RATE, 1.4, "--")],
        "twin_ylabel": "Fed Funds Rate (%)",
        "hlines":      [(1.0, "Balanced Demand / Supply (1.0)", "#666677")],
        "ylabel":      "Ratio / Rate of Change",
        "caption": (
            "BLUE  'Demand vs Stock Ratio' — above 1.0 = new orders outpacing inventory = production must rise\n"
            "AMBER 'Rate of Change' — a rising oscillator turning positive is the early-cycle demand signal\n"
            "RED   'US Interest Rates' (right axis) — higher rates tighten credit and depress new orders"
        ),
    },

    # ── 5 ─────────────────────────────────────────────────────────────────
    {
        "title": "#5   Chinese Yield Curve  (Lending Rate – 3M Interbank Rate)",
        "sub":   "China Bank Lending Health:  Rising = credit flows freely  |  Flat / Negative = credit drying up",
        "series": [
            (cn_curve,     "Long vs Short Rate Gap — wide = healthy lending, narrow/inverted = credit crunch", LINE_MAIN, 2.4, "-"),
            (cn_curve_vel, "3-Month Velocity — rate of steepening or flattening",                               LINE_VEL,  1.4, "--"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — Fed policy exports financial stress to China", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [(0, "Flat / Inversion Threshold", "#666677")],
        "shade_below":  (cn_curve, 0, SHADE_DN),
        "ylabel":       "Spread (pp) / Velocity",
        "caption": (
            "BLUE  'Long vs Short Rate Gap' — wide and rising = PBoC easing, credit flowing, positive for global growth\n"
            "AMBER 'Velocity' — shows how fast the curve is steepening or flattening\n"
            "RED   'US Interest Rates' (right axis) — when the Fed hikes, stress transmits to China's financial system"
        ),
    },

    # ── 6 ─────────────────────────────────────────────────────────────────
    {
        "title": "#6   Chinese Interest Rate Velocity  (3M / 6M / 12M Horizons)",
        "sub":   "Beijing's Emergency Response Speed:  Dropping fast = China cutting urgently  |  Rising = China tightening",
        "series": [
            (cn_vel3,  "3M Speed of China Rate Changes — big negative spike = panic cutting", LINE_MAIN, 2.4, "-"),
            (cn_vel6,  "6M Speed of Rate Changes",                                             LINE_VEL,  1.6, "--"),
            (cn_vel12, "12M Speed of Rate Changes — slow structural trend",                    LINE_ALT,  1.2, ":"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — divergence with China = currency/capital flow pressure", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [(0, "Neutral — No Rate Change", "#666677")],
        "ylabel":       "Rate Change (pp)",
        "caption": (
            "BLUE  '3M Speed' — big negative spike = China cutting rates urgently to save growth (risk-on signal)\n"
            "AMBER '6M Speed' — medium-term view of China's rate direction\n"
            "TEAL  '12M Speed' — slow-moving structural trend; when all three below zero = broad China easing cycle\n"
            "RED   'US Interest Rates' (right axis) — divergence between Fed hiking and China cutting = EM currency risk"
        ),
    },

    # ── 7 ─────────────────────────────────────────────────────────────────
    {
        "title": "#7   Baa / Aaa Spread Velocity",
        "sub":   "Corporate Debt Danger Gauge:  Rising = stress accelerating, default risk growing  |  Falling = credit calming",
        "series": [
            (credit_spread,     "Speed of Credit Stress — spikes preceded every major crisis (2008, 2020)", LINE_MAIN, 2.4, "-"),
            (credit_spread_vel, "Velocity of Spread Change — how fast stress is building",                  LINE_VEL,  1.4, "--"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — higher rates widen spreads with a lag", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [],
        "ylabel":       "Spread (%) / Velocity",
        "caption": (
            "BLUE  'Speed of Credit Stress' — Baa vs Aaa yield gap; rising = rising default risk in corporate bonds\n"
            "AMBER 'Velocity' — how fast stress is accelerating; the 2008 and 2020 spikes gave early crisis warning\n"
            "RED   'US Interest Rates' (right axis) — rate hikes tighten credit conditions, widen spreads with a lag"
        ),
    },

    # ── 8 ─────────────────────────────────────────────────────────────────
    {
        "title": "#8   World Dollar Liquidity Velocity",
        "sub":   "Global Financial Oxygen:  Rising = plenty of dollars to invest  |  Falling = dollar scarcity tightening everywhere",
        "series": [
            (dxy,     "Dollar Liquidity Level (inverted DXY) — most assets rise when this rises", LINE_MAIN, 2.4, "-"),
            (dxy_vel, "Rate of Change — how fast liquidity is being added or drained",             LINE_VEL,  1.4, "--"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — Fed hikes directly drain global dollar supply", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [(0, "Neutral — No Liquidity Change", "#666677")],
        "ylabel":       "DXY Index / Velocity",
        "caption": (
            "BLUE  'Dollar Liquidity Level' (DXY shown raw — falling DXY = more global dollar liquidity)\n"
            "AMBER 'Rate of Change' — positive = dollar weakening = liquidity expanding = risk-on globally\n"
            "RED   'US Interest Rates' (right axis) — Fed rate hikes are the primary mechanism draining global dollar supply"
        ),
    },

    # ── 9 ─────────────────────────────────────────────────────────────────
    {
        "title": "#9   German Industrial Production vs New Orders Spread",
        "sub":   "Europe's Factory Order Book:  Rising = more orders than factories can fill  |  Falling = orders drying up before output slows",
        "series": [
            (de_no_yoy, "Incoming Orders YoY %% — what customers are buying from German factories",  LINE_MAIN, 2.4, "-"),
            (de_ip_yoy, "Current Output YoY %% — what factories are currently making",               LINE_THR,  2.0, "--"),
            (de_spread, "Gap (Orders minus Output) — positive = growth coming in 6 months",          LINE_ALT,  1.2, ":"),
        ],
        "hlines":  [(0, "Zero Growth Threshold", "#666677")],
        "ylabel":  "Year-over-Year %",
        "caption": (
            "BLUE  'Incoming Orders' — forward-looking; leads production by approximately 6 months\n"
            "RED   'Current Output' — coincident; what factories are actually making right now\n"
            "TEAL  'Gap' (Orders minus Output) — positive gap widening upward = EU industrial expansion ahead\n"
            "Gap narrowing or turning negative = EU industrial slowdown 2-3 quarters ahead"
        ),
    },

    # ── 10 ────────────────────────────────────────────────────────────────
    {
        "title": "#10  US Money Creation  (M2 YoY %)",
        "sub":   "Inflation Early Warning:  M2 rising fast today = inflation in 12-18 months  |  M2 falling = inflation pressure fading",
        "series": [
            (m2_yoy,  "Money Supply Growth Rate — the fuel going into the inflation engine",                     LINE_MAIN, 2.4, "-"),
            (m2_lead, "M2 shifted +18 months forward — where inflation will likely be in 18 months",            LINE_VEL,  1.6, "--"),
            (cpi_yoy, "Actual CPI Inflation — the fire that results, lagged 12-18 months behind M2",            LINE_ALT,  1.2, ":"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — Fed's tool to reduce money supply growth", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [],
        "ylabel":       "Year-over-Year %",
        "caption": (
            "BLUE  'Money Supply Growth Rate' — M2 YoY%%; the leading fuel for inflation\n"
            "AMBER 'M2 Shifted +18M' — shows where CPI is likely headed; tracked 2022 inflation surge accurately\n"
            "TEAL  'Actual CPI Inflation' — realised inflation; always lags M2 by 12-18 months\n"
            "RED   'US Interest Rates' (right axis) — the Fed's primary tool to reduce money supply growth"
        ),
    },

    # ── 11 ────────────────────────────────────────────────────────────────
    {
        "title": "#11  US M2 Velocity  (Nominal GDP / M2)",
        "sub":   "Is Money Actually Working?  Rising = each dollar spent more times, economy active  |  Falling = money hoarded, growth weak despite printing",
        "series": [
            (m2_velocity,        "GDP per Dollar of Money Supply — high = economy efficiently using money", LINE_MAIN, 2.4, "-"),
            (ma(m2_velocity, 12), "12-Month Moving Average",                                                LINE_ALT,  1.4, "--"),
        ],
        "twin_series":  [(fedfunds, "US Interest Rates — higher rates encourage saving over spending, pushing velocity down", LINE_RATE, 1.4, "--")],
        "twin_ylabel":  "Fed Funds Rate (%)",
        "hlines":       [],
        "ylabel":       "Velocity (turns/year)",
        "caption": (
            "BLUE  'GDP per Dollar of Money Supply' — V = Nominal GDP / M2; how hard each dollar works\n"
            "TEAL  '12M Moving Average' — structural trend; velocity has declined since 1990s (money less efficient)\n"
            "RED   'US Interest Rates' (right axis) — higher rates encourage saving over spending, crushing velocity"
        ),
    },

    # ── 12 ────────────────────────────────────────────────────────────────
    {
        "title": "#12  NAHB Housing Index  vs  Interest Rate Velocity",
        "sub":   "Housing Market Stress Meter:  NAHB falling while rates rise fast = housing recession  |  NAHB recovering while rates slow = stabilising",
        "series": [
            (nahb,     "Homebuilder Confidence (Housing Starts) — falls 6-12 months before broader recession", LINE_MAIN, 2.4, "-"),
            (mort_vel, "Speed of Rate Changes (+ve = rates rising, hurts housing)",                             LINE_THR,  1.8, "--"),
        ],
        "hlines":  [(0, "Rate Velocity Neutral (rates not moving)", "#666677")],
        "ylabel":  "Starts (000s) / Rate Velocity",
        "caption": (
            "BLUE  'Homebuilder Confidence' (Housing Starts proxy) — leading recession indicator; falls 6-12 months early\n"
            "RED   'Speed of Rate Changes' — positive = rates rising fast = kills housing affordability quickly\n"
            "When red spikes up (fast rate hike) and blue falls = high risk of housing-led slowdown ahead"
        ),
    },

    # ── 13 ────────────────────────────────────────────────────────────────
    {
        "title": "#13  EU & US Bond Market Velocity  (Yield vs 48-Month MA Spread)",
        "sub":   "Are Rates Historically Expensive or Cheap?  Above zero = rates HIGH vs history = tight  |  Below zero = rates LOW = money cheap",
        "series": [
            (us10y_vs_ma, "US Bond Yield vs 4-Year Average — above zero = rates historically expensive", LINE_MAIN, 2.4, "-"),
            (de_10y_vs_ma,"EU Bond Yield vs 4-Year Average — above zero = EU rates historically expensive", LINE_THR, 2.0, "--"),
            (us10y_ma48,  "US 48-Month Moving Average (the historical 'normal' reference)",               LINE_VEL,  1.2, ":"),
        ],
        "hlines":  [(0, "Zero = Yields at 4-Year Historical Average (Neutral)", "#666677")],
        "ylabel":  "Spread vs 48M Average (pp)",
        "caption": (
            "BLUE  'US Yield vs 4-Year Average' — above zero = US rates expensive vs history = tight financial conditions\n"
            "RED   'EU Yield vs 4-Year Average' — same for Europe; both above zero = globally synchronised tightening\n"
            "AMBER 'US 48M Moving Average' — the 4-year 'normal' level; when yields fall below this, rate cuts are being priced\n"
            "Both spreads simultaneously below zero = coordinated global easing = maximum support for all asset classes"
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "text.color":       TEXT_CLR,
    "axes.labelcolor":  TEXT_CLR,
    "xtick.color":      TEXT_CLR,
    "ytick.color":      TEXT_CLR,
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL_BG,
})

# ═══════════════════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ═══════════════════════════════════════════════════════════════════════════
N               = len(charts)
FIG_H_PER_CHART = 5.8
FIG_W           = 24

print(f"Rendering {N} charts ...")

fig, axes = plt.subplots(
    nrows=N, ncols=1,
    figsize=(FIG_W, FIG_H_PER_CHART * N),
    constrained_layout=False,
)
axes = np.atleast_1d(axes)
fig.patch.set_facecolor(BG)

RECESSIONS = [
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

def shade_recessions(ax):
    for s, e in RECESSIONS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=RECESSION, alpha=0.10, zorder=0, label="_nolegend_")

# ── Draw each panel ───────────────────────────────────────────────────────
for i, (ax, ch) in enumerate(zip(axes, charts)):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
        spine.set_linewidth(0.8)

    ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="both", labelsize=9.5, colors=TEXT_CLR, length=4, width=0.7)
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))

    shade_recessions(ax)

    # ── Zone fills ────────────────────────────────────────────────────────
    for key, above in [("shade_above", True), ("shade_below", False)]:
        if key not in ch:
            continue
        s_ref, thr, col = ch[key]
        try:
            sr = s_ref.dropna()
            if sr.empty:
                continue
            if above:
                ax.fill_between(sr.index, thr, sr.values,
                                where=(sr.values >= thr), color=col, alpha=0.15, zorder=1)
            else:
                ax.fill_between(sr.index, sr.values, thr,
                                where=(sr.values <= thr), color=col, alpha=0.15, zorder=1)
        except Exception:
            pass

    # ── Primary series lines ──────────────────────────────────────────────
    legend_handles = []
    first_series = True
    for (s, label, color, lw, ls) in ch["series"]:
        try:
            s_plot = s.loc[START:END].dropna()
            if s_plot.empty:
                continue
            ax.plot(s_plot.index, s_plot.values,
                    color=color, linewidth=lw, linestyle=ls, alpha=0.93, zorder=3)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=lw, linestyle=ls, label=label)
            )
            if first_series:
                annotate_last(ax, s_plot, color)
                first_series = False
        except Exception as e:
            print(f"  [plot] {ch['title'][:40]} ... series error: {e}")

    # ── Reference lines ───────────────────────────────────────────────────
    for (val, label, color) in ch.get("hlines", []):
        ax.axhline(val, color=color, linewidth=1.0, linestyle=":", alpha=0.80, zorder=2)
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.0, linestyle=":", label=label)
        )

    if i == 0:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=RECESSION, alpha=0.3, label="NBER Recession")
        )

    # ── Twin axis (US Interest Rates overlay) ─────────────────────────────
    if ch.get("twin_series"):
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        for (s, label, color, lw, ls) in ch["twin_series"]:
            try:
                s_plot = s.loc[START:END].dropna()
                if s_plot.empty:
                    continue
                ax2.plot(s_plot.index, s_plot.values,
                         color=color, linewidth=lw, linestyle=ls, alpha=0.45, zorder=2)
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=lw, linestyle=ls,
                           label=label, alpha=0.7)
                )
            except Exception:
                pass
        twin_ylabel = ch.get("twin_ylabel", "Rate (%)")
        ax2.set_ylabel(twin_ylabel, fontsize=9, color=LINE_RATE, labelpad=4)
        ax2.tick_params(axis="y", labelsize=8.5, colors=LINE_RATE, length=3, width=0.7)
        for sname, sp in ax2.spines.items():
            sp.set_visible(sname == "right")
            if sname == "right":
                sp.set_edgecolor(LINE_RATE)
                sp.set_linewidth(0.6)
                sp.set_alpha(0.5)

    # ── Title (two-line: chart name + rising/falling meaning) ─────────────
    full_title = ch["title"]
    if ch.get("sub"):
        full_title += f"\n{ch['sub']}"

    ax.set_title(full_title, fontsize=11, fontweight="bold",
                 color=TEXT_CLR, pad=10, loc="left", linespacing=1.5)
    ax.set_ylabel(ch["ylabel"], fontsize=10, color=CAPTION_CLR, labelpad=6)

    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=8, framealpha=0.45,
              facecolor=PANEL_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_CLR, ncol=2)

    # ── Caption box (series descriptions) ────────────────────────────────
    ax.text(
        0.0, -0.22, ch["caption"],
        transform=ax.transAxes,
        fontsize=8.2, color=CAPTION_CLR,
        verticalalignment="top",
        family="monospace", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor=CAPTION_BG, edgecolor=GRID_CLR, alpha=0.95),
    )

# ── Super-title ───────────────────────────────────────────────────────────
fig.suptitle(
    f"Macro Cycle Dashboard — 20-Year History  (2004 – {END[:4]})",
    fontsize=17, fontweight="bold", color=TEXT_CLR, y=1.001,
)

fig.subplots_adjust(hspace=1.15, top=0.998, bottom=0.01, left=0.06, right=0.94)

# ── Save ──────────────────────────────────────────────────────────────────
OUT = "macro_dashboard_20yr.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved -> {OUT}")
print("Showing interactive window ...")
plt.show()
