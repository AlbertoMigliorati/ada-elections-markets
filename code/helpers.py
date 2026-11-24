"""Shared settings and functions for the elections project."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"    # written by notebooks 03-07, read by the next stage; gitignored
FIGURES = ROOT / "figures"    # written only by 08_figures_tables

# Data vintage: the date in the name of every raw file and the (excluded) end date of every Yahoo request, so all
# price series stop on the same day. Each file is downloaded ONCE (01_get_data) and every notebook reads these
# files, never a fresh download. The day each file was actually downloaded is in data/download_log.csv
# (a file added later keeps this vintage in its name but has its own download day in the log).
DOWNLOAD_DATE = "2026-11-19"
PRICES_START = "1995-01-01"   # first day requested from Yahoo for every price series

# ParlGov country code - Yahoo Finance ticker of the national stock index.
TICKERS = {
    "AUS": "^AXJO",
    "AUT": "^ATX",
    "BEL": "^BFX",
    "CAN": "^GSPTSE",
    "CHE": "^SSMI",
    "DEU": "^GDAXI",
    "ESP": "^IBEX",
    "FRA": "^FCHI",
    "GBR": "^FTSE",
    "GRC": "GD.AT",
    "IRL": "^ISEQ",
    "ISR": "TA35.TA",     # trades Sunday-Thursday until Dec 2025; Yahoo has no Sunday observations
    "ITA": "FTSEMIB.MI",
    "JPN": "^N225",
    "NLD": "^AEX",
    "TUR": "XU100.IS",
    "NZL": "^NZ50",
    "SWE": "^OMX",
    "FIN": "^OMXH25",
    "PRT": "PSI20.LS",
    "EST": "^OMXTGI",
    "LTU": "^OMXVGI",
    "DNK": "^OMXC25",
}

# External market factor of the market model: candidates in order of preference (Yahoo ticker - name).
# load_external() uses the first one whose series covers the whole period of the national indices.
EXTERNAL_CANDIDATES = {"^990100-USD-STRD": "MSCI World", "^GSPC": "S&P 500"}

PARLGOV_URL = "https://parlgov.org/data/parlgov-development_csv-utf-8/"
PARLGOV_TABLES = ["view_election", "view_cabinet"]

EVENTS_FROM = "1997-01-01"   # events (elections, cabinets) from this date on
MIN_HISTORY = 270            # trading days of index history needed before day 0


# =====================================================================================================
# ANALYSIS SETTINGS
# Every number used by the analysis, in one place. Notebooks import these constants and never redefine them.
# Windows are in trading days relative to day 0 (on each index's own calendar) unless stated otherwise.
# =====================================================================================================

SEED = 20261119                       # every random step
# Sample and calendar
SAMPLE_END = "2023-06-25"             # last ParlGov date; normal days end here
OVERLAP_DROP = 1                      # same-country day 0 at most this far apart: keep the earlier
OVERLAP_FLAG = 10                     # ... more than 1 and at most this far apart: flag overlap10
LONG_GAP_DAYS = 4                     # a return spanning more than this many CALENDAR days is a long gap
ISR_TWO_SESSION_WEEKDAY = 0           # Monday: Israeli return spans two sessions (Sunday missing)
ISR_SUNDAY_TRADING_END = "2025-12-28" # last Sunday-Thursday week of TASE (after the sample)
# Returns
STALE_ZERO_RUN = 2                    # run of >= this many consecutive zero returns: repeated closes missing
# Closes of the frozen file that are not the index (checked against the exchange's daily official lists):
# treated as missing, both dates included.
INVALID_CLOSES = {"GRC": [("2014-12-19", "2015-05-12")]}   # Yahoo GD.AT is another series in this span
EXTREME_RETURN = 0.15                 # |log return| above this is listed, not removed
# Global factor
GLOBAL_ENTRY_HISTORY = 250            # trading days of own history before an index enters the factor
GLOBAL_MAX_STALE_DAYS = 5             # CALENDAR days: max age of index j's close at the span start
MIN_GLOBAL = 8                        # minimum contributing indices, else factor missing
# External factor (candidates in EXTERNAL_CANDIDATES, chosen by load_external)
EXTERNAL_MAX_GAP_DAYS = 7             # longest allowed gap between two closes of the series (calendar days)
# Market model and abnormal returns
EST_WIN = (-250, -20)                 # estimation window, both ends included
MIN_EST_DAYS = 150                    # valid estimation days after exclusions, else event dropped
POL_EXCL = 1                          # exclude +-1 trading day around political dates from estimation
RESID_DDOF = 3                        # residual sd: alpha, beta0, beta1
AR_WIN = (-19, 10)                    # days for which AR and SAR are stored (outside EST_WIN)
NORMAL_EXCL = 10                      # normal day: more than +-10 trading days from any political date
PRE_VOL_WIN = (-19, -1)               # window of vol_pre, outside EST_WIN
VOL_DDOF = 1                          # ddof of the sd of AR in vol_pre and of sd_c
# Outcome
P_MAIN = 0.95                         # headline threshold quantile
P_SEC = 0.90                          # single secondary threshold quantile
OUTCOME_DAYS = (0, 1)                 # strong = max |AR| over day 0 and day +1
# Predictors
MAJORITY_SHARE = 0.5                  # minority if cabinet seats below this share
LR_UNKNOWN_FLAG = 0.10                # elections with more unknown left_right seat share are listed
# Event study and placebo
PLACEBO_RADIUS = 730                  # CALENDAR days around day 0 for the matched pool
PLACEBO_MIN_PAIRS = 100               # fewer pairs: use the wide radius (once)
PLACEBO_RADIUS_WIDE = 1460            # CALENDAR days
B_BOOT = 5000                         # country-cluster bootstrap replicates
CI_LEVEL = 0.95                       # percentile intervals
N_GROUP_PERM = 10000                  # permutations of the group label within country (crises vs elections)
# Prediction
PURGE_DAYS = 7                        # CALENDAR days: training events this close to a test event removed
INNER_FOLDS = 5                       # GroupKFold by country for the penalty
C_GRID = tuple(np.logspace(-3, 2, 11))  # inverse penalty grid, 0.001 ... 100
N_PERM = 999                          # permutations of the political block (prediction test)
N_PERM_FALLBACK = 499                 # used if N_PERM runs would exceed the time limit
PERM_TIME_LIMIT_MIN = 45              # minutes, measured in the Nuvolos workspace
# Tests
ALPHA = 0.05                          # Holm correction across the main tests
# Robustness
COVID_WINDOW = ("2020-02-24", "2020-03-31")  # events with day 0 or +1 inside are dropped
POWER_RATES = (0.15, 0.20, 0.25, 0.30)       # power of the crisis test at these true rates
# Reaction types
PATH_WIN = (-10, 10)                  # daily SAR path, 21 values
N_PLACEBO_SAMPLES = 200               # placebo samples of the same size
PARALLEL_PCT = 95                     # percentile of placebo eigenvalues
K_GRID = tuple(range(1, 7))           # number of clusters, BIC; K = 1 allowed
GMM_N_INIT = 10                       # Gaussian mixture, diagonal covariance
KMEANS_N_INIT = 10                    # k-means check
NAME_PEAK = 2                         # naming rules for clusters
NAME_QUIET = 1                        # naming rules for clusters
# Random forest (optional)
RF_N_TREES = 500
RF_MIN_LEAF_GRID = (5, 10, 20)        # chosen by the inner validation
# =====================================================================================================


def raw_file(prefix):
    """Path of the frozen raw file, e.g. raw_file('index_prices') -> data/raw/index_prices_<DOWNLOAD_DATE>.csv."""
    path = RAW / f"{prefix}_{DOWNLOAD_DATE}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_get_data first.")
    return path


def load_derived(name, parse_dates=None):
    """Read data/derived/<name>.csv, written by an earlier notebook (03-07)."""
    path = DERIVED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the notebook that writes it first (see README).")
    return pd.read_csv(path, parse_dates=parse_dates)


def save_derived(df, name):
    """Write df to data/derived/<name>.csv (no index) and return the path."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    path = DERIVED / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_prices():
    """Daily closes of the 23 indices, one column per Yahoo ticker."""
    return pd.read_csv(raw_file("index_prices"), index_col="date", parse_dates=True)


def load_external():
    """Daily closes of the external market factor: the first of EXTERNAL_CANDIDATES whose series starts within a
    week of PRICES_START, runs to SAMPLE_END and has no gap longer than EXTERNAL_MAX_GAP_DAYS.
    Returns (closes, ticker)."""
    ext = pd.read_csv(raw_file("external_index"), index_col="date", parse_dates=True)
    start = pd.Timestamp(PRICES_START) + pd.Timedelta(days=7)
    for ticker in EXTERNAL_CANDIDATES:
        s = ext[ticker].dropna() if ticker in ext else pd.Series(dtype=float)
        if (len(s) and s.index[0] <= start and s.index[-1] >= pd.Timestamp(SAMPLE_END)
                and s.index.to_series().diff().dt.days.max() <= EXTERNAL_MAX_GAP_DAYS):
            return s.rename(ticker), ticker
    raise ValueError("no external candidate covers the period of the national indices")


def load_elections():
    """One row per parliamentary election since EVENTS_FROM in the 23 countries."""
    el = pd.read_csv(raw_file("parlgov_view_election"), parse_dates=["election_date"])
    el = el.drop_duplicates("election_id")
    el = el[(el["election_type"] == "parliament")
            & (el["election_date"] >= EVENTS_FROM)
            & el["country_name_short"].isin(TICKERS)]
    return el.sort_values(["country_name_short", "election_date"])


def load_cabinets():
    """ParlGov view_cabinet as frozen: one row per cabinet and party, all countries and years (no filter)."""
    return pd.read_csv(raw_file("parlgov_view_cabinet"), parse_dates=["election_date", "start_date"])


def candidate_cabinets(cabinets):
    """Cabinets that started without a new election since EVENTS_FROM (ParlGov: same election as the previous cabinet)."""
    cab = cabinets.drop_duplicates("cabinet_id")
    prev_election = cab.set_index("cabinet_id")["election_id"]
    cab = cab.assign(prev_election=cab["previous_cabinet_id"].map(prev_election))
    return cab[(cab["start_date"] >= EVENTS_FROM)
               & cab["country_name_short"].isin(TICKERS)
               & (cab["election_id"] == cab["prev_election"])]


def eligible_from(prices, country, n=MIN_HISTORY):
    """First date with at least n earlier trading days of index history (NaT if the series is too short)."""
    s = prices[TICKERS[country]].dropna()
    return s.index[n] if len(s) > n else pd.NaT
