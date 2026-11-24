"""Paths, settings and data loaders of the elections project."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"
RAW, DERIVED = DATA / "raw", DATA / "derived"     # derived: written by notebooks 02-06, gitignored

# Data vintage: the date in the name of every raw file and the (excluded) end date of every Yahoo request. Each file
# is downloaded once by 01_get_data; the day it was actually downloaded is in data/download_log.csv.
DOWNLOAD_DATE = "2026-11-19"
PRICES_START = "1995-01-01"
PARLGOV_URL = "https://parlgov.org/data/parlgov-development_csv-utf-8/"
PARLGOV_TABLES = ["view_election", "view_cabinet"]

# ParlGov country code -> Yahoo ticker of the national index. Yahoo has no Sunday closes for TASE (ISR).
TICKERS = {"AUS": "^AXJO", "AUT": "^ATX", "BEL": "^BFX", "CAN": "^GSPTSE", "CHE": "^SSMI", "DEU": "^GDAXI",
           "ESP": "^IBEX", "FRA": "^FCHI", "GBR": "^FTSE", "GRC": "GD.AT", "IRL": "^ISEQ", "ISR": "TA35.TA",
           "ITA": "FTSEMIB.MI", "JPN": "^N225", "NLD": "^AEX", "TUR": "XU100.IS", "NZL": "^NZ50", "SWE": "^OMX",
           "FIN": "^OMXH25", "PRT": "PSI20.LS", "EST": "^OMXTGI", "LTU": "^OMXVGI", "DNK": "^OMXC25"}
# External market factor: candidates in order of preference; load_external() takes the first that covers the period.
EXTERNAL_CANDIDATES = {"^990100-USD-STRD": "MSCI World", "^GSPC": "S&P 500"}

# =====================================================================================================
# ANALYSIS SETTINGS: every number of the analysis. Notebooks import them and never redefine them.
# Windows are in trading days relative to day 0, on each index's own calendar, unless stated otherwise.
# =====================================================================================================
SEED = 20261119
# Sample and calendar
EVENTS_FROM = "1997-01-01"            # events from this date on
SAMPLE_END = "2023-06-25"             # last ParlGov date; normal days end here
MIN_HISTORY = 270                     # trading days of index history needed before day 0
OVERLAP_DROP = 1                      # same-country day 0 at most this far apart: keep the earlier
LONG_GAP_DAYS = 4                     # CALENDAR days: a longer return is a long-gap return
ISR_TWO_SESSION_WEEKDAY = 0           # Monday: the Israeli return spans two sessions (Sunday missing)
ISR_SUNDAY_TRADING_END = "2025-12-28" # last Sunday-Thursday week of TASE (after the sample)
# Returns
STALE_ZERO_RUN = 2                    # runs of >= this many zero returns: the repeated closes are removed
INVALID_CLOSES = {"GRC": [("2014-12-19", "2015-05-12")]}  # not the index (checked against the Athens Exchange lists)
EXTREME_RETURN = 0.15                 # |log return| above this is listed, not removed
# Factors of the market model
GLOBAL_ENTRY_HISTORY = 250            # trading days of own history before an index enters g
GLOBAL_MAX_STALE_DAYS = 5             # CALENDAR days: max age of a close at the start of a span (g and w)
MIN_GLOBAL = 8                        # fewer contributing indices: g missing
EXTERNAL_MAX_GAP_DAYS = 7             # CALENDAR days: longest allowed gap in the external series
# Market model and abnormal returns
EST_WIN = (-250, -20)                 # estimation window, both ends included
MIN_EST_DAYS = 150                    # valid estimation days needed
POL_EXCL = 1                          # estimation days within this distance of a political date are left out
AR_WIN = (-19, 10)                    # days with a stored AR and SAR
NORMAL_EXCL = 10                      # normal day: farther than this from every political date
PRE_VOL_WIN = (-19, -1)               # window of vol_pre
VOL_DDOF = 1                          # ddof of the sd in vol_pre and sd_c
# Outcome
OUTCOME_DAYS = (0, 1)                 # reaction = max |AR| over day 0 and day +1
P_MAIN, P_SEC = 0.95, 0.90            # quantiles of the binary outcome (robustness)
# Predictors
MAJORITY_SHARE = 0.5                  # minority if the cabinet parties hold fewer seats than this share
# Event study and placebo
PLACEBO_RADIUS = 730                  # CALENDAR days around day 0 for the matched pool
PLACEBO_MIN_PAIRS = 100               # every pool must have at least this many pairs
B_BOOT = 5000                         # country-cluster bootstrap replicates
CI_LEVEL = 0.95                       # percentile intervals
N_GROUP_PERM = 10000                  # permutations of the group label within country (crises vs elections)
ALPHA = 0.05                          # Holm correction across the main tests
COVID_WINDOW = ("2020-02-24", "2020-03-31")  # robustness: events with day 0 or +1 inside are dropped
# Prediction
PURGE_DAYS = 7                        # CALENDAR days: training events this close to a test event are removed
INNER_FOLDS = 5                       # GroupKFold by country for the penalty
N_PERM = 999                          # permutations of the political block
# Reaction types
PATH_WIN = (-10, 10)                  # daily SAR path, 21 values
N_PLACEBO_SAMPLES = 200               # placebo samples of the same size
PARALLEL_PCT = 95                     # percentile of the placebo eigenvalues
K_GRID = tuple(range(1, 7))           # number of clusters by BIC; K = 1 allowed
GMM_N_INIT = KMEANS_N_INIT = 10       # initialisations of the Gaussian mixture and of the k-means check
NAME_PEAK, NAME_QUIET = 2, 1          # naming rules for clusters


def raw_file(prefix):
    """data/raw/<prefix>_<DOWNLOAD_DATE>.csv, which must exist."""
    path = RAW / f"{prefix}_{DOWNLOAD_DATE}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_get_data first.")
    return path


def load_derived(name, parse_dates=None):
    path = DERIVED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the notebook that writes it first (see README).")
    return pd.read_csv(path, parse_dates=parse_dates)


def save_derived(df, name):
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(path := DERIVED / f"{name}.csv", index=False)
    return path


def load_prices():
    """Daily closes of the 23 indices, one column per Yahoo ticker."""
    return pd.read_csv(raw_file("index_prices"), index_col="date", parse_dates=True)


def load_external():
    """(closes, ticker) of the first EXTERNAL_CANDIDATES series that starts within a week of PRICES_START, runs to
    SAMPLE_END and has no gap longer than EXTERNAL_MAX_GAP_DAYS."""
    ext = pd.read_csv(raw_file("external_index"), index_col="date", parse_dates=True)
    for ticker in EXTERNAL_CANDIDATES:
        s = ext[ticker].dropna() if ticker in ext else pd.Series(dtype=float)
        if (len(s) and s.index[0] <= pd.Timestamp(PRICES_START) + pd.Timedelta(days=7)
                and s.index[-1] >= pd.Timestamp(SAMPLE_END) and s.index.diff().days.max() <= EXTERNAL_MAX_GAP_DAYS):
            return s.rename(ticker), ticker
    raise ValueError("no external candidate covers the period of the national indices")


def load_elections():
    """One row per parliamentary election since EVENTS_FROM in the 23 countries."""
    el = pd.read_csv(raw_file("parlgov_view_election"), parse_dates=["election_date"]).drop_duplicates("election_id")
    el = el[(el["election_type"] == "parliament") & (el["election_date"] >= EVENTS_FROM)
            & el["country_name_short"].isin(TICKERS)]
    return el.sort_values(["country_name_short", "election_date"])


def load_cabinets():
    """ParlGov view_cabinet as frozen: one row per cabinet and party, all countries and years."""
    return pd.read_csv(raw_file("parlgov_view_cabinet"), parse_dates=["election_date", "start_date"])


def candidate_cabinets(cabinets):
    """Cabinets since EVENTS_FROM in the 23 countries that started without a new election (same election_id as the
    previous cabinet)."""
    cab = cabinets.drop_duplicates("cabinet_id")
    prev_election = cab["previous_cabinet_id"].map(cab.set_index("cabinet_id")["election_id"])
    return cab[(cab["start_date"] >= EVENTS_FROM) & cab["country_name_short"].isin(TICKERS)
               & (cab["election_id"] == prev_election)]
