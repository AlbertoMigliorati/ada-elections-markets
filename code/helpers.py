"""Shared settings and functions for the elections project."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
FIGURES = ROOT / "figures"

# Date in the name of the raw files. 
DOWNLOAD_DATE = "2026-11-19"

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

PARLGOV_URL = "https://parlgov.org/data/parlgov-development_csv-utf-8/"
PARLGOV_TABLES = ["view_election", "view_cabinet"]

EVENTS_FROM = "1997-01-01"   # events (elections, cabinets) from this date on
MIN_HISTORY = 270            # trading days of index history needed before day 0


def raw_file(prefix):
    """Path of the frozen raw file, e.g. raw_file('index_prices') -> data/raw/index_prices_<DOWNLOAD_DATE>.csv."""
    path = RAW / f"{prefix}_{DOWNLOAD_DATE}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 01_get_data first.")
    return path


def load_prices():
    """Daily closes of the 23 indices, one column per Yahoo ticker."""
    return pd.read_csv(raw_file("index_prices"), index_col="date", parse_dates=True)


def load_elections():
    """One row per parliamentary election since EVENTS_FROM in the 23 countries."""
    el = pd.read_csv(raw_file("parlgov_view_election"), parse_dates=["election_date"])
    el = el.drop_duplicates("election_id")
    el = el[(el["election_type"] == "parliament")
            & (el["election_date"] >= EVENTS_FROM)
            & el["country_name_short"].isin(TICKERS)]
    return el.sort_values(["country_name_short", "election_date"])


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
