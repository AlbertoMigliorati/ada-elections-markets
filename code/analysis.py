"""Functions shared by more than one analysis notebook (03 onwards)."""
import numpy as np
import pandas as pd

from helpers import DATA, OVERLAP_DROP, OVERLAP_FLAG, STALE_ZERO_RUN, TICKERS

ANALYSIS_GROUPS = ("election", "crisis", "planned")
CRISIS_LABELS = {"1": "crisis", "0": "planned"}   # is_crisis / new_is_crisis -> group


# ----------------------------------------------------------------------------------------------------
# Trading calendar
# ----------------------------------------------------------------------------------------------------

def valid_closes(prices, country):
    """Closes of the country's index that are valid under the stale-price rule.

    A close equal to the previous observation is a zero log return. Inside a run of at least
    STALE_ZERO_RUN consecutive zero returns, the repeated closes are not valid (removed); the first
    close of the run is kept. A single zero return is kept.
    """
    s = prices[TICKERS[country]].dropna()
    zero = s.eq(s.shift())                          # zero return at t (False for the first close)
    run_id = (~zero).cumsum()                       # a run = one non-zero step followed by zero steps
    run_len = zero.groupby(run_id).transform("sum")
    stale = zero & (run_len >= STALE_ZERO_RUN)
    return s[~stale]


def trading_calendar(prices, country):
    """Dates with a valid close of the country's index (stale-price rule applied), ascending."""
    return valid_closes(prices, country).index


def calendar_position(calendar, dates):
    """Position on `calendar` of the first trading day on or after each date (NaN if there is none)."""
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    pos = np.searchsorted(calendar.to_numpy(), dates, side="left").astype(float)
    pos[(pos >= len(calendar)) | pd.isna(dates)] = np.nan
    return pos


# ----------------------------------------------------------------------------------------------------
# Robustness sets and overlap
# ----------------------------------------------------------------------------------------------------

def load_robustness_sets():
    """data/robustness_sets.csv with every column as text (empty cells are '')."""
    return pd.read_csv(DATA / "robustness_sets.csv", dtype=str, keep_default_na=False)


def overlap_flags(events):
    """Overlap drop and overlap10 flag among the eligible events of each country (any group).

    Day 0 at most OVERLAP_DROP trading days apart: the earlier is kept, the later dropped.
    Kept events more than OVERLAP_DROP and at most OVERLAP_FLAG trading days apart are both flagged.
    Distances are differences of day-0 positions on the country's trading calendar (column day0_pos).
    Two events on the same day 0, or a chain of three events each within OVERLAP_DROP of the next,
    have no fixed rule: the function stops instead of choosing.
    Returns two boolean Series aligned with events (False for events that are not eligible).
    """
    drop = pd.Series(False, index=events.index)
    flag10 = pd.Series(False, index=events.index)
    for country, g in events[events["eligible"]].groupby("country"):
        g = g.sort_values("day0_pos")
        ids = g["event_id"].to_numpy()
        gaps = np.diff(g["day0_pos"].to_numpy())
        tie = np.flatnonzero(gaps == 0)
        if tie.size:
            raise ValueError(f"{country}: events share day 0: {[list(ids[i:i + 2]) for i in tie]}; "
                             "no rule is fixed for which one to keep. Ask before going on.")
        close = gaps <= OVERLAP_DROP
        chain = np.flatnonzero(close[1:] & close[:-1])
        if chain.size:
            raise ValueError(f"{country}: chains of overlapping events: {[list(ids[i:i + 3]) for i in chain]}; "
                             "no rule is fixed for resolving them. Ask before going on.")
        later = np.r_[False, close]
        drop[g.index[later]] = True
        kept = g[~later]
        d = np.diff(kept["day0_pos"].to_numpy())
        near = (d > OVERLAP_DROP) & (d <= OVERLAP_FLAG)
        flag10[kept.index[np.r_[near, False] | np.r_[False, near]]] = True
    return drop, flag10


def apply_robustness_set(events, set_name):
    """Event table of one robustness set ('main' = main labels), analysis groups only.

    Starts from the main rows (variant == 'main') of the events table written by 03. Rows of the set
    with action 'drop' are removed; 'relabel' moves the event to the crisis group (new_is_crisis 1) or
    the planned group (0); a non-empty new_trigger_date swaps in the row of that event whose variant is
    the set name (its day 0 is mapped from the first listed date). Ambiguous rows that the set does not
    relabel are left out. crisis and lr_shift_el follow the new group, and the overlap drop and the
    overlap10 flag are recomputed among the eligible events of the set.
    """
    ev = events[events["variant"] == "main"]
    if set_name != "main":
        rs = load_robustness_sets()
        rs = rs[rs["set"] == set_name]
        if rs.empty:
            raise ValueError(f"unknown robustness set {set_name!r}")
        ev = ev[~ev["event_id"].isin(rs.loc[rs["action"] == "drop", "event_id"])]
        relabel = rs[rs["action"] == "relabel"]
        new_day0 = relabel.loc[relabel["new_trigger_date"] != "", "event_id"]
        alt = events[(events["variant"] == set_name) & events["event_id"].isin(new_day0)]
        if len(alt) != len(new_day0):
            raise ValueError(f"{set_name}: events table lacks the alternative day-0 rows")
        ev = pd.concat([ev[~ev["event_id"].isin(new_day0)], alt])
        ev = ev.assign(group=ev["event_id"].map(relabel.set_index("event_id")["new_is_crisis"]
                                                .map(CRISIS_LABELS)).fillna(ev["group"]))
    ev = ev[ev["group"].isin(ANALYSIS_GROUPS)].reset_index(drop=True)
    ev["crisis"] = (ev["group"] == "crisis").astype(int)
    ev["lr_shift_el"] = ev["lr_shift"].where(ev["group"] == "election", 0.0)
    ev["overlap_drop"], ev["overlap10"] = overlap_flags(ev)
    return ev.sort_values(["country", "day0", "event_id"]).reset_index(drop=True)
