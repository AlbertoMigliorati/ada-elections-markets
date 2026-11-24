"""Functions shared by more than one analysis notebook (03 onwards)."""
import time

import numpy as np
import pandas as pd

from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from helpers import (ALPHA, AR_WIN, B_BOOT, C_GRID, CI_LEVEL, DATA, EST_WIN, EVENTS_FROM, GLOBAL_ENTRY_HISTORY,
                     GLOBAL_MAX_STALE_DAYS, INNER_FOLDS, INVALID_CLOSES, ISR_SUNDAY_TRADING_END,
                     ISR_TWO_SESSION_WEEKDAY, LONG_GAP_DAYS, MIN_EST_DAYS, MIN_GLOBAL, N_GROUP_PERM, NORMAL_EXCL,
                     OUTCOME_DAYS, OVERLAP_DROP, OVERLAP_FLAG, P_MAIN, P_SEC, PLACEBO_MIN_PAIRS, PLACEBO_RADIUS,
                     PLACEBO_RADIUS_WIDE, POL_EXCL, PRE_VOL_WIN, PURGE_DAYS, RESID_DDOF, RF_N_TREES, SAMPLE_END, SEED,
                     STALE_ZERO_RUN, TICKERS, VOL_DDOF, eligible_from)

ANALYSIS_GROUPS = ("election", "crisis", "planned")
CRISIS_LABELS = {"1": "crisis", "0": "planned"}   # is_crisis / new_is_crisis -> group


# ----------------------------------------------------------------------------------------------------
# Trading calendar
# ----------------------------------------------------------------------------------------------------

def in_invalid_span(country, dates):
    """True for dates inside a span of INVALID_CLOSES of the country (both ends included)."""
    dates = pd.DatetimeIndex(dates)
    out = np.zeros(len(dates), dtype=bool)
    for start, end in INVALID_CLOSES.get(country, []):
        out |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return out


def touches_invalid_span(country, start, end):
    """True where the period [start, end] overlaps a span of INVALID_CLOSES of the country."""
    start, end = pd.DatetimeIndex(start), pd.DatetimeIndex(end)
    out = np.zeros(len(start), dtype=bool)
    for s0, s1 in INVALID_CLOSES.get(country, []):
        out |= (start <= pd.Timestamp(s1)) & (end >= pd.Timestamp(s0))
    return out


def valid_closes(prices, country):
    """Closes of the country's index that are valid.

    Closes inside a span of INVALID_CLOSES (data that are not the index) are removed first. Then the
    stale-price rule: a close equal to the previous observation is a zero log return; inside a run of at
    least STALE_ZERO_RUN consecutive zero returns, the repeated closes are not valid (removed); the first
    close of the run is kept. A single zero return is kept.
    """
    s = prices[TICKERS[country]].dropna()
    s = s[~in_invalid_span(country, s.index)]
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


def apply_robustness_set(events, set_name, outcomes=None):
    """Event table of one robustness set ('main' = main labels), analysis groups only.

    With `outcomes` (event_outcomes written by 04), an event is eligible only if it is eligible in both
    03 (events.eligible) and 04 (outcomes.eligible), and the overlap rule is applied to those events.
    Without it, eligibility is that of 03 alone (as written by 03).
    Starts from the main rows (variant == 'main') of the events table written by 03. Rows of the set
    with action 'drop' are removed; 'relabel' moves the event to the crisis group (new_is_crisis 1) or
    the planned group (0); a non-empty new_trigger_date swaps in the row of that event whose variant is
    the set name (its day 0 is mapped from the first listed date). Ambiguous rows that the set does not
    relabel are left out. crisis and lr_shift_el follow the new group, and the overlap drop and the
    overlap10 flag are recomputed among the eligible events of the set.
    """
    if outcomes is not None:
        ok04 = outcomes.set_index(["event_id", "variant"])["eligible"].astype(bool)
        key = pd.MultiIndex.from_frame(events[["event_id", "variant"]])
        if not key.isin(ok04.index).all():
            raise ValueError("event_outcomes lacks rows of the events table")
        events = events.assign(eligible=events["eligible"].astype(bool) & ok04.reindex(key).to_numpy())
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


# ----------------------------------------------------------------------------------------------------
# Returns, global factor and market model
# ----------------------------------------------------------------------------------------------------

def day_class(country, dates):
    """Day class used for thresholds: Israel has 'monday' (a two-session return while TASE traded on
    Sunday) and 'other'; every other country has the single class 'all'."""
    dates = pd.DatetimeIndex(dates)
    if country != "ISR":
        return np.full(len(dates), "all", dtype=object)
    two_sessions = (dates.weekday == ISR_TWO_SESSION_WEEKDAY) & (dates <= pd.Timestamp(ISR_SUNDAY_TRADING_END))
    return np.where(two_sessions, "monday", "other").astype(object)


def returns_table(prices, country):
    """Log returns between consecutive valid closes of the country's index (its trading calendar).

    Columns: r, gap_days (calendar days since the previous valid close), long_gap (gap_days above
    LONG_GAP_DAYS), day_class. The first close has no return.
    """
    p = valid_closes(prices, country)
    gap = pd.Series(p.index, index=p.index).diff().dt.days
    return pd.DataFrame({"r": np.log(p).diff(), "gap_days": gap, "long_gap": gap > LONG_GAP_DAYS,
                         "day_class": day_class(country, p.index)}, index=p.index)


def global_factor(prices):
    """Global factor for every observation of every country: dict country -> DataFrame(g, n_contrib).

    For country c, observation t with previous observation s, index j != c contributes
    ln P(j, t*) - ln P(j, s*), where x* is j's last valid close at or before x, if both s* and t* are on or
    after j's entry date (first date with GLOBAL_ENTRY_HISTORY earlier valid closes) and s* is at most
    GLOBAL_MAX_STALE_DAYS calendar days before s. g is the equal-weighted mean of the contributions, missing
    when fewer than MIN_GLOBAL indices contribute. "Last close at or before" is found with a vectorised
    binary search on each index's calendar (no loop over dates).
    """
    closes = {c: valid_closes(prices, c) for c in TICKERS}
    clean = pd.DataFrame({TICKERS[c]: s for c, s in closes.items()})
    entry = {c: eligible_from(clean, c, GLOBAL_ENTRY_HISTORY) for c in TICKERS}
    max_age = np.timedelta64(GLOBAL_MAX_STALE_DAYS, "D")
    out = {}
    for c in TICKERS:
        dates = closes[c].index.to_numpy()
        t, s = dates[1:], dates[:-1]
        total = np.zeros(len(t))
        count = np.zeros(len(t), dtype=int)
        for j in TICKERS:
            if j == c or pd.isna(entry[j]):
                continue
            cal = closes[j].index.to_numpy()
            logp = np.log(closes[j].to_numpy())
            it = np.searchsorted(cal, t, side="right") - 1
            i_s = np.searchsorted(cal, s, side="right") - 1
            ok = i_s >= 0
            i_s0 = np.maximum(i_s, 0)
            s_star = cal[i_s0]
            ok &= (s_star >= np.datetime64(entry[j])) & (s - s_star <= max_age)
            total += np.where(ok, logp[np.maximum(it, 0)] - logp[i_s0], 0.0)
            count += ok
        g = np.where(count >= MIN_GLOBAL, total / np.maximum(count, 1), np.nan)
        out[c] = pd.DataFrame({"g": np.r_[np.nan, g], "n_contrib": np.r_[0, count]}, index=closes[c].index)
    return out


def political_distance(calendar, trading_days):
    """Distance in trading days from each day of `calendar` to the nearest political date (inf if none)."""
    pos = np.unique(calendar.get_indexer(pd.DatetimeIndex(pd.Series(trading_days).dropna())))
    if (pos < 0).any():
        raise ValueError("a political trading day is not on the trading calendar")
    idx = np.arange(len(calendar))
    if len(pos) == 0:
        return np.full(len(calendar), np.inf)
    k = np.searchsorted(pos, idx)
    left = np.where(k > 0, idx - pos[np.maximum(k - 1, 0)], np.inf)
    right = np.where(k < len(pos), pos[np.minimum(k, len(pos) - 1)] - idx, np.inf)
    return np.minimum(left, right)


def build_panels(prices, political_dates):
    """One table per country on its trading calendar: r, gap_days, long_gap, day_class, g, n_contrib,
    g_lag (factor of the previous observation), pol_dist (trading days to the nearest political date of
    that country) and est_ok (the day may enter an estimation window: r, g and g_lag present, not a
    long-gap return, not the day after a long-gap return (its lagged factor spans the whole gap), more than
    POL_EXCL trading days from any political date)."""
    gf = global_factor(prices)
    panels = {}
    for c in TICKERS:
        p = returns_table(prices, c).join(gf[c])
        p["g_lag"] = p["g"].shift(1)
        p["pol_dist"] = political_distance(p.index, political_dates.loc[political_dates["country"] == c,
                                                                        "trading_day"])
        after_gap = p["long_gap"].shift(1, fill_value=False).astype(bool)
        p["est_ok"] = (p[["r", "g", "g_lag"]].notna().all(axis=1) & ~p["long_gap"] & ~after_gap
                       & (p["pol_dist"] > POL_EXCL))
        panels[c] = p
    return panels


def market_model(panel):
    """Market model r = a + b0*g + b1*g_lag fitted by OLS for a window placed around every day.

    For target position T the window is positions T+EST_WIN[0] .. T+EST_WIN[1] of the country's calendar,
    restricted to days with est_ok. All windows are fitted at once from cumulative sums of the masked
    cross-products (closed-form 3x3 normal equations). A fit needs at least MIN_EST_DAYS valid days.
    Returns, per target day: n_est, a, b0, b1, s (residual sd, RESID_DDOF degrees of freedom removed),
    r2 and min_contrib (fewest global-factor contributors among the valid estimation days).
    """
    ok = panel["est_ok"].to_numpy()
    r = np.where(ok, panel["r"].to_numpy(), 0.0)
    g0 = np.where(ok, panel["g"].to_numpy(), 0.0)
    g1 = np.where(ok, panel["g_lag"].to_numpy(), 0.0)
    terms = {"n": ok.astype(float), "g0": g0, "g1": g1, "r": r, "g0g0": g0 * g0, "g0g1": g0 * g1,
             "g1g1": g1 * g1, "g0r": g0 * r, "g1r": g1 * r, "rr": r * r}
    size = len(panel)
    lo, hi = EST_WIN
    target = np.arange(size)
    start, stop = np.clip(target + lo, 0, size), np.clip(target + hi + 1, 0, size)
    w = {}
    for k, v in terms.items():
        cum = np.r_[0.0, np.cumsum(v)]
        w[k] = cum[stop] - cum[start]
    n = np.rint(w["n"]).astype(int)
    fit = n >= MIN_EST_DAYS
    xtx = np.stack([np.stack([w["n"], w["g0"], w["g1"]]),
                    np.stack([w["g0"], w["g0g0"], w["g0g1"]]),
                    np.stack([w["g1"], w["g0g1"], w["g1g1"]])]).transpose(2, 0, 1)[fit]
    xty = np.stack([w["r"], w["g0r"], w["g1r"]], axis=1)[fit]
    beta = np.linalg.solve(xtx, xty[:, :, None])[:, :, 0]
    ssr = np.maximum(w["rr"][fit] - (beta * xty).sum(axis=1), 0.0)
    sst = w["rr"][fit] - w["r"][fit] ** 2 / n[fit]
    out = pd.DataFrame(np.nan, index=panel.index, columns=["a", "b0", "b1", "s", "r2", "min_contrib"])
    out.loc[fit, ["a", "b0", "b1"]] = beta
    out.loc[fit, "s"] = np.sqrt(ssr / (n[fit] - RESID_DDOF))
    out.loc[fit, "r2"] = 1 - ssr / sst
    contrib = pd.Series(np.where(ok, panel["n_contrib"].to_numpy(dtype=float), np.inf))
    window_min = contrib.rolling(hi - lo + 1, min_periods=1).min().shift(-hi).to_numpy()
    out.loc[fit, "min_contrib"] = window_min[fit]
    out.insert(0, "n_est", n)
    return out


def predicted_return(fit, g, g_lag):
    """Normal return a + b0*g + b1*g_lag (fit: one fitted window, or one per day aligned with g)."""
    return fit["a"] + fit["b0"] * g + fit["b1"] * g_lag


def event_abnormal_returns(panel, fits, target, rel_days):
    """AR on days target+rel_days with the window placed around target (NaN outside the calendar)."""
    pos = target + np.asarray(rel_days)
    inside = (pos >= 0) & (pos < len(panel))
    ar = np.full(len(pos), np.nan)
    p = panel.iloc[pos[inside]]
    ar[inside] = (p["r"] - predicted_return(fits.iloc[target], p["g"], p["g_lag"])).to_numpy()
    return ar


def normal_days(panels, fits):
    """One-day AR for every trading day from EVENTS_FROM to SAMPLE_END (window placed around that day).

    normal = valid AR, not a long-gap return, more than NORMAL_EXCL trading days from any political date.
    """
    rows = []
    for c, p in panels.items():
        ar = p["r"] - predicted_return(fits[c], p["g"], p["g_lag"])
        normal = ar.notna() & ~p["long_gap"] & (p["pol_dist"] > NORMAL_EXCL)
        keep = (p.index >= EVENTS_FROM) & (p.index <= SAMPLE_END)
        rows.append(pd.DataFrame({"country": c, "date": p.index, "day_class": p["day_class"].to_numpy(),
                                  "ar": ar.to_numpy(), "normal": normal.to_numpy()})[keep])
    return pd.concat(rows, ignore_index=True)


QUANTILES = {f"q{P_MAIN * 100:.0f}": P_MAIN, f"q{P_SEC * 100:.0f}": P_SEC}


def thresholds(normal):
    """q(country, p, day class): p-quantile (numpy default method) of |AR| over the normal days."""
    nd = normal[normal["normal"]]
    rows = []
    for (c, cls), g in nd.groupby(["country", "day_class"]):
        absar = np.abs(g["ar"].to_numpy())
        rows.append({"country": c, "day_class": cls, "n_normal": len(g),
                     **{name: np.quantile(absar, p) for name, p in QUANTILES.items()}})
    return pd.DataFrame(rows)


def event_outcomes(events, panels, fits, normal, thr):
    """Outcomes of every row of the events table (main and alternative day-0 rows).

    An event is eligible if it is eligible in 03 and, here: the window has at least MIN_EST_DAYS valid days,
    day +1 exists, the global factor is defined on day 0 and day +1, and AR(0) and AR(+1) can be computed.
    The first failing condition is the drop reason. For eligible events: AR and SAR on AR_WIN, the strong
    flags (max |AR| over OUTCOME_DAYS above the threshold of each day's own class) and vol_pre.
    Returns (outcomes, event_ar, pre_missing): pre_missing lists eligible events with a missing AR inside
    PRE_VOL_WIN (vol_pre is left missing for them).
    """
    sd_c = normal[normal["normal"]].groupby("country")["ar"].std(ddof=VOL_DDOF)
    q = thr.set_index(["country", "day_class"])
    rel = np.arange(AR_WIN[0], AR_WIN[1] + 1)
    pre = (rel >= PRE_VOL_WIN[0]) & (rel <= PRE_VOL_WIN[1])
    out, paths, pre_missing = [], [], []
    for e in events.itertuples(index=False):
        p, f = panels[e.country], fits[e.country]
        t0 = int(e.day0_pos)
        if p.index[t0] != e.day0:
            raise ValueError(f"{e.event_id}: day0_pos does not point to day0 on the trading calendar")
        fe = f.iloc[t0]
        days = [t0 + k for k in OUTCOME_DAYS]
        row = {"event_id": e.event_id, "variant": e.variant, "country": e.country,
               "s": fe["s"], "b0": fe["b0"], "b1": fe["b1"], "r2": fe["r2"], "n_est_days": fe["n_est"],
               "min_contributors": fe["min_contrib"],
               "long_gap": bool(any(d < len(p) and p["long_gap"].iloc[d] for d in days))}
        ar = event_abnormal_returns(p, f, t0, rel)
        ar_days = {k: ar[rel == k][0] for k in OUTCOME_DAYS}
        if not e.eligible:
            reason = f"not eligible in 03: {e.ineligible_reason}"
        elif fe["n_est"] < MIN_EST_DAYS:
            reason = f"fewer than {MIN_EST_DAYS} valid estimation days"
        elif max(days) >= len(p):
            reason = "no day +1"
        elif p["g"].iloc[days].isna().any():
            reason = "global factor missing on day 0 or day +1"
        elif any(np.isnan(v) for v in ar_days.values()):
            reason = "AR(0) or AR(+1) not computable"
        else:
            reason = ""
        row["eligible"] = reason == ""
        row["drop_reason"] = reason
        if row["eligible"]:
            for k, v in ar_days.items():
                row[f"ar{k}"], row[f"sar{k}"] = v, v / fe["s"]
            for name in QUANTILES:
                row[f"strong{name[1:]}"] = int(any(abs(ar_days[k]) > q.loc[(e.country, p["day_class"].iloc[d]),
                                                                            name]
                                                   for k, d in zip(OUTCOME_DAYS, days)))
            if np.isnan(ar[pre]).any():
                pre_missing.append({"event_id": e.event_id, "variant": e.variant,
                                    "valid_pre_days": int((~np.isnan(ar[pre])).sum())})
                row["vol_pre"] = np.nan
            else:
                row["vol_pre"] = np.log(np.std(ar[pre], ddof=VOL_DDOF) / sd_c[e.country])
            paths.append(pd.DataFrame({"event_id": e.event_id, "variant": e.variant, "rel_day": rel,
                                       "ar": ar, "sar": ar / fe["s"]}))
        out.append(row)
    cols = (["event_id", "variant", "country"] + [f"{x}{k}" for x in ("ar", "sar") for k in OUTCOME_DAYS]
            + [f"strong{name[1:]}" for name in QUANTILES]
            + ["vol_pre", "s", "b0", "b1", "r2", "n_est_days", "min_contributors", "long_gap", "eligible",
               "drop_reason"])
    outcomes = pd.DataFrame(out).reindex(columns=cols)
    for name in QUANTILES:
        outcomes[f"strong{name[1:]}"] = outcomes[f"strong{name[1:]}"].astype("Int64")
    return outcomes, pd.concat(paths, ignore_index=True), pd.DataFrame(pre_missing)


# ----------------------------------------------------------------------------------------------------
# Matched placebo, tests, bootstrap and placebo paths (05 onwards)
# ----------------------------------------------------------------------------------------------------

def normal_pairs(normal, thr):
    """Pairs (d, d+1) of consecutive trading days of a country that are both normal days.

    `normal` is normal_days.csv (every trading day from EVENTS_FROM to SAMPLE_END, so consecutive rows of a
    country are consecutive trading days). One row per pair: country, date (d), date1 (d+1), day_class (of d)
    and, for each threshold name of QUANTILES, the flags above_<name>_0 and above_<name>_1: |AR| of d and of
    d+1 above the threshold of that day's own class.
    """
    q = thr.set_index(["country", "day_class"])
    nd = normal.sort_values(["country", "date"]).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(nd[["country", "day_class"]])
    nxt = nd.groupby("country")[["date", "normal"]].shift(-1)
    out = pd.DataFrame({"country": nd["country"], "date": nd["date"], "date1": nxt["date"],
                        "day_class": nd["day_class"]})
    for name in QUANTILES:
        above = pd.Series(nd["ar"].abs().to_numpy() > q[name].reindex(key).to_numpy())
        out[f"above_{name}_0"] = above.to_numpy()
        out[f"above_{name}_1"] = above.groupby(nd["country"]).shift(-1).eq(True).to_numpy()
    pair = nd["normal"].astype(bool).to_numpy() & nxt["normal"].eq(True).to_numpy()
    return out[pair].reset_index(drop=True)


def placebo_pool(pairs, country, day0, day_class0):
    """Matched placebo pool of one event: the pairs of its country whose first day d has the day class of
    day 0 and lies within PLACEBO_RADIUS calendar days of day 0; if that gives fewer than PLACEBO_MIN_PAIRS
    pairs, the radius is PLACEBO_RADIUS_WIDE (once, never more). Returns (pool, radius)."""
    cand = pairs[(pairs["country"] == country) & (pairs["day_class"] == day_class0)]
    dist = (cand["date"] - pd.Timestamp(day0)).abs().dt.days
    radius = PLACEBO_RADIUS if (dist <= PLACEBO_RADIUS).sum() >= PLACEBO_MIN_PAIRS else PLACEBO_RADIUS_WIDE
    return cand[dist <= radius], radius


def placebo_rate(pool, name, days):
    """pi: share of the pool's pairs with |AR| above threshold `name` on at least one of `days` (0 = d, 1 = d+1)."""
    hit = np.zeros(len(pool), dtype=bool)
    for k in days:
        hit |= pool[f"above_{name}_{k}"].to_numpy()
    return hit.mean() if len(pool) else np.nan


def event_strong(ev, thr, name, days):
    """Strong flag of events at threshold `name` over `days`: |AR(k)| above the threshold of day k's own class.
    ev needs columns country, ar<k> and class<k> for every k in days."""
    q = thr.set_index(["country", "day_class"])[name]
    hit = np.zeros(len(ev), dtype=bool)
    for k in days:
        qk = q.reindex(pd.MultiIndex.from_arrays([ev["country"], ev[f"class{k}"]])).to_numpy()
        hit |= ev[f"ar{k}"].abs().to_numpy() > qk
    return hit.astype(int)


def poisson_binomial_pmf(probs):
    """Exact distribution of a sum of independent Bernoulli(p_i), by convolution."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def poisson_binomial_upper(k, probs):
    """P(S >= k) for S a sum of independent Bernoulli(probs)."""
    return float(poisson_binomial_pmf(probs)[int(k):].sum())


def cluster_bootstrap(countries, values, in_a, in_b=None):
    """Country-cluster bootstrap of mean(values over group a), minus mean(values over group b) if given.

    The countries with at least one event of the tested groups are resampled with replacement, B_BOOT times,
    with a numpy Generator seeded with SEED. A replicate in which a tested group is empty has no value.
    Returns (replicates, number of replicates without a value).
    """
    countries = np.asarray(countries)
    in_a = np.asarray(in_a, dtype=bool)
    in_b = np.zeros(len(in_a), dtype=bool) if in_b is None else np.asarray(in_b, dtype=bool)
    keep = in_a | in_b
    ids, inv = np.unique(countries[keep], return_inverse=True)
    v, a, b = np.asarray(values, dtype=float)[keep], in_a[keep], in_b[keep]
    sa, na = np.bincount(inv, v * a, len(ids)), np.bincount(inv, a, len(ids))
    sb, nb = np.bincount(inv, v * b, len(ids)), np.bincount(inv, b, len(ids))
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B_BOOT)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (w @ sa) / (w @ na)
        if b.any():
            stat = stat - (w @ sb) / (w @ nb)
    stat[~np.isfinite(stat)] = np.nan
    return stat, int(np.isnan(stat).sum())


def percentile_interval(replicates):
    """Percentile interval at CI_LEVEL of the bootstrap replicates (missing replicates left out)."""
    tail = 100 * (1 - CI_LEVEL) / 2
    return tuple(np.nanpercentile(replicates, [tail, 100 - tail]))


def group_permutation_test(countries, values, is_a):
    """Two-sided test of mean(values | a) - mean(values | not a), permuting the group label within country.

    N_GROUP_PERM permutations with a numpy Generator seeded with SEED; p = (1 + #{|T*| >= |T|}) / (1 + N).
    Returns (observed statistic, p-value).
    """
    countries = np.asarray(countries)
    values = np.asarray(values, dtype=float)
    is_a = np.asarray(is_a, dtype=bool)
    n_a, n_b = is_a.sum(), (~is_a).sum()
    observed = values[is_a].mean() - values[~is_a].mean()
    rng = np.random.default_rng(SEED)
    labels = np.empty((N_GROUP_PERM, len(values)), dtype=bool)
    for c in np.unique(countries):
        idx = np.flatnonzero(countries == c)
        labels[:, idx] = rng.permuted(np.tile(is_a[idx], (N_GROUP_PERM, 1)), axis=1)
    perm = labels @ values / n_a - (~labels) @ values / n_b
    p = (1 + (np.abs(perm) >= np.abs(observed) - 1e-12).sum()) / (1 + N_GROUP_PERM)
    return observed, p


def p1_power(probs, rates):
    """Power of the one-sided exact test of P1 at ALPHA: critical count k = smallest k with P(S >= k) <= ALPHA
    under the Poisson-binomial null with probabilities `probs`; power = P(Binomial(N, rate) >= k).
    Returns a DataFrame with rate, N, critical count, size of the test and power."""
    pmf = poisson_binomial_pmf(probs)
    upper = pmf[::-1].cumsum()[::-1]                     # upper[k] = P(S >= k)
    k = int(np.flatnonzero(upper <= ALPHA)[0])
    n = len(probs)
    return pd.DataFrame({"rate": list(rates), "n_events": n, "critical_count": k, "size": upper[k],
                         "power": [binom.sf(k - 1, n, r) for r in rates]})


def ar_paths(panel, fits, positions, rel_days):
    """AR paths of the dates at calendar positions `positions`: AR on position + rel_days with the market model
    fitted on the window placed around that date. The same construction as an event's AR path (rows: dates,
    columns: rel_days; NaN outside the calendar or where an input is missing)."""
    pos = np.asarray(positions)[:, None] + np.asarray(rel_days)[None, :]
    inside = (pos >= 0) & (pos < len(panel))
    p = np.clip(pos, 0, len(panel) - 1)
    r, g, gl = (panel[c].to_numpy(dtype=float) for c in ("r", "g", "g_lag"))
    f = fits.iloc[np.asarray(positions)]
    ar = r[p] - (f["a"].to_numpy()[:, None] + f["b0"].to_numpy()[:, None] * g[p] + f["b1"].to_numpy()[:, None] * gl[p])
    ar[~inside] = np.nan
    return ar


def sar_paths(panel, fits, positions, rel_days):
    """SAR paths: ar_paths divided by the residual sd of the window placed around each date."""
    return ar_paths(panel, fits, positions, rel_days) / fits.iloc[np.asarray(positions)]["s"].to_numpy()[:, None]


def pre_volatility(panel, fits, positions, sd_c):
    """vol_pre of the dates at calendar positions `positions`, built as an event's vol_pre: ln of the sd
    (ddof VOL_DDOF) of AR over PRE_VOL_WIN, window placed around the date, over the country's normal-day sd
    sd_c. NaN if an AR of the window is missing."""
    rel = np.arange(PRE_VOL_WIN[0], PRE_VOL_WIN[1] + 1)
    ar = ar_paths(panel, fits, positions, rel)
    return np.log(np.std(ar, axis=1, ddof=VOL_DDOF) / sd_c)


def draw_placebo_paths(pool_paths, rng):
    """One placebo path per event: a date drawn at random among the event's pool dates whose path has no
    missing value (equivalent to redrawing until a complete path is found). `pool_paths` is a list of arrays
    (pool dates x days), one per event."""
    out = []
    for paths in pool_paths:
        valid = np.flatnonzero(np.isfinite(paths).all(axis=1))
        if not len(valid):
            raise ValueError("an event has no placebo date with a complete path")
        out.append(paths[rng.choice(valid)])
    return np.vstack(out)


# ----------------------------------------------------------------------------------------------------
# Prediction: nested leave-one-country-out validation (06)
# ----------------------------------------------------------------------------------------------------

MODEL_FEATURES = {"M1": ["vol_pre"],
                  "M2": ["vol_pre", "crisis"],
                  "M3": ["vol_pre", "crisis", "enp", "coalition", "minority", "lr_shift_el"]}
POLITICAL_BLOCK = ["enp", "coalition", "minority", "lr_shift_el"]   # permuted jointly for P3


def outer_folds(countries, day0):
    """Leave-one-country-out folds with the purge: training events whose day 0 is within PURGE_DAYS calendar
    days of any event of the test country are removed from that fold's training set.
    Returns a list of dicts: country, train and test (positions), n_purged, min_gap (smallest |day 0 difference|
    in days between a training and a test event)."""
    countries = np.asarray(countries)
    days = pd.to_datetime(pd.Series(day0)).to_numpy().astype("datetime64[D]").astype(np.int64)
    folds = []
    for c in np.unique(countries):
        test = np.flatnonzero(countries == c)
        other = np.flatnonzero(countries != c)
        gap = np.abs(days[other][:, None] - days[test][None, :]).min(axis=1)
        train = other[gap > PURGE_DAYS]
        folds.append({"country": c, "train": train, "test": test, "n_purged": len(other) - len(train),
                      "min_gap": int(np.abs(days[train][:, None] - days[test][None, :]).min())})
    return folds


def logistic_model(C, penalty="l2"):
    """Pipeline(StandardScaler, LogisticRegression): ridge (l2, lbfgs) or lasso (l1, liblinear), no class weights.
    The penalty is set through l1_ratio (scikit-learn >= 1.8 deprecates the `penalty` argument)."""
    if penalty == "l2":
        logit = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=10_000, random_state=SEED)
    elif penalty == "l1":
        logit = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=10_000, random_state=SEED)
    else:
        raise ValueError(penalty)
    return Pipeline([("scale", StandardScaler()), ("logit", logit)])


def forest_model(min_leaf):
    """Random forest of method D: RF_N_TREES trees, max_features 'sqrt', random_state SEED."""
    return RandomForestClassifier(n_estimators=RF_N_TREES, max_features="sqrt", min_samples_leaf=min_leaf,
                                  random_state=SEED, n_jobs=1)


def inner_choice(X, y, groups, make, grid):
    """Hyperparameter chosen on a training set: GroupKFold(INNER_FOLDS) by country, criterion mean log-loss over
    the inner folds; on a tie the first value of `grid` wins. Returns (chosen value, mean log-loss per value)."""
    splits = list(GroupKFold(n_splits=INNER_FOLDS).split(X, y, groups))
    loss = []
    for v in grid:
        fold_loss = [log_loss(y[va], make(v).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1], labels=[0, 1])
                     for tr, va in splits]
        loss.append(np.mean(fold_loss))
    return grid[int(np.argmin(loss))], np.array(loss)


def nested_predictions(X, y, groups, folds, kind="l2"):
    """Out-of-fold probabilities of a strong reaction.

    kind 'M0': the training base rate. kind 'l2' / 'l1': logistic Pipeline, C chosen by inner_choice over
    C_GRID on the training set of each outer fold, then refitted on that training set. kind 'rf': random forest,
    min_samples_leaf chosen the same way over `RF_MIN_LEAF_GRID` (passed in as kind ('rf', grid)).
    Returns (predictions aligned with y, list with one dict per fold: country, chosen value, nonzero coefficients
    for the logistic models)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    info = []
    for f in folds:
        tr, te = f["train"], f["test"]
        if kind == "M0":
            pred[te] = y[tr].mean()
            info.append({"country": f["country"], "value": np.nan})
            continue
        if isinstance(kind, tuple):
            grid, make = kind[1], forest_model
        else:
            grid, make = C_GRID, (lambda C, k=kind: logistic_model(C, k))
        value, _ = inner_choice(X[tr], y[tr], groups[tr], make, grid)
        m = make(value).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
        row = {"country": f["country"], "value": value}
        if not isinstance(kind, tuple):
            scaler = m.named_steps["scale"]
            sd = X[tr].std(axis=0)
            if not (np.allclose(scaler.mean_, X[tr].mean(axis=0)) and np.allclose(scaler.scale_, np.where(sd == 0, 1, sd))):
                raise AssertionError("scaler not fitted on the training set of the fold")
            row["nonzero"] = m.named_steps["logit"].coef_[0] != 0
        info.append(row)
    return pred, info


def permute_block(block, cells, rng):
    """Rows of `block` permuted jointly within each cell (the whole row moves), cells in sorted order."""
    out = block.copy()
    for cell in np.unique(cells):
        idx = np.flatnonzero(cells == cell)
        out[idx] = block[idx[rng.permutation(len(idx))]]
    return out


def event_losses(y, p):
    """Per-event log-loss and squared error (Brier) of probabilities p. For the log-loss p is clipped to
    [eps, 1 - eps] (float eps), as scikit-learn's log_loss does; it matters only for the random forest."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), (p - y) ** 2


def point_metrics(y, p):
    """Pooled log-loss, AUC and Brier score of out-of-fold predictions (scikit-learn)."""
    return {"logloss": log_loss(y, p, labels=[0, 1]), "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)}


def bootstrap_metrics(countries, y, preds, deltas):
    """Country-cluster bootstrap of the pooled out-of-fold metrics, without refitting.

    Countries are resampled with replacement B_BOOT times (numpy Generator seeded with SEED; the same draws as
    cluster_bootstrap): each event is weighted by the number of times its country is drawn. For every model in
    `preds` (dict name -> probabilities): weighted log-loss, Brier and AUC (Mann-Whitney with weights; ties
    count one half); for every pair (a, b) in `deltas`: log-loss(a) - log-loss(b) on the same draws. A replicate
    without a positive or without a negative has no AUC. Returns dict (name, metric) -> replicates.
    """
    countries = np.asarray(countries)
    y = np.asarray(y, dtype=int)
    ids, inv = np.unique(countries, return_inverse=True)
    rng = np.random.default_rng(SEED)
    w = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B_BOOT)[:, inv].astype(float)
    wsum = w.sum(axis=1)
    pos, neg = y == 1, y == 0
    out = {}
    for name, p in preds.items():
        ll, sq = event_losses(y, p)
        out[(name, "logloss")] = w @ ll / wsum
        out[(name, "brier")] = w @ sq / wsum
        p = np.asarray(p, dtype=float)
        k = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
        with np.errstate(invalid="ignore", divide="ignore"):
            auc = np.einsum("bi,ij,bj->b", w[:, pos], k, w[:, neg]) / (w[:, pos].sum(axis=1) * w[:, neg].sum(axis=1))
        auc[~np.isfinite(auc)] = np.nan
        out[(name, "auc")] = auc
    for a, b in deltas:
        out[(f"{a}-{b}", "delta_logloss")] = out[(a, "logloss")] - out[(b, "logloss")]
    return out


# Specifications of 06: samples, folds, model fits and the P3 permutation test

STRONG = {q: "strong" + q[1:] for q in QUANTILES}   # outcome column of each threshold: strong95, strong90


def prediction_sample(events, outcomes, set_name):
    """Eligible elections and crises of one set after the overlap rule (planned changes left out), with the
    outcomes and vol_pre of 04."""
    t = apply_robustness_set(events, set_name, outcomes)
    t = t[t["eligible"] & ~t["overlap_drop"] & t["group"].isin(["crisis", "election"])]
    t = t.merge(outcomes[["event_id", "variant", *STRONG.values(), "vol_pre"]], on=["event_id", "variant"],
                validate="one_to_one")
    return t.reset_index(drop=True)


def prediction_specs(sets):
    """(variant, set, outcome column, with Israel) of the main specification and of the secondary ones:
    S2 threshold P_SEC, S4 without Israel, S5 every robustness set."""
    y_main, y_sec = STRONG["q95"], STRONG["q90"]
    return ([("main", "main", y_main, True), ("q90", "main", y_sec, True), ("no_israel", "main", y_main, False)]
            + [("main", s, y_main, True) for s in sets if s != "main"])


def prepare_spec(sample, ycol, israel, verbose=False):
    """Arrays and outer folds of one specification; checks the folds and prints positives per fold."""
    t = sample if israel else sample[sample["country"] != "ISR"].reset_index(drop=True)
    d = {"t": t, "y": t[ycol].astype(int).to_numpy(), "countries": t["country"].to_numpy(),
         "cells": (t["country"] + "|" + t["group"]).to_numpy(),
         "X": {m: t[f].to_numpy(dtype=float) for m, f in MODEL_FEATURES.items()}}
    d["folds"] = outer_folds(d["countries"], t["day0"])
    check_folds(d, verbose)
    return d


def check_folds(d, verbose=False):
    """No test-country event in training; after the purge every training event is more than PURGE_DAYS calendar
    days from every test event. Prints positives per fold (all of them if verbose) and the folds with none."""
    days = pd.to_datetime(d["t"]["day0"]).to_numpy()
    for f in d["folds"]:
        tr, te = f["train"], f["test"]
        assert not np.isin(d["countries"][tr], d["countries"][te]).any(), "test-country event in training"
        gap = np.abs((days[tr][:, None] - days[te][None, :]) / np.timedelta64(1, "D")).min()
        assert gap > PURGE_DAYS and f["min_gap"] > PURGE_DAYS, f["country"]
        assert len(np.intersect1d(tr, te)) == 0
    pos = pd.DataFrame({"country": [f["country"] for f in d["folds"]],
                        "test_events": [len(f["test"]) for f in d["folds"]],
                        "test_positives": [int(d["y"][f["test"]].sum()) for f in d["folds"]],
                        "train_events": [len(f["train"]) for f in d["folds"]],
                        "train_positives": [int(d["y"][f["train"]].sum()) for f in d["folds"]],
                        "purged": [f["n_purged"] for f in d["folds"]]})
    if verbose:
        print(pos.to_string(index=False))
    zero = pos.loc[pos["test_positives"] == 0, "country"].tolist()
    print(f"{len(d['folds'])} folds, {int(d['y'].sum())} positives of {len(d['y'])}; folds with zero positives "
          f"in the test country ({len(zero)}, still scored in the pooled metrics): {zero}")


def fit_models(d, models=("M0", "M1", "M2", "M3")):
    """Out-of-fold predictions of the models (M0: training base rate; M1-M3: nested ridge), the C chosen in each
    fold and the time of each nested run (d['seconds_M3'] is the work repeated by each permutation of P3)."""
    preds, chosen = {}, {}
    for m in models:
        start = time.perf_counter()
        if m == "M0":
            preds[m], _ = nested_predictions(d["X"]["M1"], d["y"], d["countries"], d["folds"], "M0")
        else:
            preds[m], info = nested_predictions(d["X"][m], d["y"], d["countries"], d["folds"], "l2")
            chosen[m] = [i["value"] for i in info]
        d[f"seconds_{m}"] = time.perf_counter() - start
    for m, p in preds.items():
        assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), f"{m}: prediction outside (0, 1)"
    if "M0" in preds:
        check_m0(d, preds["M0"])
    d["preds"], d["chosen"] = preds, chosen
    return d


def check_m0(d, p0):
    """In each fold M0 predicts the training base rate b_k, so its pooled log-loss is exactly the entropy of the
    overall base rate plus, for every test country, the extra loss of predicting b_k instead of the overall rate.
    That extra loss is large when strong reactions are concentrated in a few countries; it is printed."""
    y, base = d["y"], d["y"].mean()
    entropy = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    extra = {}
    for f in d["folds"]:
        b, yt = y[f["train"]].mean(), y[f["test"]]
        assert (p0[f["test"]] == b).all(), f"M0 is not the training base rate in fold {f['country']}"
        extra[f["country"]] = (event_losses(yt, np.full(len(yt), b))[0].sum()
                               - event_losses(yt, np.full(len(yt), base))[0].sum()) / len(y)
    m0 = point_metrics(y, p0)["logloss"]
    assert abs(m0 - entropy - sum(extra.values())) < 1e-12, (m0, entropy, sum(extra.values()))
    top = pd.Series(extra).sort_values(ascending=False).head(3).round(4).to_dict()
    print(f"M0 log-loss {m0:.4f} = base-rate entropy {entropy:.4f} + {m0 - entropy:.4f} from differences in the "
          f"training base rate across folds (largest: {top})")


def p3_test(d, n_perm):
    """P3 on one specification: observed Δ = log-loss(M1) - log-loss(M3), and its null from n_perm permutations of
    the political block within country x group cells (numpy Generator seeded with SEED; outcome, vol_pre and
    crisis kept; M1 does not use the block, so its predictions are the same in every permutation and are not
    recomputed). One-sided p = (1 + #{Δ* >= Δ}) / (n_perm + 1). Adds delta, null, p3 and n_perm to d."""
    y = d["y"]
    observed = point_metrics(y, d["preds"]["M1"])["logloss"] - point_metrics(y, d["preds"]["M3"])["logloss"]
    rng = np.random.default_rng(SEED)
    cols = [MODEL_FEATURES["M3"].index(f) for f in POLITICAL_BLOCK]
    ll1 = event_losses(y, d["preds"]["M1"])[0].mean()
    null = np.empty(n_perm)
    start = time.perf_counter()
    for i in range(n_perm):
        X = d["X"]["M3"].copy()
        X[:, cols] = permute_block(X[:, cols], d["cells"], rng)
        p3, _ = nested_predictions(X, y, d["countries"], d["folds"], "l2")
        null[i] = ll1 - event_losses(y, p3)[0].mean()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_perm} permutations, {(time.perf_counter() - start) / 60:.1f} min")
    d.update(delta=observed, null=null, p3=(1 + (null >= observed - 1e-12).sum()) / (n_perm + 1), n_perm=n_perm)
    return d
