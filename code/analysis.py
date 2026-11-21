"""Functions shared by more than one analysis notebook (03 onwards)."""
import numpy as np
import pandas as pd

from helpers import (AR_WIN, DATA, EST_WIN, EVENTS_FROM, GLOBAL_ENTRY_HISTORY, GLOBAL_MAX_STALE_DAYS,
                     INVALID_CLOSES, ISR_SUNDAY_TRADING_END, ISR_TWO_SESSION_WEEKDAY, LONG_GAP_DAYS, MIN_EST_DAYS, MIN_GLOBAL,
                     NORMAL_EXCL, OUTCOME_DAYS, OVERLAP_DROP, OVERLAP_FLAG, P_MAIN, P_SEC, POL_EXCL,
                     PRE_VOL_WIN, RESID_DDOF, SAMPLE_END, STALE_ZERO_RUN, TICKERS, VOL_DDOF, eligible_from)

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
