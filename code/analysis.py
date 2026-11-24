"""Functions shared by more than one analysis notebook (02 onwards)."""
import numpy as np
import pandas as pd

from helpers import (AR_WIN, B_BOOT, CI_LEVEL, DATA, EST_WIN, EVENTS_FROM, GLOBAL_ENTRY_HISTORY, GLOBAL_MAX_STALE_DAYS,
                     INVALID_CLOSES, ISR_SUNDAY_TRADING_END, ISR_TWO_SESSION_WEEKDAY, LONG_GAP_DAYS, MIN_EST_DAYS,
                     MIN_GLOBAL, N_GROUP_PERM, NORMAL_EXCL, OUTCOME_DAYS, OVERLAP_DROP, P_MAIN, P_SEC,
                     PLACEBO_MIN_PAIRS, PLACEBO_RADIUS, POL_EXCL, PRE_VOL_WIN, PURGE_DAYS, SAMPLE_END, SEED,
                     STALE_ZERO_RUN, TICKERS, VOL_DDOF)

ANALYSIS_GROUPS = ("election", "crisis", "planned")
CRISIS_LABELS = {"1": "crisis", "0": "planned"}            # is_crisis / new_is_crisis -> group
QUANTILES = {f"q{P_MAIN * 100:.0f}": P_MAIN, f"q{P_SEC * 100:.0f}": P_SEC}
FACTORS = ("g", "g_lag", "w", "w_lag")                     # regressors of the market model, besides the constant


# ----------------------------------------------------------------------------------------------------
# Trading calendar, robustness sets and overlap (02)
# ----------------------------------------------------------------------------------------------------

def in_invalid_span(country, start, end=None):
    """True where the date (or the period [start, end]) touches a span of INVALID_CLOSES of the country."""
    start = pd.DatetimeIndex(start)
    end = start if end is None else pd.DatetimeIndex(end)
    out = np.zeros(len(start), dtype=bool)
    for s0, s1 in INVALID_CLOSES.get(country, []):
        out |= (start <= pd.Timestamp(s1)) & (end >= pd.Timestamp(s0))
    return out


def valid_closes(prices, country):
    """Closes of the index without INVALID_CLOSES and without the repeated closes of a run of at least STALE_ZERO_RUN
    zero returns (the first close of the run and a single zero return are kept). Its dates are the trading calendar."""
    s = prices[TICKERS[country]].dropna()
    s = s[~in_invalid_span(country, s.index)]
    zero = s.eq(s.shift())
    return s[~(zero & zero.groupby((~zero).cumsum()).transform("sum").ge(STALE_ZERO_RUN))]


def calendar_position(calendar, dates):
    """Position on `calendar` of the first trading day on or after each date (NaN if none)."""
    dates = pd.to_datetime(pd.Series(dates)).to_numpy()
    pos = np.searchsorted(calendar.to_numpy(), dates).astype(float)
    pos[(pos >= len(calendar)) | pd.isna(dates)] = np.nan
    return pos


def load_robustness_sets():
    return pd.read_csv(DATA / "robustness_sets.csv", dtype=str, keep_default_na=False)


def overlap_flags(events):
    """Among the eligible events of each country, the later of two day 0s at most OVERLAP_DROP trading days apart is
    dropped. Ties and chains of three have no rule: the function stops."""
    drop = pd.Series(False, index=events.index)
    for country, g in events[events["eligible"]].sort_values("day0_pos").groupby("country"):
        gap = np.diff(g["day0_pos"].to_numpy())
        close = gap <= OVERLAP_DROP
        if (gap == 0).any() or (close[1:] & close[:-1]).any():
            raise ValueError(f"{country}: events on the same day 0 or a chain of overlaps")
        drop[g.index[np.r_[False, close]]] = True
    return drop


def apply_robustness_set(events, set_name, outcomes=None):
    """Analysis events of one robustness set ('main' = main labels), from the events table of 02.

    Eligible = eligible in 02 and, if `outcomes` (03) is given, in 03. The set's 'drop' rows are removed, 'relabel'
    rows change group, a new trigger date swaps in the event's row of that variant; ambiguous rows that the set does
    not relabel are left out. crisis follows the group and the overlap rule is applied again."""
    if outcomes is not None:
        ok = outcomes.set_index(["event_id", "variant"])["eligible"].astype(bool)
        events = events.assign(eligible=events["eligible"].astype(bool).to_numpy()
                               & ok.reindex(pd.MultiIndex.from_frame(events[["event_id", "variant"]])).to_numpy())
    ev = events[events["variant"] == "main"]
    if set_name != "main":
        rs = load_robustness_sets()
        rs = rs[rs["set"] == set_name]
        if rs.empty:
            raise ValueError(f"unknown robustness set {set_name!r}")
        relabel = rs[rs["action"] == "relabel"]
        new_day0 = relabel.loc[relabel["new_trigger_date"] != "", "event_id"]
        alt = events[(events["variant"] == set_name) & events["event_id"].isin(new_day0)]
        if len(alt) != len(new_day0):
            raise ValueError(f"{set_name}: events table lacks the alternative day-0 rows")
        ev = pd.concat([ev[~ev["event_id"].isin(rs.loc[rs["action"] == "drop", "event_id"]) & ~ev["event_id"].isin(new_day0)],
                        alt])
        ev = ev.assign(group=ev["event_id"].map(relabel.set_index("event_id")["new_is_crisis"].map(CRISIS_LABELS))
                       .fillna(ev["group"]))
    ev = ev[ev["group"].isin(ANALYSIS_GROUPS)].reset_index(drop=True)
    ev["crisis"] = (ev["group"] == "crisis").astype(int)
    ev["overlap_drop"] = overlap_flags(ev)
    return ev.sort_values(["country", "day0", "event_id"]).reset_index(drop=True)


def analysis_sample(events, outcomes, set_name="main", groups=ANALYSIS_GROUPS):
    """Events of one set in `groups`, eligible in 02 and 03 and kept by the overlap rule, with their outcomes."""
    t = apply_robustness_set(events, set_name, outcomes)
    t = t[t["eligible"] & ~t["overlap_drop"] & t["group"].isin(groups)]
    return t.merge(outcomes.drop(columns=["country", "eligible"]), on=["event_id", "variant"],
                   validate="one_to_one").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------
# Returns, factors, market model and outcomes (03)
# ----------------------------------------------------------------------------------------------------

def day_class(country, dates):
    """'monday' for an Israeli Monday while TASE traded on Sunday (a two-session return), 'other' for the other
    Israeli days, 'all' elsewhere."""
    dates = pd.DatetimeIndex(dates)
    if country != "ISR":
        return np.full(len(dates), "all", dtype=object)
    two = (dates.weekday == ISR_TWO_SESSION_WEEKDAY) & (dates <= pd.Timestamp(ISR_SUNDAY_TRADING_END))
    return np.where(two, "monday", "other").astype(object)


def span_returns(closes, t, s, entry=None):
    """ln P(t*) - ln P(s*) for the spans from s to t, with x* the last close of `closes` at or before x. NaN if s* is
    missing, more than GLOBAL_MAX_STALE_DAYS calendar days before s, or before `entry`."""
    cal, logp = closes.index.to_numpy(), np.log(closes.to_numpy())
    it, i_s = (np.searchsorted(cal, x, side="right") - 1 for x in (t, s))
    s_star = cal[np.maximum(i_s, 0)]
    ok = (i_s >= 0) & (s - s_star <= np.timedelta64(GLOBAL_MAX_STALE_DAYS, "D"))
    if entry is not None:
        ok &= s_star >= np.datetime64(entry)
    return np.where(ok, logp[np.maximum(it, 0)] - logp[np.maximum(i_s, 0)], np.nan)


def build_panels(prices, political_dates, external):
    """One table per country on its trading calendar.

    r: log return between valid closes; long_gap: it spans more than LONG_GAP_DAYS calendar days; day_class.
    g: equal-weighted mean of the other indices' returns over the same span (an index enters after
    GLOBAL_ENTRY_HISTORY valid closes; missing with fewer than MIN_GLOBAL contributors); w: return of the external
    series (load_external) over the same span; g_lag, w_lag: the previous observation's values.
    pol_dist: trading days to the nearest political date; est_ok: the day may enter an estimation window (not a
    long-gap return or the day after one, more than POL_EXCL days from a political date)."""
    closes = {c: valid_closes(prices, c) for c in TICKERS}
    entry = {c: s.index[GLOBAL_ENTRY_HISTORY] for c, s in closes.items() if len(s) > GLOBAL_ENTRY_HISTORY}
    panels = {}
    for c, own in closes.items():
        t, s = own.index[1:].to_numpy(), own.index[:-1].to_numpy()
        other = np.array([span_returns(closes[j], t, s, entry[j]) for j in entry if j != c])
        n = np.isfinite(other).sum(axis=0)
        g = np.where(n >= MIN_GLOBAL, np.nansum(other, axis=0) / np.maximum(n, 1), np.nan)
        w = span_returns(external, t, s)
        gap = own.index.to_series().diff().dt.days
        p = pd.DataFrame({"r": np.log(own).diff(), "long_gap": gap > LONG_GAP_DAYS,
                          "day_class": day_class(c, own.index), "g": np.r_[np.nan, g], "n_contrib": np.r_[0, n],
                          "w": np.r_[np.nan, w]}, index=own.index)
        p["g_lag"], p["w_lag"] = p["g"].shift(), p["w"].shift()
        pol = np.unique(own.index.get_indexer(political_dates.loc[political_dates["country"] == c, "trading_day"]
                                              .dropna()))
        if (pol < 0).any():
            raise ValueError(f"{c}: a political trading day is not on the trading calendar")
        idx = np.arange(len(own))
        p["pol_dist"] = (np.abs(idx[:, None] - pol[None, :]).min(axis=1) if len(pol) else np.inf)
        p["est_ok"] = p["r"].notna() & ~p["long_gap"] & ~p["long_gap"].shift(fill_value=False) & (p["pol_dist"] > POL_EXCL)
        panels[c] = p
    return panels


def market_model(panel, factors=FACTORS):
    """OLS of r on a constant and `factors` for the window EST_WIN placed around every day (est_ok days with all
    regressors), all windows at once from cumulative sums of the cross-products. A fit needs MIN_EST_DAYS days.
    Returns per day: a, one coefficient per factor, s (residual sd, one degree of freedom per coefficient), r2, n_est."""
    X = np.column_stack([np.ones(len(panel)), panel[list(factors)].to_numpy(float)])
    y = panel["r"].to_numpy(float)
    ok = panel["est_ok"].to_numpy(bool) & np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = np.where(ok[:, None], X, 0.0), np.where(ok, y, 0.0)
    lo, hi = np.clip(np.arange(len(y))[:, None] + np.array(EST_WIN) + [0, 1], 0, len(y)).T

    def window(v):
        cum = np.concatenate([np.zeros((1, *v.shape[1:])), np.cumsum(v, axis=0)])
        return cum[hi] - cum[lo]

    xtx, xty, yy, sy = window(X[:, :, None] * X[:, None, :]), window(X * y[:, None]), window(y * y), window(y)
    n, k = np.rint(window(ok.astype(float))).astype(int), X.shape[1]
    fit = n >= MIN_EST_DAYS
    beta = np.full((len(y), k), np.nan)
    beta[fit] = np.linalg.solve(xtx[fit], xty[fit][:, :, None])[:, :, 0]
    ssr = np.maximum(yy - (beta * xty).sum(axis=1), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(beta, index=panel.index, columns=["a", *factors]).assign(
            s=np.sqrt(ssr / (n - k)), r2=1 - ssr / (yy - sy ** 2 / n), n_est=n)
    out.attrs["factors"] = tuple(factors)
    return out


def ar_paths(panel, fits, positions, rel_days):
    """AR on positions + rel_days with the market model fitted around each position (rows: positions; NaN outside
    the calendar or where an input is missing)."""
    positions = np.asarray(positions)
    pos = positions[:, None] + np.asarray(rel_days)[None, :]
    p = np.clip(pos, 0, len(panel) - 1)
    f = fits.iloc[positions]
    pred = f["a"].to_numpy()[:, None] + sum(f[x].to_numpy()[:, None] * panel[x].to_numpy(float)[p]
                                            for x in fits.attrs["factors"])
    return np.where((pos >= 0) & (pos < len(panel)), panel["r"].to_numpy(float)[p] - pred, np.nan)


def sar_paths(panel, fits, positions, rel_days):
    return ar_paths(panel, fits, positions, rel_days) / fits["s"].to_numpy()[np.asarray(positions)][:, None]


def pre_volatility(panel, fits, positions, sd_c):
    """vol_pre: ln of the sd of AR over PRE_VOL_WIN (model fitted around each position) over the normal-day sd sd_c."""
    ar = ar_paths(panel, fits, positions, np.arange(PRE_VOL_WIN[0], PRE_VOL_WIN[1] + 1))
    return np.log(np.std(ar, axis=1, ddof=VOL_DDOF) / sd_c)


def normal_days(panels, fits):
    """One-day AR of every trading day from EVENTS_FROM to SAMPLE_END (model fitted around that day); normal = a
    valid AR, not a long-gap return, more than NORMAL_EXCL trading days from every political date."""
    rows = []
    for c, p in panels.items():
        ar = ar_paths(p, fits[c], np.arange(len(p)), [0])[:, 0]
        rows.append(pd.DataFrame({"country": c, "date": p.index, "day_class": p["day_class"], "ar": ar,
                                  "normal": np.isfinite(ar) & ~p["long_gap"] & (p["pol_dist"] > NORMAL_EXCL)})
                    [(p.index >= EVENTS_FROM) & (p.index <= SAMPLE_END)])
    return pd.concat(rows, ignore_index=True)


def thresholds(normal):
    """Quantiles QUANTILES of |AR| over the normal days of each country and day class."""
    nd = normal[normal["normal"]].assign(absar=lambda d: d["ar"].abs())
    q = nd.groupby(["country", "day_class"])["absar"]
    return pd.DataFrame({"n_normal": q.size(), **{name: q.quantile(p) for name, p in QUANTILES.items()}}).reset_index()


def event_outcomes(events, panels, fits, normal, thr):
    """Outcomes of every row of the events table. Eligible: eligible in 02, MIN_EST_DAYS estimation days, AR(0) and
    AR(+1) computable. For eligible rows: AR and SAR on OUTCOME_DAYS, the strong flags (|AR| of a day above the
    threshold of its class), vol_pre; and the AR/SAR paths on AR_WIN. Returns (outcomes, paths)."""
    sd_c = normal[normal["normal"]].groupby("country")["ar"].std(ddof=VOL_DDOF)
    q = thr.set_index(["country", "day_class"])
    rel = np.arange(AR_WIN[0], AR_WIN[1] + 1)
    rows, paths = [], []
    for e in events.itertuples(index=False):
        p, f, t0 = panels[e.country], fits[e.country], int(e.day0_pos)
        if p.index[t0] != e.day0:
            raise ValueError(f"{e.event_id}: day0_pos does not point to day0")
        ar, days = ar_paths(p, f, [t0], rel)[0], [t0 + k for k in OUTCOME_DAYS]
        ar_k = {k: ar[rel == k][0] for k in OUTCOME_DAYS}
        reason = (f"not eligible in 02: {e.ineligible_reason}" if not e.eligible
                  else f"fewer than {MIN_EST_DAYS} estimation days" if f["n_est"].iloc[t0] < MIN_EST_DAYS
                  else "AR(0) or AR(+1) not computable" if np.isnan(list(ar_k.values())).any() else "")
        row = {"event_id": e.event_id, "variant": e.variant, "country": e.country, "eligible": not reason,
               "drop_reason": reason, "s": f["s"].iloc[t0], "n_est": f["n_est"].iloc[t0],
               "long_gap": bool(p["long_gap"].iloc[[d for d in days if d < len(p)]].any())}
        if not reason:
            row |= {f"{x}{k}": v / (f["s"].iloc[t0] if x == "sar" else 1) for k, v in ar_k.items() for x in ("ar", "sar")}
            row |= {f"strong{name[1:]}": int(any(abs(ar_k[k]) > q.loc[(e.country, p["day_class"].iloc[d]), name]
                                                 for k, d in zip(OUTCOME_DAYS, days))) for name in QUANTILES}
            pre = ar[(rel >= PRE_VOL_WIN[0]) & (rel <= PRE_VOL_WIN[1])]
            row["vol_pre"] = np.log(np.std(pre, ddof=VOL_DDOF) / sd_c[e.country])     # NaN if an AR is missing
            paths.append(pd.DataFrame({"event_id": e.event_id, "variant": e.variant, "rel_day": rel, "ar": ar,
                                       "sar": ar / f["s"].iloc[t0]}))
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True)


# ----------------------------------------------------------------------------------------------------
# Placebo, tests and bootstrap (04 onwards)
# ----------------------------------------------------------------------------------------------------

def normal_pairs(normal, thr):
    """Pairs (d, d+1) of consecutive trading days of a country that are both normal: country, date (d), date1,
    day_class (of d), react = max(|AR(d)|, |AR(d+1)|) and, per threshold, above_<name>_0 / _1 (each day against
    the threshold of its own class)."""
    nd = normal.sort_values(["country", "date"]).reset_index(drop=True)
    nxt = nd.groupby("country").shift(-1)
    out = nd[["country", "date", "day_class"]].assign(date1=nxt["date"], react=np.fmax(nd["ar"].abs(), nxt["ar"].abs()))
    q = thr.set_index(["country", "day_class"]).reindex(pd.MultiIndex.from_frame(nd[["country", "day_class"]]))
    for name in QUANTILES:
        above = pd.Series(nd["ar"].abs().to_numpy() > q[name].to_numpy())
        out[f"above_{name}_0"], out[f"above_{name}_1"] = above, above.groupby(nd["country"]).shift(-1).eq(True)
    return out[nd["normal"].astype(bool) & nxt["normal"].eq(True)].reset_index(drop=True)


def placebo_pool(pairs, country, day0, day_class0):
    """The pairs of the event's country whose d has day 0's class and lies within PLACEBO_RADIUS calendar days of
    day 0 (at least PLACEBO_MIN_PAIRS of them)."""
    pool = pairs[(pairs["country"] == country) & (pairs["day_class"] == day_class0)
                 & ((pairs["date"] - pd.Timestamp(day0)).abs().dt.days <= PLACEBO_RADIUS)]
    if len(pool) < PLACEBO_MIN_PAIRS:
        raise ValueError(f"{country} {day0}: only {len(pool)} placebo pairs")
    return pool


def placebo_rate(pool, name, days):
    """Share of the pool's pairs above threshold `name` on at least one of `days` (0 = d, 1 = d+1)."""
    return np.logical_or.reduce([pool[f"above_{name}_{k}"].to_numpy() for k in days]).mean()


def event_strong(ev, thr, name, days):
    """1 if |AR(k)| of an event is above the threshold of day k's class for some k in `days` (columns ar<k>, class<k>)."""
    q = thr.set_index(["country", "day_class"])[name]
    return np.logical_or.reduce([ev[f"ar{k}"].abs().to_numpy()
                                 > q.reindex(pd.MultiIndex.from_arrays([ev["country"], ev[f"class{k}"]])).to_numpy()
                                 for k in days]).astype(int)


def poisson_binomial_upper(k, probs):
    """P(S >= k) for S a sum of independent Bernoulli(probs), by exact convolution."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return float(pmf[int(k):].sum())


def cluster_bootstrap(countries, values, in_a, in_b=None):
    """B_BOOT country-cluster bootstrap replicates of mean(values | a) [- mean(values | b)], resampling the countries
    with events of the tested groups. Returns (replicates, number without a value)."""
    in_a = np.asarray(in_a, dtype=bool)
    in_b = np.zeros_like(in_a) if in_b is None else np.asarray(in_b, dtype=bool)
    keep = in_a | in_b
    ids, inv = np.unique(np.asarray(countries)[keep], return_inverse=True)
    v = np.asarray(values, dtype=float)[keep]
    w = np.random.default_rng(SEED).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=B_BOOT)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = sum(sign * (w @ np.bincount(inv, v * m, len(ids))) / (w @ np.bincount(inv, m, len(ids)))
                   for sign, m in [(1, in_a[keep]), (-1, in_b[keep])] if m.any())
    stat = np.where(np.isfinite(stat), stat, np.nan)
    return stat, int(np.isnan(stat).sum())


def percentile_interval(replicates):
    tail = 100 * (1 - CI_LEVEL) / 2
    return tuple(np.nanpercentile(replicates, [tail, 100 - tail]))


def group_permutation_test(countries, values, is_a):
    """mean(values | a) - mean(values | not a) and its two-sided p, permuting the label within country N_GROUP_PERM
    times: p = (1 + #{|T*| >= |T|}) / (1 + N)."""
    countries, values, is_a = np.asarray(countries), np.asarray(values, dtype=float), np.asarray(is_a, dtype=bool)
    rng = np.random.default_rng(SEED)
    labels = np.empty((N_GROUP_PERM, len(values)), dtype=bool)
    for c in np.unique(countries):
        idx = np.flatnonzero(countries == c)
        labels[:, idx] = rng.permuted(np.tile(is_a[idx], (N_GROUP_PERM, 1)), axis=1)
    observed = values[is_a].mean() - values[~is_a].mean()
    perm = labels @ values / is_a.sum() - (~labels) @ values / (~is_a).sum()
    return observed, (1 + (np.abs(perm) >= np.abs(observed) - 1e-12).sum()) / (1 + N_GROUP_PERM)


def draw_placebo_paths(pool_paths, rng):
    """One path per event, drawn among its pool dates whose path has no missing value."""
    out = []
    for paths in pool_paths:
        valid = np.flatnonzero(np.isfinite(paths).all(axis=1))
        if not len(valid):
            raise ValueError("an event has no placebo date with a complete path")
        out.append(paths[rng.choice(valid)])
    return np.vstack(out)


# ----------------------------------------------------------------------------------------------------
# Prediction: leave-one-country-out folds and permutations (05)
# ----------------------------------------------------------------------------------------------------

def outer_folds(countries, day0):
    """Leave-one-country-out folds; training events within PURGE_DAYS calendar days of a test event are removed.
    Returns one dict per fold: country, train and test (positions)."""
    countries = np.asarray(countries)
    days = pd.to_datetime(pd.Series(day0)).to_numpy().astype("datetime64[D]").astype(np.int64)
    folds = []
    for c in np.unique(countries):
        test, other = np.flatnonzero(countries == c), np.flatnonzero(countries != c)
        train = other[np.abs(days[other][:, None] - days[test][None, :]).min(axis=1) > PURGE_DAYS]
        folds.append({"country": c, "train": train, "test": test})
    return folds


def permute_block(block, cells, rng):
    """Rows of `block` permuted jointly within each cell, cells in sorted order."""
    out = block.copy()
    for cell in np.unique(cells):
        idx = np.flatnonzero(cells == cell)
        out[idx] = block[idx[rng.permutation(len(idx))]]
    return out
