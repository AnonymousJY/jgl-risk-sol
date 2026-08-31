"""
Partition one estimated parameter across names into ordered tiers.

The problem: each name i has a posterior for a loading (gamma_i, beta_i,
rho_i). We want to sort names into K ordered tiers - Low / Medium / High -
so a desk can carry an exposure limit per tier.

Two decisions, deliberately separated, because they fail in different ways:

  1. WHERE do the tier boundaries go?      -> optimal_partition()
  2. WHICH tier does name i belong to?     -> assign_tiers()

Boundaries are a property of the cross-section. Membership is a property of
one name's posterior. Conflating them is what puts a name with a wide
posterior into the High tier on the strength of estimation noise alone.

On the algorithm. In one dimension, k-means has an EXACT solution by dynamic
programming - Wang & Song (2011), "Ckmeans.1d.dp: Optimal k-means clustering
in one dimension by dynamic programming", The R Journal 3(2), 29-33. Unlike
Lloyd's algorithm in higher dimensions there is no local optimum to get stuck
in, no random restarts, and no seed dependence: the same input always returns
the same globally optimal partition. That reproducibility is worth a great
deal when the output feeds a limit framework that has to be explained to a
validator.

Cost is O(K n^2) here, which is nothing for the universe sizes involved.
Wang & Song give an O(K n log n) refinement that is not worth the complexity
at n in the thousands.

Weights. Pass w_i = 1 / posterior_variance_i to precision-weight the
boundary placement, so a name whose loading is barely identified has less
say in where the cuts land than one estimated sharply. This is the cheap
version of the measurement-error correction; see PosteriorSummary for the
draws-based route used by assign_tiers().
"""

import numpy as np

__all__ = [
    "optimal_partition",
    "select_k",
    "assign_tiers",
    "tier_report",
]

TIER_NAMES_3 = ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# 1. Where do the boundaries go
# ---------------------------------------------------------------------------

def _prefix(x, w):
    """Weighted prefix sums, so any segment's cost is O(1) to evaluate."""
    n = len(x)
    pw = np.zeros(n + 1)
    pwx = np.zeros(n + 1)
    pwx2 = np.zeros(n + 1)
    np.cumsum(w, out=pw[1:])
    np.cumsum(w * x, out=pwx[1:])
    np.cumsum(w * x * x, out=pwx2[1:])
    return pw, pwx, pwx2


def _cost(i, j, pw, pwx, pwx2):
    """Weighted within-segment sum of squares for sorted x[i:j].

    sum_i w_i (x_i - mu)^2  =  sum w x^2  -  (sum w x)^2 / sum w
    """
    sw = pw[j] - pw[i]
    if sw <= 0:
        return 0.0
    swx = pwx[j] - pwx[i]
    swx2 = pwx2[j] - pwx2[i]
    c = swx2 - swx * swx / sw
    return c if c > 0 else 0.0


def optimal_partition(values, k, weights=None):
    """Globally optimal k-tier partition of a 1-D array.

    Parameters
    ----------
    values : (n,) array
        Point estimate of the loading per name, e.g. posterior mean gamma_i.
    k : int
        Number of tiers.
    weights : (n,) array, optional
        Per-name weight. Pass 1 / posterior variance to precision-weight.

    Returns
    -------
    cuts : (k-1,) array
        Ascending boundaries. A name with value v sits in tier
        ``np.searchsorted(cuts, v)``.
    labels : (n,) int array
        Tier index per name, in the ORIGINAL input order.
    wss : float
        Total weighted within-tier sum of squares of the optimal partition.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n:
        raise ValueError("k=%d exceeds the %d names supplied" % (k, n))

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.size != n:
        raise ValueError("weights and values must be the same length")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    pw, pwx, pwx2 = _prefix(xs, ws)

    INF = np.inf
    # D[j, i] = optimal cost of splitting the first i points into j+1 tiers
    D = np.full((k, n + 1), INF)
    B = np.zeros((k, n + 1), dtype=int)      # backtrack: start index of last tier

    for i in range(1, n + 1):
        D[0, i] = _cost(0, i, pw, pwx, pwx2)

    for j in range(1, k):
        for i in range(j + 1, n + 1):
            best, arg = INF, j
            for m in range(j, i):
                if D[j - 1, m] == INF:
                    continue
                c = D[j - 1, m] + _cost(m, i, pw, pwx, pwx2)
                if c < best:
                    best, arg = c, m
            D[j, i] = best
            B[j, i] = arg

    # backtrack
    bounds = []
    i = n
    for j in range(k - 1, 0, -1):
        m = B[j, i]
        bounds.append(m)
        i = m
    bounds.reverse()

    labels_sorted = np.zeros(n, dtype=int)
    edges = [0] + bounds + [n]
    for t in range(k):
        labels_sorted[edges[t]:edges[t + 1]] = t

    # boundary sits midway between the last point of one tier and the first
    # of the next, so it does not coincide with an observed name
    cuts = np.array([0.5 * (xs[m - 1] + xs[m]) for m in bounds])

    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted
    return cuts, labels, float(D[k - 1, n])


# ---------------------------------------------------------------------------
# 2. How many tiers does the data actually support
# ---------------------------------------------------------------------------

def select_k(values, weights=None, k_max=6):
    """BIC over k = 1..k_max, treating each tier as a Gaussian component.

    Returns
    -------
    best_k : int
    table : list of (k, wss, bic)

    Read this as a diagnostic, not a decision. If the density is unimodal -
    which is the common case for a continuous risk loading across a broad
    universe - BIC will keep improving with k without any of the tiers being
    a real mode. Run a dip test (Hartigan & Hartigan 1985) FIRST. If
    unimodality cannot be rejected, the tier count is an operational choice,
    like a credit rating scale, and should be defended as one rather than
    presented as a discovered structure.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    out = []
    for k in range(1, min(k_max, n) + 1):
        cuts, labels, wss = optimal_partition(x, k, weights)
        # Gaussian log-likelihood with per-tier variance
        ll = 0.0
        ok = True
        for t in range(k):
            xt = x[labels == t]
            nt = xt.size
            if nt < 2:
                ok = False
                break
            var = xt.var(ddof=0)
            if var <= 0:
                ok = False
                break
            ll += -0.5 * nt * (np.log(2 * np.pi * var) + 1.0) + nt * np.log(nt / n)
        if not ok:
            out.append((k, wss, np.inf))
            continue
        p = 3 * k - 1                        # means, variances, weights
        out.append((k, wss, float(-2 * ll + p * np.log(n))))
    finite = [(k, b) for k, _, b in out if np.isfinite(b)]
    best_k = min(finite, key=lambda t: t[1])[0] if finite else 1
    return best_k, out


# ---------------------------------------------------------------------------
# 3. Which tier does each name go in
# ---------------------------------------------------------------------------

def assign_tiers(draws, cuts, min_prob=0.5, fallback=None):
    """Assign each name to a tier using its POSTERIOR MASS, not its point estimate.

    Parameters
    ----------
    draws : (n_names, n_draws) array
        Posterior draws of the loading for each name.
    cuts : (k-1,) array
        Boundaries from optimal_partition().
    min_prob : float
        A name is placed in its most likely tier only if that tier holds at
        least this much posterior mass. Otherwise it falls back - see below.
    fallback : int, optional
        Tier index to use when no tier reaches min_prob. Defaults to the
        middle tier. The point is that a name we cannot place confidently
        should not be sitting in the High tier driving a limit breach.

    Returns
    -------
    labels : (n_names,) int
    probs : (n_names, k) float
        Posterior probability of each tier, per name. Report this - it is the
        honest statement of how much the assignment can be relied on.
    confident : (n_names,) bool
        Whether the name cleared min_prob.
    """
    d = np.atleast_2d(np.asarray(draws, dtype=float))
    cuts = np.asarray(cuts, dtype=float)
    k = cuts.size + 1
    if fallback is None:
        fallback = k // 2

    tier_of_draw = np.searchsorted(cuts, d)          # (n_names, n_draws)
    n_names = d.shape[0]
    probs = np.zeros((n_names, k))
    for t in range(k):
        probs[:, t] = (tier_of_draw == t).mean(axis=1)

    labels = probs.argmax(axis=1)
    confident = probs.max(axis=1) >= min_prob
    labels = np.where(confident, labels, fallback)
    return labels, probs, confident


def tier_report(names, labels, probs, tier_names=TIER_NAMES_3):
    """Plain-text table: name, assigned tier, and the full posterior split."""
    k = probs.shape[1]
    tn = tier_names if len(tier_names) == k else tuple("T%d" % i for i in range(k))
    width = max(len(str(s)) for s in names) if len(names) else 6
    head = "%-*s  %-7s  %s" % (width, "name", "tier", "  ".join("P(%s)" % t for t in tn))
    lines = [head, "-" * len(head)]
    for i, nm in enumerate(names):
        lines.append("%-*s  %-7s  %s"
                     % (width, nm, tn[labels[i]],
                        "  ".join("%6.3f" % p for p in probs[i])))
    return "\n".join(lines)
