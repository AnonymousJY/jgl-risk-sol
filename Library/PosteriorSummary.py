"""
PosteriorSummary.py — explicit, version-independent posterior summaries.

WHY THIS EXISTS

The estimators previously read credible-interval bounds out of
``az.summary`` **by column position**:

    columns = params_df.columns
    column_ci_lower = columns[2]
    column_ci_upper = columns[3]

That is fragile in two ways at once. The column layout is an ArviZ
implementation detail, and — more seriously — the *meaning* of those columns
is a library default that has changed: ArviZ < 1.0 reported a 94% highest
density interval, ArviZ >= 1.0 reports an 89% equal-tailed interval. A codebase
that reads them positionally keeps running across that change while every
stored interval silently changes meaning.

This module computes the summary from the posterior samples directly, so the
convention lives here rather than in whichever ArviZ happens to be installed.

CONVENTION (fixed, project-wide)

    95% EQUAL-TAILED interval: the 2.5% and 97.5% posterior quantiles.

Equal-tailed, not highest-density. For the skewed posteriors in this model the
two differ, and an equal-tailed interval is what the quantile language of
"2.5%" and "97.5%" means.

For a normal posterior the interval width relates to the standard deviation by

    sd = width / (2 * 1.959964) = width / 3.919928

which is the factor the diagnostics use. Do not substitute an HDI factor.
"""

import numpy as np

# Project-wide convention. Change here and nowhere else.
CI_PROB = 0.95
CI_LOWER_Q = (1.0 - CI_PROB) / 2.0          # 0.025
CI_UPPER_Q = 1.0 - CI_LOWER_Q               # 0.975

# width -> sd, for a normal posterior. 2 * Phi^{-1}(0.975).
CI_WIDTH_TO_SD = 1.0 / 3.919927969080108

# Stamped into saved parameter files so estimates produced under different
# conventions can never be silently pooled.
CI_CONVENTION = "eti_0.95"


def _draws(idata, name):
    """Flat array of posterior draws for one variable, across ArviZ versions."""
    post = idata["posterior"] if not hasattr(idata, "posterior") else idata.posterior
    # ArviZ >= 1.0 hands back a DataTree; .dataset / .to_dataset() gives a Dataset
    for attr in ("dataset", "to_dataset"):
        if hasattr(post, attr) and not hasattr(post, "data_vars"):
            post = getattr(post, attr)
            post = post() if callable(post) else post
            break
    return np.asarray(post[name].values).ravel()


def summarize(idata, name, transform=None):
    """(mean, q2.5, q97.5) for one posterior variable.

    `transform` is applied to the DRAWS before summarising, not to the summary.
    That matters: exp(mean(log x)) is not mean(x), and the previous code
    exponentiated the summary rather than the draws, which reports a geometric
    rather than an arithmetic mean and shifts the interval.
    """
    x = _draws(idata, name)
    if transform is not None:
        x = transform(x)
    return (float(np.mean(x)),
            float(np.quantile(x, CI_LOWER_Q)),
            float(np.quantile(x, CI_UPPER_Q)))


def interval_to_sd(lower, upper):
    """Approximate posterior sd from an interval under this convention."""
    return (np.asarray(upper) - np.asarray(lower)) * CI_WIDTH_TO_SD
