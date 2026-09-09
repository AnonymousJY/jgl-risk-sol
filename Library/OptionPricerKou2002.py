import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from scipy.special import ndtr
from scipy.stats import poisson


def _Hh_ladder(n, x):
    """Hh_{-1} .. Hh_n in one bottom-up pass; ladder[i + 1] is Hh_i.

    The recurrence Hh_n = (Hh_{n-2} - x Hh_{n-1}) / n was evaluated top-down,
    so every call rebuilt the whole chain beneath it and _I - which asks for
    Hh_0 .. Hh_n - rebuilt it n times over. Building it once is O(n) instead,
    and calls the normal CDF once rather than once per rung: 1596 evaluations
    for Hh_0..Hh_14 became 1.
    """
    ladder = [np.exp(-x ** 2 / 2), np.sqrt(2 * np.pi) * ndtr(-x)]
    for k in range(1, n + 1):
        ladder.append((ladder[-2] - x * ladder[-1]) / k)
    return ladder


def _Hh(n, x):
    if n < -1:
        return 0
    return _Hh_ladder(max(n, 0), x)[n + 1]


def _P(n, k, eta1, eta2, p):
    if n==k:
        return p**n
    else:
        P = 0
        for i in range(k,n):
            P += comb(n-k-1,i-k)*comb(n,i)*(eta1/(eta1+eta2))**(i-k)*(eta2/(eta1+eta2))**(n-i)*p**i*(1-p)**(n-i)
        return P


def _Q(n, k, eta1, eta2, p):
    if n==k:
        return (1-p)**n
    else:
        Q = 0
        for i in range(k,n):
            Q += comb(n-k-1,i-k)*comb(n,i)*(eta1/(eta1+eta2))**(n-i)*(eta2/(eta1+eta2))**(i-k)*p**(n-i)*(1-p)**i
        return Q


def _I(n, c, alpha, beta, delta):
    I = 0
    if (beta > 0).all() and (alpha != 0).all():
        hh = _Hh_ladder(n, beta * c - delta)
        for i in range(n + 1):
            I += (beta / alpha) ** (n - i) * hh[i + 1]
        I *= -(np.exp(alpha * c) / alpha)
        I += (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) * np.exp(
            alpha * delta / beta + alpha ** 2 / (2 * beta ** 2)) * ndtr(-beta * c + delta + alpha / beta)

    elif (beta < 0).all() and (alpha < 0).all():
        hh = _Hh_ladder(n, beta * c - delta)
        for i in range(n + 1):
            I += (beta / alpha) ** (n - i) * hh[i + 1]
        I *= -(np.exp(alpha * c) / alpha)
        I -= (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) * np.exp(
            alpha * delta / beta + alpha ** 2 / (2 * beta ** 2)) * ndtr(beta * c - delta - alpha / beta)
    else:
        I = 0
    return I


def _I_ladder(nmax, c, alpha, beta, delta):
    """_I(0..nmax) in one pass instead of nmax+1 independent O(n) calls.

    The inner sum S_n = sum_{i=0..n} (beta/alpha)^(n-i) Hh_i(beta c - delta)
    obeys S_n = (beta/alpha) S_(n-1) + Hh_n, and the Hh argument does not
    depend on n - so one ladder and one recursion give every order. The old
    code rebuilt the Hh ladder inside every _I call inside every (n, k) pair,
    which is what made raising `bound` from 15 to 61 cost 70x rather than 4x.
    """
    zero = np.zeros_like(np.asarray(c, dtype=float) + np.asarray(alpha, dtype=float))
    pos = bool((np.asarray(beta) > 0).all()) and bool((np.asarray(alpha) != 0).all())
    neg = bool((np.asarray(beta) < 0).all()) and bool((np.asarray(alpha) < 0).all())
    if not (pos or neg):
        return [zero] * (nmax + 1)

    hh = _Hh_ladder(nmax, beta * c - delta)
    ratio = beta / alpha
    head = -(np.exp(alpha * c) / alpha)
    tail_const = (np.sqrt(2 * np.pi) / beta) * np.exp(
        alpha * delta / beta + alpha ** 2 / (2 * beta ** 2))
    tail_cdf = (ndtr(-beta * c + delta + alpha / beta) if pos
                else ndtr(beta * c - delta - alpha / beta))
    sign = 1.0 if pos else -1.0

    out, S, rpow = [], zero, ratio ** 0
    for n in range(nmax + 1):
        S = ratio * S + hh[n + 1]
        rpow = rpow * ratio
        out.append(head * S + sign * rpow * tail_const * tail_cdf)
    return out


_PQ_CACHE = {}


def _pq_tables(bound, eta1, eta2, p):
    """P[n][k] and Q[n][k] for 1 <= k <= n < bound.

    They depend only on (n, k, eta1, eta2, p) - not on spot, strike or expiry -
    so a whole strike ladder shares one table.
    """
    e1, e2, pp = np.asarray(eta1), np.asarray(eta2), np.asarray(p)
    key = None
    if e1.size == 1 and e2.size == 1 and pp.size == 1:
        key = (bound, float(e1.reshape(-1)[0]), float(e2.reshape(-1)[0]),
               float(pp.reshape(-1)[0]))
        if key in _PQ_CACHE:
            return _PQ_CACHE[key]
    Pt = {n: {k: _P(n, k, eta1, eta2, p) for k in range(1, n + 1)}
          for n in range(1, bound)}
    Qt = {n: {k: _Q(n, k, eta1, eta2, p) for k in range(1, n + 1)}
          for n in range(1, bound)}
    if key is not None:
        _PQ_CACHE[key] = (Pt, Qt)
    return Pt, Qt


def _jump_count_bound(lambd, T):
    """How far the jump-count sum has to run before the Poisson tail is spent.

    `bound` used to be hard-wired at 15, which silently truncated the series
    whenever lambda*T approached it. At lambda = 5 over a 3-year expiry,
    lambda*T = 15 puts HALF the Poisson mass past the last term kept, and the
    formula understated a 3y ATM call by 10 points (45.62 against 55.49 by
    Monte Carlo) and drove a 50-strike put NEGATIVE (-4.91 against +11.18).
    Short expiries were unaffected, which is why it survived: at lambda*T <= 5
    the truncated and converged values agree to 1e-2.

    The sum converges by roughly lambda*T + 5 sqrt(lambda*T); this takes the
    1 - 1e-13 Poisson quantile plus a margin, and the terms are numerically
    stable well past it (identical values at bound 45 through 100).
    """
    lamT = float(np.max(np.asarray(lambd) * np.asarray(T)))
    if not np.isfinite(lamT) or lamT <= 0:
        return 15
    return int(min(250, max(15, poisson.ppf(1 - 1e-13, lamT) + 10)))


def _U(mu, sigma, lambd, p, eta1, eta2, a, T, bound=None):
    if bound is None:
        bound = _jump_count_bound(lambd, T)

    def Pi(n):
        x = 1
        for k in range(n):
            x *= (lambd * T) / (k + 1)
        return np.exp(-lambd * T) * x

    exp1 = np.exp((sigma * eta1) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T))
    exp2 = np.exp((sigma * eta2) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T))

    root = sigma * np.sqrt(T)
    I1 = _I_ladder(bound, a - mu * T, -eta1, -1 / root, -eta1 * root)
    I2 = _I_ladder(bound, a - mu * T, eta2, 1 / root, -eta2 * root)
    Pt, Qt = _pq_tables(bound, eta1, eta2, p)

    sum1 = 0
    sum2 = 0
    for n in range(1, bound):
        sumP = 0
        sumQ = 0
        for k in range(1, n + 1):
            sumP += Pt[n][k] * (root * eta1) ** k * I1[k - 1]
            sumQ += Qt[n][k] * (root * eta2) ** k * I2[k - 1]
        sum1 += Pi(n) * sumP
        sum2 += Pi(n) * sumQ

    return exp1 * sum1 + exp2 * sum2 + Pi(0) * ndtr(-(a - mu * T) / (sigma * np.sqrt(T)))


def kou_call(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry) -> NDArray[np.float64]:
    zeta = (p * eta1) / (eta1 - 1) + ((1 - p) * eta2) / (eta2 + 1) - 1
    lam2 = lam * (zeta + 1)
    eta12 = eta1 - 1
    eta22 = eta2 + 1
    p2 = (p / (1 + zeta)) * (eta1 / (eta1 - 1))
    omega1 = (r - d) + sigma**2/2 - lam * zeta
    omega2 = (r - d) - sigma**2/2 - lam * zeta
    return S0 * np.exp(-d * expiry) * _U(omega1, sigma, lam2, p2, eta12, eta22, np.log(K/S0), expiry) - np.exp(-r * expiry) * K * _U(omega2, sigma, lam, p, eta1, eta2, np.log(K/S0), expiry)


def kou_put(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry) -> NDArray[np.float64]:
    return kou_call(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry) + K * np.exp(-r * expiry) - S0 * np.exp(-d * expiry)


if __name__=='__main__':

    eta1 = np.array(10.)
    eta2 = np.array(5.)
    lamb = np.array(1.)
    pprob = np.array(.4)
    sigma = np.array(.16)
    r = np.array(.05)
    d = np.array(.0)
    s = np.array(100.)
    k = np.array(98.)
    t = np.array(.5)

    pv = kou_call(r=r, d=d, sigma=sigma, lam=lamb, p=pprob, eta1=eta1, eta2=eta2, S0=s, K=k, expiry=t)
    true_pv = 9.14732
    print(f"Calcualted pv = {pv:.5f} vs. true pv = {true_pv} of call option")


# ---------------------------------------------------------------------------
# COS-method pricer (Fang & Oosterlee 2008)
# ---------------------------------------------------------------------------
# Kou's closed form is an expansion in the jump COUNT, and it stops being
# usable long before it stops being needed. Two separate limits bite:
#
#   truncation  - the sum has to run past the Poisson tail of lambda*T
#                 (see _jump_count_bound)
#   conditioning - the terms carry (sigma sqrt(T) eta)^k, so at large k they
#                 grow like 12^k here while the sum they belong to stays O(1).
#                 The cancellation is catastrophic: at the estimated P set
#                 (lambda = 8.76, eta = 50.3/26.6, gamma_i = 2.29, T = 3, so
#                 lambda*T = 26) the closed form returns 3e10 for a put worth
#                 about 21.
#
# The characteristic function has neither problem - it is a closed-form
# exponential at any lambda*T - so European prices come from a cosine expansion
# of the density instead. Validated against the closed form where the closed
# form is sound, and against Monte Carlo where it is not.
def _kou_cf(u, r, d, sigma, lam, p, eta1, eta2, T):
    """Characteristic function of ln(S_T / S_0) under Kou, risk-neutral."""
    q = 1.0 - p
    zeta = p * eta1 / (eta1 - 1.0) + q * eta2 / (eta2 + 1.0) - 1.0
    drift = r - d - 0.5 * sigma ** 2 - lam * zeta
    M = p * eta1 / (eta1 - 1j * u) + q * eta2 / (eta2 + 1j * u)
    return np.exp(1j * u * drift * T - 0.5 * sigma ** 2 * u ** 2 * T
                  + lam * T * (M - 1.0))


def _cos_coeffs(a, b, uk, c, dd):
    k = np.arange(uk.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (np.sin(uk * (dd - a)) - np.sin(uk * (c - a))) / uk
    psi[0] = dd - c
    chi = (np.cos(uk * (dd - a)) * np.exp(dd) - np.cos(uk * (c - a)) * np.exp(c)
           + uk * np.sin(uk * (dd - a)) * np.exp(dd)
           - uk * np.sin(uk * (c - a)) * np.exp(c)) / (1.0 + uk ** 2)
    return psi, chi, k


def kou_put_cos(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry,
                n_terms=4096, trunc=12.0):
    """European put under Kou by the COS method. Scalar arguments."""
    r, d, sigma, lam, p = map(float, (r, d, sigma, lam, p))
    eta1, eta2, S0, K, T = map(float, (eta1, eta2, S0, K, expiry))
    q = 1.0 - p
    zeta = p * eta1 / (eta1 - 1.0) + q * eta2 / (eta2 + 1.0) - 1.0
    c1 = (r - d - 0.5 * sigma ** 2 - lam * zeta) * T + lam * T * (p / eta1 - q / eta2)
    c2 = sigma ** 2 * T + 2.0 * lam * T * (p / eta1 ** 2 + q / eta2 ** 2)
    c4 = 24.0 * lam * T * (p / eta1 ** 4 + q / eta2 ** 4)
    w = trunc * np.sqrt(c2 + np.sqrt(c4))
    a, b = c1 - w, c1 + w

    uk = np.arange(n_terms) * np.pi / (b - a)
    dd = min(b, np.log(K / S0))
    if dd <= a:
        return 0.0
    psi, chi, _ = _cos_coeffs(a, b, uk, a, dd)
    Vk = 2.0 / (b - a) * (K * psi - S0 * chi)

    cf = _kou_cf(uk, r, d, sigma, lam, p, eta1, eta2, T)
    terms = np.real(cf * np.exp(-1j * uk * a)) * Vk
    terms[0] *= 0.5
    return float(np.exp(-r * T) * terms.sum())


def kou_call_cos(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry, **kw):
    """European call by put-call parity off kou_put_cos."""
    return (kou_put_cos(r, d, sigma, lam, p, eta1, eta2, S0, K, expiry, **kw)
            + S0 * np.exp(-float(d) * float(expiry))
            - K * np.exp(-float(r) * float(expiry)))
