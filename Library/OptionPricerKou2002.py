import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from scipy.stats import norm


def _Hh(n, x):
    if n<-1: return 0
    elif n==-1:
        return np.exp(-x**2/2)
    elif n==0:
        return np.sqrt(2*np.pi)*norm.cdf(-x)
    else:
        return (_Hh(n-2,x)-x*_Hh(n-1,x))/n


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
        for i in range(n + 1):
            I += (beta / alpha) ** (n - i) * _Hh(i, beta * c - delta)
        I *= -(np.exp(alpha * c) / alpha)
        I += (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) * np.exp(
            alpha * delta / beta + alpha ** 2 / (2 * beta ** 2)) * norm.cdf(-beta * c + delta + alpha / beta)

    elif (beta < 0).all() and (alpha < 0).all():
        for i in range(n + 1):
            I += (beta / alpha) ** (n - i) * _Hh(i, beta * c - delta)
        I *= -(np.exp(alpha * c) / alpha)
        I -= (beta / alpha) ** (n + 1) * (np.sqrt(2 * np.pi) / beta) * np.exp(
            alpha * delta / beta + alpha ** 2 / (2 * beta ** 2)) * norm.cdf(beta * c - delta - alpha / beta)
    else:
        I = 0
    return I


def _U(mu, sigma, lambd, p, eta1, eta2, a, T, bound=15):
    def Pi(n):
        x = 1
        for k in range(n):
            x *= (lambd * T) / (k + 1)
        return np.exp(-lambd * T) * x

    exp1 = np.exp((sigma * eta1) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T))
    exp2 = np.exp((sigma * eta2) ** 2 * T / 2) / (sigma * np.sqrt(2 * np.pi * T))

    sum1 = 0
    sum2 = 0
    for n in range(1, bound):
        sumP = 0
        sumQ = 0
        for k in range(1, n + 1):
            sumP += _P(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta1) ** k * _I(k - 1, a - mu * T, -eta1,
                                                                                  -1 / (sigma * np.sqrt(T)),
                                                                                  -sigma * eta1 * np.sqrt(T))
            sumQ += _Q(n, k, eta1, eta2, p) * (sigma * np.sqrt(T) * eta2) ** k * _I(k - 1, a - mu * T, eta2,
                                                                                  1 / (sigma * np.sqrt(T)),
                                                                                  -sigma * eta2 * np.sqrt(T))
        sum1 += Pi(n) * sumP
        sum2 += Pi(n) * sumQ

    return exp1 * sum1 + exp2 * sum2 + Pi(0) * norm.cdf(-(a - mu * T) / (sigma * np.sqrt(T)))


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
