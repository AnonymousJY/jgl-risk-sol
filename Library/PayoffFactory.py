from Library.Payoff import *


def payoff_mc_factory(name: str) -> PayoffBase:

    payoffs = {
        'call': PayoffCall,
        'put': PayoffPut,
        'binary call': PayoffBinaryCall,
        'binary put': PayoffBinaryPut,
        'spread': PayoffSpread,
        'forward': PayoffForward
    }

    return payoffs[name.lower()]
