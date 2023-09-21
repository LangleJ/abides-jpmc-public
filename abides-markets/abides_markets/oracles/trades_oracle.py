import datetime as dt
import logging
from math import sqrt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from .oracle import Oracle


logger = logging.getLogger(__name__)


class TradesOracle(Oracle):
    """The TradesOracle requires one parameter: a pd.Series indexed by ns time.
    a mean reversion coefficient, and a shock variance."""

    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: Dict[str, pd.Series],
    ) -> None:
        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close
        self.symbols: Dict[str, pd.Series] = symbols

        # The dictionary r holds the fundamenal value series for each symbol.
        self.r: Dict[str, pd.Series] = {}

        for symbol in symbols:
            s = symbols[symbol]
            self.r[symbol] = s

        logger.debug(f"TradesOracle initialized for symbols {symbols.keys()}")


    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the daily open price for the symbol given.

        In the case of the MeanRevertingOracle, this will simply be the first
        fundamental value, which is also the fundamental mean. We will use the
        mkt_open time as given, however, even if it disagrees with this.
        """

        # If we did not already know mkt_open, we should remember it.
        if (mkt_open is not None) and (self.mkt_open is None):
            self.mkt_open = mkt_open

        logger.debug(
            "Oracle: client requested {symbol} at market open: {}", self.mkt_open
        )

        loc = self.r[symbol].index.get_loc(pd.Timestamp(self.mkt_open), method='ffill')
        open_price = self.r[symbol].iloc[loc]
        logger.debug(f"Oracle: market open price was was {open_price}")

        return open_price

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        """Return a noisy observation of the current fundamental value.

        While the fundamental value for a given equity at a given time step does
        not change, multiple agents observing that value will receive different
        observations.

        Only the Exchange or other privileged agents should use noisy=False.

        sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.

        Each agent must pass its RandomState object to ``observe_price``.  This
        ensures that each agent will receive the same answers across multiple
        same-seed simulations even if a new agent has been added to the experiment.
        """

        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.r[symbol].loc[pd.Timestamp(self.mkt_close - 1)]
        else:
            loc = self.r[symbol].index.get_loc(pd.Timestamp(current_time), method='ffill')
            r_t = self.r[symbol].iloc[loc]

        # Generate a noisy observation of fundamental value at the current time.
        if sigma_n == 0:
            obs = r_t
        else:
            obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))

        logger.debug(f"Oracle: current fundamental value is {r_t} at {current_time}")
        logger.debug(f"Oracle: giving client value observation {obs}")

        #print(obs)
        # Reminder: all simulator prices are specified in integer cents.
        return obs
