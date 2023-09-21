import logging
from typing import Optional

import numpy as np

from abides_core import Message, NanosecondTime

from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent

import pandas as pd

logger = logging.getLogger(__name__)

class WorldAgent(TradingAgent):
    def __init__(
        self,
        trades: pd.Series,  # qty bought, +ve for buy -ve for sell
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "IBM",
        starting_cash: int = 100_000,
        log_orders: float = False,
    ) -> None:
        # Base class init.
        super().__init__(id, name, type, random_state, starting_cash, log_orders)

        self.trades = trades
        # Store important parameters particular to the ZI agent.
        self.symbol: str = symbol  # symbol to trade

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading: bool = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state: str = "AWAITING_WAKEUP"

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time: Optional[NanosecondTime] = None

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # self.kernel is set in Agent.kernel_initializing()
        # self.exchange_id is set in TradingAgent.kernel_starting()

        super().kernel_starting(start_time)

        self.oracle = self.kernel.oracle

    def kernel_stopping(self) -> None:
        # Always call parent method to be safe.
        super().kernel_stopping()

        # Print end of day valuation.
        H = int(round(self.get_holdings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        # marked to fundamental
        rT = self.oracle.observe_price(
            self.symbol, self.current_time, sigma_n=0, random_state=self.random_state
        )

        # final (real) fundamental value times shares held.
        surplus = rT * H

        logger.debug("Surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings["CASH"] - self.starting_cash
        surplus = float(surplus) / self.starting_cash

        self.logEvent("FINAL_VALUATION", surplus, True)

        logger.debug(
            "{} final report.  Holdings: {}, end cash: {}, start cash: {}, final fundamental: {}, surplus: {}",
            self.name,
            H,
            self.holdings["CASH"],
            self.starting_cash,
            rT,
            surplus,
        )

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(current_time)

        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                logger.debug(f"{self.name} is ready to start trading now.")

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        iloc = self.trades.index.get_loc(pd.Timestamp(self.current_time), method='ffill')

        if self.prev_wake_time is None:
            self.next_time = self.current_time
            while self.next_time<=self.current_time:
                self.next_time = self.trades.index[iloc].value
                iloc += 1
            self.set_wakeup(self.next_time)
        else:
            if self.current_time == self.next_time:
                # This was a scheduled wake
                iloc = self.trades.index.get_loc(pd.Timestamp(self.current_time))
                self.next_time = self.trades.index[iloc+1].value

                self.set_wakeup(self.next_time)
            else:
                # This was a wake frquency wake
                return

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        if type(self) == WorldAgent:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

        self.prev_wake_time = self.current_time
        return

    def placeOrder(self) -> None:
        # estimate final value of the fundamental price
        # used for surplus calculation
        self.current_time
        trade = self.trades.loc[pd.Timestamp(self.next_time)]
        quantity = abs(trade)
        buy = trade > 0
        side = Side.BID if buy == 1 else Side.ASK

        if quantity > 0:
            self.place_market_order(self.symbol, quantity, side)
        return

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receive_message(current_time, sender_id, message)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == "AWAITING_SPREAD":
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if isinstance(message, QuerySpreadResponseMsg):
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed:
                    return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                val = self.next_time - self.current_time
                #if self.next_time == self.current_time:
                self.placeOrder()
                self.state = "AWAITING_WAKEUP"

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?


    def get_wake_frequency(self) -> NanosecondTime:
        val = 1 #self.random_state.randint(low = 0, high = 100)
        return val
