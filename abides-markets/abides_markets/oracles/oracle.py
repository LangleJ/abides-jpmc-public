from abides_core import NanosecondTime
import numpy as np

class Oracle:
    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        raise NotImplementedError
    
    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        raise NotImplementedError
