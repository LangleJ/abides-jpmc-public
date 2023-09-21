import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'/home/john/abides-jpmc-public/abides-markets')
sys.path.insert(0,'/home/john/abides-jpmc-public/abides-core')

from abides_core import abides
from abides_core.utils import parse_logs_df, ns_date, str_to_ns, fmt_ts
from abides_markets.configs import jl02

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

df = pd.read_csv("/home/john/data/ETHBTC-trades-2023-09-07.csv", names=['id','price','qty','base_qty', 'isSell','?'], index_col=4 )

df.index = pd.to_datetime(df.index, unit='ms')
df = df[~df.index.duplicated(keep='first')]
df = df.sort_index()

prices_df = df['price']
min_diff = abs(prices_df.diff().min())
prices_df = prices_df / min_diff 
prices_df *= 100.0
prices_df = prices_df.astype(int)

trades_df = -df['qty'] * (df['isSell'].astype(int)*2-1)
trades_df = (trades_df / trades_df.abs().min()).astype(int)

monotonic = df.index.is_monotonic_increasing
assert(monotonic)

mkt_open = prices_df.index[0].value
#mkt_close = df.index[-1].value + str_to_ns("00:30:00")
mkt_close = prices_df.index[0].value + str_to_ns("04:00:00")

config = jl02.build_config(mkt_open=mkt_open, mkt_close=mkt_close, symbols={'ETHBTC':prices_df}, trades={'ETHBTC':trades_df})
#config = rmsc04.build_config()
end_state = abides.run( config )

_df = prices_df[prices_df.index.astype(int) < mkt_close]
_idx = _df.index.astype(int) - _df.index[0].value
plt.plot(_idx, _df.values, 'r')


order_book = end_state["agents"][0].order_books["ETHBTC"]
L1 = order_book.get_L1_snapshots()
best_bids = pd.DataFrame(L1["best_bids"],columns=["time","price","qty"])
best_asks = pd.DataFrame(L1["best_asks"],columns=["time","price","qty"])

## All times are in ns from 1970, remove the date component to put them in ns from midnight
best_bids["time"] = best_bids["time"].apply( lambda x: x - ns_date(x) )
best_asks["time"] = best_asks["time"].apply( lambda x: x - ns_date(x) )

plt.plot(best_bids.time,best_bids.price, 'b')
plt.plot(best_asks.time,best_asks.price, 'b')

