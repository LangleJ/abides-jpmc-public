# JL01 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 2     Adaptive Market Maker Agents
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents
import sys 
import os
from datetime import datetime

import numpy as np
import pandas as pd



from abides_core.utils import str_to_ns, datetime_str_to_ns, get_wake_time
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    WorldAgent
)

from abides_markets.oracles import TradesOracle
from abides_markets.models import OrderSizeModel
from abides_markets.utils import generate_latency_model


########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    mkt_open,
    mkt_close,
    symbols = {},    # {'symbol':pd.Series}
    trades = {}, # {'symbol':pd.Series}
    num_noise_agents=0,
    num_value_agents=1,
    num_momentum_agents=0,
    num_mm = 1,

    seed=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
    #date="20210205",
#    end_time="10:00:00",
    stdout_log_level="DEBUG",
    starting_cash=10_000_000,  # Cash in this simulator is always in CENTS.
    log_orders=True,  # if True log everything
    # 1) Exchange Agent
    book_logging=True,
    book_log_depth=10,
    stream_history_length=500,
    exchange_log_orders=None,
    # 2) Noise Agent
    
    # 3) Value Agents
    
    r_bar=100_000,  # true mean fundamental value
    kappa=1.67e-15,  # Value Agents appraisal of mean-reversion
    lambda_a=5.7e-12,  # ValueAgent arrival rate

    # 4) Market Maker Agents
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
    
    mm_window_size="adaptive",
    mm_pov=0.025,
    mm_num_ticks=10,
    mm_wake_up_freq="60S",
    mm_min_order_size=1,
    mm_skew_beta=0,
    mm_price_skew=4,
    mm_level_spacing=5,
    mm_spread_alpha=0.75,
    mm_backstop_quantity=0,
    mm_cancel_limit_delay=50,  # 50 nanoseconds
    # 5) Momentum Agents
    
):
    """
    create the background configuration for rmsc04
    These are all the non-learning agent that will run in the simulation
    :param seed: seed of the experiment
    :type seed: int
    :param log_orders: debug mode to print more
    :return: all agents of the config
    :rtype: list
    """

    # fix seed
    np.random.seed(seed)

    def path_wrapper(pomegranate_model_json):
        """
        temporary solution to manage calls from abides-gym or from the rest of the code base
        TODO:find more general solution
        :return:
        :rtype:
        """
        # get the  path of the file
        path = os.getcwd()
        if path.split("/")[-1] == "abides_gym":
            return "../" + pomegranate_model_json
        else:
            return pomegranate_model_json

    mm_wake_up_freq = str_to_ns(mm_wake_up_freq)

    # order size model
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model
    # market marker derived parameters
    MM_PARAMS = [
        (mm_window_size, mm_pov, mm_num_ticks, mm_wake_up_freq, mm_min_order_size),
    ] * num_mm
    NUM_MM = len(MM_PARAMS)
    # noise derived parameters
    SIGMA_N = r_bar / 100000  # observation noise variance   # was 100

    # date&time
    #DATE = int(pd.to_datetime(date).to_datetime64())
    MKT_OPEN = mkt_open 
    MKT_CLOSE = mkt_close
    # These times needed for distribution of arrival times of Noise Agents
    NOISE_MKT_OPEN = mkt_open - str_to_ns("00:30:00")
    NOISE_MKT_CLOSE = mkt_close + str_to_ns("16:00:00")

    oracle = TradesOracle(MKT_OPEN, NOISE_MKT_CLOSE, symbols)

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []
    symbol_list = list(symbols.keys())
    Ns = len(symbol_list)

    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=MKT_OPEN,
                mkt_close=MKT_CLOSE,
                symbols=list(symbols.keys()),
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=stream_history_length,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    agents.extend(
        [
            WorldAgent(
                id=1,
                name="World Agent",
                type="WorldAgent",
                trades = trades[symbol_list[k]],
                symbol=symbol_list[k],
                starting_cash=starting_cash,
                log_orders=log_orders,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            ) for k in  range(Ns)
        ]   
    )
    agent_types.extend(["WorldAgent"])
    agent_count += 1

    if num_noise_agents > 0:
        agents.extend(
            [
                NoiseAgent(
                    id=agent_count + j+k*Ns,
                    name="NoiseAgent {}".format(j),
                    type="NoiseAgent",
                    symbol=symbol_list[k],
                    starting_cash=starting_cash,
                    wakeup_time=get_wake_time(NOISE_MKT_OPEN, NOISE_MKT_CLOSE),
                    log_orders=log_orders,
                    order_size_model=ORDER_SIZE_MODEL,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                    ),
                )
                for j in range( num_noise_agents)  for k in range(Ns)
            ]
        )
        agent_count += num_noise_agents * len(symbols.keys())
        agent_types.extend(["NoiseAgent"])

    if num_value_agents > 0:
        agents.extend(
            [
                ValueAgent(
                    id=agent_count + j+k*Ns,
                    name="Value Agent {}".format(j),
                    type="ValueAgent",
                    symbol=symbol_list[k],
                    starting_cash=starting_cash,
                    sigma_n=SIGMA_N,
                    r_bar=r_bar,
                    kappa=kappa,
                    lambda_a=lambda_a,
                    log_orders=log_orders,
                    order_size_model=ORDER_SIZE_MODEL,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                    ),
                )
                for j in range(num_value_agents)   for k in range(Ns)
            ]
        )
        agent_count += num_value_agents * len(symbols.keys())
        agent_types.extend(["ValueAgent"])

    if NUM_MM > 0:
        agents.extend(
            [
                AdaptiveMarketMakerAgent(
                    id=agent_count + j+k*Ns,
                    name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                    type="AdaptivePOVMarketMakerAgent",
                    symbol=symbol_list[k],
                    starting_cash=starting_cash,
                    pov=MM_PARAMS[idx][1],
                    min_order_size=MM_PARAMS[idx][4],
                    window_size=MM_PARAMS[idx][0],
                    num_ticks=MM_PARAMS[idx][2],
                    wake_up_freq=MM_PARAMS[idx][3],
                    poisson_arrival=True,
                    cancel_limit_delay=mm_cancel_limit_delay,
                    skew_beta=mm_skew_beta,
                    price_skew_param=mm_price_skew,
                    level_spacing=mm_level_spacing,
                    spread_alpha=mm_spread_alpha,
                    #subscribe=True, #JL
                    backstop_quantity=mm_backstop_quantity,
                    log_orders=log_orders,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                    ),
                )
                for idx, j in enumerate(range(NUM_MM))   for k in range(Ns)
            ]
        )
        agent_count += NUM_MM * len(symbols.keys())
        agent_types.extend("POVMarketMakerAgent")

    if num_momentum_agents > 0:
        agents.extend(
            [
                MomentumAgent(
                    id=agent_count+j+k*Ns,
                    name="MOMENTUM_AGENT_{}".format(j),
                    type="MomentumAgent",
                    symbol=symbol_list[k],
                    starting_cash=starting_cash,
                    min_size=1,
                    max_size=10,
                    wake_up_freq=str_to_ns("37s"),
                    poisson_arrival=True,
                    log_orders=log_orders,
                    order_size_model=ORDER_SIZE_MODEL,
                    random_state=np.random.RandomState(
                        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                    ),
                )
                for j in range(num_momentum_agents)   for k in range(Ns)
            ]
        )
        agent_count += num_momentum_agents * len(symbols.keys())
        agent_types.extend("MomentumAgent")

    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    # LATENCY
    latency_model = generate_latency_model(agent_count)

    default_computation_delay = 50  # 50 nanoseconds

    ##kernel args
    kernelStartTime = MKT_OPEN - str_to_ns("00:30:00")
    kernelStopTime = MKT_CLOSE + str_to_ns("1s")

    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"oracle": oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }
