import os
import sys
import pandas as pd
from flow_package.multi_df_env import MultiDfEnv, EnvConfig

def test_multi_df_env():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_binary_part1.csv.gz'))
    df = pd.read_csv(path)

    print(len(df))

    config = EnvConfig(
        data=df,
        label_column="Label",
        render_mode=None,
        max_steps=100,
        test_mode=False,
        rolling_window=50,
        normalize_method='zscore'
    )

    env = MultiDfEnv(config)
    obs, info = env.reset()

    print(obs)
    print(info)
    assert obs.shape == (len(df.columns) + 2,)  # +2 for additional info