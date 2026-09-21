# 只會在第一次 import 時讀一次
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
# 從 pages/ 回到 project/

csv_path = os.path.join(BASE_DIR, "data", "real_penalties.csv")

df = pd.read_csv(csv_path)

# 建立 player → 慣用腳 對照表
PLAYER_FOOT_MAP = (
    df.groupby("player_name")["body_part"]
      .agg(lambda x: x.value_counts().idxmax())
      .to_dict()
)