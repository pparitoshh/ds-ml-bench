import json

import pandas as pd


def lift(segment_df: pd.DataFrame) -> float:
    ctrl = segment_df[segment_df["variant"] == "control"]
    treat = segment_df[segment_df["variant"] == "treatment"]
    ctrl_rate = ctrl["conversions"].sum() / ctrl["users"].sum()
    treat_rate = treat["conversions"].sum() / treat["users"].sum()
    return float((treat_rate - ctrl_rate) / ctrl_rate)


df = pd.read_csv("ab_data.csv")

answer = {
    "overall_lift": lift(df),
    "segment_a_lift": lift(df[df["segment"] == "A"]),
    "segment_b_lift": lift(df[df["segment"] == "B"]),
}

with open("answer.json", "w") as f:
    json.dump(answer, f, indent=2)
