
import pandas as pd

# Read the full file
df = pd.read_csv("/data/nlahaye/NatureNet/Blue_Whale_In/blue_2015_2017_argos_w_deploy_locs_filtered.streamlined.csv")

# Split into 2015 vs 2017 based on the timestamp string
df_2015 = df[df["timestamp"].astype(str).str.startswith("2015")]
df_2017 = df[df["timestamp"].astype(str).str.startswith("2017")]

# Write out to separate CSVs (no index column)
df_2015.to_csv("/data/nlahaye/NatureNet/Blue_Whale_In/blue_argos_2015_only.csv", index=False)
df_2017.to_csv("/data/nlahaye/NatureNet/Blue_Whale_In/blue_argos_2017_only.csv", index=False)



