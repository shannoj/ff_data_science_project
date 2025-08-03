import pandas as pd

BasicStats = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/stats/Basic_Stats.csv")
GameLogsRunning = pd.read_csv("../FF_DATA_SCIENCE_PROJECT/stats/Game_Logs_Runningback.csv")

GameLogsRunning.columns = [c.replace(' ', '_') for c in GameLogsRunning.columns]
print(GameLogsRunning.columns)

y = GameLogsRunning.Rushing_Yards
print(y.head())

