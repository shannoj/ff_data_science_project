import pandas as pd


PlayerStats2025 = pd.read_csv("enhanced_stats_2025_3.csv", low_memory=False)


def PassingPoints(df):
    df['Passing_Points'] = (df.passing_yards)/20
    return df

def RushingPoints(df):
    df['Rushing_Points'] = (df.rushing_yards)/10
    return df

def TDPoints(df):
    df['TD_Points'] = (df.rushing_tds + df.passing_tds)*6
    return df

def completion_percent(df):

    df['comp_percent'] = (df.completions / df.attempts)

    return df

def avg_pass_yards_allow(df):

    columns = ['passing_yards', 'week', 'opponent', 'position']

    df_2 = df[columns]

    df_yards = df_2[(df_2['position'] == 'QB') & (df_2['passing_yards'] > 0)].copy()

    #df_yards = df_yards.rename(columns={'opponent': 'def_team'})

    df_yards = df_yards.drop('position', axis=1)
    df_yards = df_yards.sort_values(['opponent', 'week'])  # Changed def_team → opponent
    
    # Calculate stats (change def_team → opponent everywhere)
    df_yards['cumulative_avg'] = df_yards.groupby('opponent')['passing_yards'].expanding().mean().reset_index(level=0, drop=True)
    df_yards['def_recent_avg_allowed'] = df_yards.groupby('opponent')['passing_yards'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    df_yards['def_5game_avg_allowed'] = df_yards.groupby('opponent')['passing_yards'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    df_yards['def_consistency'] = df_yards.groupby('opponent')['passing_yards'].rolling(window=5, min_periods=2).std().reset_index(level=0, drop=True)
    
    # Shift
    df_yards['cumulative_avg'] = df_yards.groupby('opponent')['cumulative_avg'].shift(1)
    df_yards['def_recent_avg_allowed'] = df_yards.groupby('opponent')['def_recent_avg_allowed'].shift(1)
    df_yards['def_5game_avg_allowed'] = df_yards.groupby('opponent')['def_5game_avg_allowed'].shift(1)
    df_yards['def_consistency'] = df_yards.groupby('opponent')['def_consistency'].shift(1)
    
    df_yards['def_trend'] = df_yards['def_recent_avg_allowed'] - df_yards['cumulative_avg']
    
    # Aggregate (change def_team → opponent)
    defense_stats = df_yards.groupby(['opponent', 'week']).agg({
        'cumulative_avg': 'last',
        'def_recent_avg_allowed': 'last',
        'def_5game_avg_allowed': 'last',
        'def_trend': 'last',
        'def_consistency': 'last'
    }).reset_index()

    # Fill NaN (Week 1 has no prior data after shift)
    defense_cols = ['cumulative_avg', 'def_recent_avg_allowed', 
                    'def_5game_avg_allowed', 'def_trend', 'def_consistency']
    
    for col in defense_cols:
        league_avg = defense_stats[col].mean()
        defense_stats[col] = defense_stats[col].fillna(league_avg)

    return defense_stats

defense = avg_pass_yards_allow(PlayerStats2025)

print(defense.columns)


