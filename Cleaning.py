def QB_cleaning(file, is_2022 = False):

    position = 'position'
    Quarterbacks = 'QB'

    mask = file[position] == Quarterbacks

    PlayerStats_QB = file[mask]

    drop_empty_cols = [col for col in PlayerStats_QB.columns if (PlayerStats_QB[col] == 0).all()]

    PlayerStats_QB_cleaned = PlayerStats_QB.drop(columns=drop_empty_cols)

    PlayerStats_QB_cleaned_2 = PlayerStats_QB_cleaned.dropna(axis=1, how='all')

    if is_2022:
        Qb_Stats = PlayerStats_QB_cleaned_2.drop(columns=['passing_2pt_conversions','receiving_tds','rushing_fumbles_lost','rushing_fumbles','carries','rushing_epa','passing_yards_after_catch','passing_air_yards','week','rushing_2pt_conversions','sack_yards','season','opponent_team','season_type','recent_team','position_group','player_display_name','player_name','position','fantasy_points','racr','wopr','headshot_url','receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa', 'target_share','air_yards_share'])
    else:
        Qb_Stats = PlayerStats_QB_cleaned_2.drop(columns=['passing_2pt_conversions','rushing_fumbles_lost','rushing_fumbles','carries','rushing_epa','passing_yards_after_catch','passing_air_yards','week','rushing_2pt_conversions','sack_yards','season','opponent_team','season_type','recent_team','position_group','player_display_name','player_name','position','fantasy_points','racr','wopr','headshot_url','receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa','receiving_2pt_conversions', 'target_share','air_yards_share'])
    return Qb_Stats

def add_year_suffix(df, year):
    return df.rename(columns={col: f"{col}_{year}" for col in df.columns if col != 'player_id'})
