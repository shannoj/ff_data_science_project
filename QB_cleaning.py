def QB_cleaning(file):

    position = 'position'
    Quarterbacks = 'QB'

    mask = file[position] == Quarterbacks

    PlayerStats_QB = file[mask]

    drop_empty_cols = [col for col in PlayerStats_QB.columns if (PlayerStats_QB[col] == 0).all()]

    PlayerStats_QB_cleaned = PlayerStats_QB.drop(columns=drop_empty_cols)

    PlayerStats_QB_cleaned_2 = PlayerStats_QB_cleaned.dropna(axis=1, how='all')

    Qb_Stats = PlayerStats_QB_cleaned_2.drop(columns=['racr','wopr','headshot_url','receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa','receiving_2pt_conversions', 'target_share','air_yards_share'])
    return Qb_Stats