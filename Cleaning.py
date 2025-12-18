from sklearn.preprocessing import LabelEncoder

def position_cleaning(df, position):

    if position == 'QB':

        PlayerStats_QB = df[df['position'] == position].copy()

        drop_empty_cols = [col for col in PlayerStats_QB.columns if (PlayerStats_QB[col] == 0).all()]

        PlayerStats_QB_cleaned = PlayerStats_QB.drop(columns=drop_empty_cols)

        PlayerStats_QB_cleaned_2 = PlayerStats_QB_cleaned.dropna(axis=1, how='all')

        Qb_Stats = PlayerStats_QB_cleaned_2.drop(columns=['player_id','receiving_tds','player_display_name','season','season_type','recent_team','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url','receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa', 'target_share','air_yards_share'])

        return Qb_Stats
    
    elif position == 'WR':

        PlayerStats_WR = df[df['position'] == position].copy()

        WR_Stats = PlayerStats_WR.drop(columns=['player_id','player_display_name','season','season_type','recent_team','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

        drop_empty_cols = [col for col in WR_Stats.columns if (WR_Stats[col] == 0).all()]

        PlayerStats_WR_cleaned = WR_Stats.drop(columns=drop_empty_cols)

        PlayerStats_WR_cleaned_2 = PlayerStats_WR_cleaned.dropna(axis=1, how='all')

        return PlayerStats_WR_cleaned_2
    
    elif position == 'RB':

        PlayerStats_RB = df[df['position'] == position].copy()

        RB_Stats = PlayerStats_RB.drop(columns=['player_id','player_display_name','season','season_type','recent_team','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

        drop_empty_cols = [col for col in RB_Stats.columns if (RB_Stats[col] == 0).all()]

        PlayerStats_RB_cleaned = RB_Stats.drop(columns=drop_empty_cols)

        PlayerStats_RB_cleaned_2 = PlayerStats_RB_cleaned.dropna(axis=1, how='all')

        return PlayerStats_RB_cleaned_2
    
    elif position == 'TE':

        PlayerStats_TE = df[df['position'] == position].copy()

        TE_Stats = PlayerStats_TE.drop(columns=['player_id','player_display_name','season','season_type','recent_team','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

        drop_empty_cols = [col for col in TE_Stats.columns if (TE_Stats[col] == 0).all()]

        PlayerStats_TE_cleaned = TE_Stats.drop(columns=drop_empty_cols)

        PlayerStats_TE_cleaned_2 = PlayerStats_TE_cleaned.dropna(axis=1, how='all')

        return PlayerStats_TE_cleaned_2

def add_year_suffix(df, year):
    return df.rename(columns={col: f"{col}_{year}" for col in df.columns if col != 'player_id'})

def handle_categoricals(df):

    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df_encoded = df.copy()

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df_encoded
