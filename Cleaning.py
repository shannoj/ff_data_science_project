from sklearn.preprocessing import LabelEncoder
from features import completion_percent, avg_pass_yards_allow

def position_cleaning(df, position):

    if position == 'QB':

        defense_stats = avg_pass_yards_allow(df)

        PlayerStats_QB = df[df['position'] == position].copy()

        PlayerStats_QB = completion_percent(PlayerStats_QB)

        PlayerStats_QB = PlayerStats_QB[PlayerStats_QB['passing_yards'] > 0]

        drop_empty_cols = [col for col in PlayerStats_QB.columns if (PlayerStats_QB[col] == 0).all()]

        drop_cols = [col for col in PlayerStats_QB.columns if col.startswith('fg') or col.startswith('pat') or col.startswith('pat') or col.startswith('punt') or col.startswith('kickoff') or col.startswith('gwfg') or col.startswith('fumble') or col.startswith('rushing')]

        def_own_cols = [col for col in PlayerStats_QB.columns if col.startswith('def_') and col.endswith('_x')]

        cols_to_drop = drop_empty_cols + drop_cols + def_own_cols

        PlayerStats_QB_cleaned = PlayerStats_QB.drop(columns= cols_to_drop)

        PlayerStats_QB_cleaned_2 = PlayerStats_QB_cleaned.dropna(axis=1, how='all')

        Qb_Stats = PlayerStats_QB_cleaned_2.drop(columns=['carries', 'passing_2pt_conversions', 'completions', 'attempts', 'passing_tds','passing_interceptions','sacks_suffered','sack_yards_lost','sack_fumbles','sack_fumbles_lost','passing_air_yards','passing_yards_after_catch','passing_first_downs','player_id','receiving_tds','season','season_type','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url','receptions', 'targets','receiving_yards','receiving_air_yards','receiving_yards_after_catch','receiving_first_downs','receiving_epa', 'target_share','air_yards_share', 'penalties', 'penalty_yards', 'fantasy_points_ppr', 'player_display_name'])

        QbStats_enhanced = Qb_Stats.merge(
                defense_stats,
                on=['week', 'opponent'],
                how='left'
            )
        
        QbStats_enhanced = QbStats_enhanced.drop(columns=['opponent', 'week'])

        return QbStats_enhanced
    
    elif position == 'WR':

        PlayerStats_WR = df[df['position'] == position].copy()

        WR_Stats = PlayerStats_WR.drop(columns=['player_id','player_display_name','season','season_type','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

        drop_empty_cols = [col for col in WR_Stats.columns if (WR_Stats[col] == 0).all()]

        PlayerStats_WR_cleaned = WR_Stats.drop(columns=drop_empty_cols)

        PlayerStats_WR_cleaned_2 = PlayerStats_WR_cleaned.dropna(axis=1, how='all')

        return PlayerStats_WR_cleaned_2
    
    elif position == 'RB':

        PlayerStats_RB = df[df['position'] == position].copy()

        RB_Stats = PlayerStats_RB.drop(columns=['player_id','player_display_name','season','season_type','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

        drop_empty_cols = [col for col in RB_Stats.columns if (RB_Stats[col] == 0).all()]

        PlayerStats_RB_cleaned = RB_Stats.drop(columns=drop_empty_cols)

        PlayerStats_RB_cleaned_2 = PlayerStats_RB_cleaned.dropna(axis=1, how='all')

        return PlayerStats_RB_cleaned_2
    
    elif position == 'TE':

        PlayerStats_TE = df[df['position'] == position].copy()

        TE_Stats = PlayerStats_TE.drop(columns=['player_id','player_display_name','season','season_type','position_group','player_name','position','fantasy_points','racr','wopr','headshot_url'])

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
