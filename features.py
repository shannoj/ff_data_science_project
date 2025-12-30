
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