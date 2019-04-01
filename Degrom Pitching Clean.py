import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)



def zoner(x):
    return "Zone " + str(x)


def hit_location(x):
    if x == 1:
        return 'Pitcher'
    elif x == 2:
        return 'Catcher'
    elif x == 3:
        return 'First Base'
    elif x==4:
        return 'Second Base'
    elif x==5:
        return 'Third Base'
    elif x==6:
        return 'Shortstop'
    elif x==7:
        return 'Left Field'
    elif x==8:
        return 'Center Field'
    elif x==9:
        return 'Right Field'
    else:
        return 'no hit location'


def lefty_righty(x):
    if x == 'L':
        return 'Left Handed Batter'
    else:
        return 'Right Handed batter'


def strike_ball(x):
    if x == 'X' or x == 'S':
        return 'Strike'
    else:
        return 'Ball'


def bbtype(x):
    if x != x:
        return 'Pitch Not Put in Play'


def on_third(x):
    if x != x:
        return 'No Runner on Third'
    else:
        return 'Runner on Third'


def on_second(x):
    if x != x:
        return 'No Runner on Second'
    else:
        return 'Runner on Second'


def on_first(x):
    if x != x:
        return 'No Runner on First'
    else:
        return 'Runner on First'


def makezero(x):
    if x != x:
        return 0
    else:
        return x


def if_fielding(x):
    if x=='Standard':
        return 'Standard Infield Alignment'
    elif x == 'Strategic':
        return 'Strategic Infield Alignment'
    else:
        return x


def of_fielding(x):
    if x=='Standard':
        return 'Standard Outfield Alignment'
    elif x == 'Strategic':
        return 'Strategic Outfield Alignment'
    else:
        return x



if __name__ == "__main__":
    df = pd.read_csv('Degrom Pitches 2018.csv')
    df['hit_location'] = df['hit_location'].apply(hit_location)
    df['zone'] = df['zone'].apply(zoner)
    df['Batter L or R'] = df['Batter L or R'].apply(lefty_righty)
    df['type'] = df['type'].apply(strike_ball)
    df['bb_type'] = df['bb_type'].apply(bbtype)
    df['on_3b'] = df['on_3b'].apply(on_third)
    df['on_2b'] = df['on_2b'].apply(on_second)
    df['on_1b'] = df['on_1b'].apply(on_first)
    df['hc_x']= df['hc_x'].apply(makezero)
    df['hc_y']= df['hc_y'].apply(makezero)
    df['hit_distance_sc']= df['hit_distance_sc'].apply(makezero)
    df['launch_speed']= df['launch_speed'].apply(makezero)
    df['launch_angle']= df['launch_angle'].apply(makezero)
    df['if_fielding_alignment'] = df['if_fielding_alignment'].apply(if_fielding)
    df['of_fielding_alignment'] = df['of_fielding_alignment'].apply(of_fielding)
    df.to_csv('pre dummy clean.csv')
    dummy_columns = ['events','description','zone', 'hit_location',
                    'bb_type','if_fielding_alignment','of_fielding_alignment']
    lb_make = LabelEncoder()
    df['pitch_type_code'] = lb_make.fit_transform(df['pitch_type'])
    df['Batter L or R Code'] = lb_make.fit_transform(df['Batter L or R'])
    df['on_3b_code'] = lb_make.fit_transform(df['on_3b'])
    df['on_2b_code'] = lb_make.fit_transform(df['on_2b'])
    df['on_1b_code'] = lb_make.fit_transform(df['on_1b'])
    df['ball_strike_code'] = lb_make.fit_transform(df['type'])
    df = pd.get_dummies(data = df, columns = dummy_columns)
    df['release_spin_rate'] = df['release_spin_rate'].fillna(2564)
    df['effective_speed'] = df['effective_speed'].fillna(abs((df['release_speed'] - df['effective_speed']).mean()))
    df.to_csv('Degrom Pitches 2018 Cleaned.csv')

