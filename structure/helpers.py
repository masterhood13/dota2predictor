# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.
import os
import joblib
from sklearn.preprocessing import MinMaxScaler


def find_dict_in_list(dicts, key, value):
    try:
        return next(item for item in dicts if item[key] == value)
    except KeyError:
        return {"sum": None, "n": 0}


# Feature Engineering Functions
def calculate_team_features(df, team_prefix):
    """
    Function to calculate team-based features for a given prefix (radiant or dire).
    """
    # Team Hero Win Rate: Average win rate of the heroes for the team
    hero_winrate_cols = [f"{team_prefix}_player_{i}_hero_winrate" for i in range(1, 6)]
    df[f"{team_prefix}_avg_hero_winrate"] = df[hero_winrate_cols].mean(axis=1)
    #
    # # Team Player Win Rate: Average win rate of the players for the team
    # player_winrate_cols = [f"{team_prefix}_player_{i}_winrate" for i in range(1, 6)]
    # df[f"{team_prefix}_avg_player_winrate"] = df[player_winrate_cols].mean(axis=1)

    # Team Kills, Deaths, Assists, Rosh kills, last_hits, denies, hero/tower damage
    kills_cols = [f"{team_prefix}_player_{i}_kills" for i in range(1, 6)]
    deaths_cols = [f"{team_prefix}_player_{i}_deaths" for i in range(1, 6)]
    assists_cols = [f"{team_prefix}_player_{i}_assists" for i in range(1, 6)]
    roshans_killed_cols = [
        f"{team_prefix}_player_{i}_roshans_killed" for i in range(1, 6)
    ]
    last_hits_cols = [f"{team_prefix}_player_{i}_last_hits" for i in range(1, 6)]
    denies_cols = [f"{team_prefix}_player_{i}_denies" for i in range(1, 6)]

    hero_damage_cols = [f"{team_prefix}_player_{i}_hero_damage" for i in range(1, 6)]
    tower_damage_cols = [f"{team_prefix}_player_{i}_tower_damage" for i in range(1, 6)]

    df[f"{team_prefix}_avg_kills"] = df[kills_cols].mean(axis=1)
    df[f"{team_prefix}_avg_deaths"] = df[deaths_cols].mean(axis=1)
    df[f"{team_prefix}_avg_assists"] = df[assists_cols].mean(axis=1)
    df[f"{team_prefix}_avg_roshans_killed"] = df[roshans_killed_cols].mean(axis=1)
    df[f"{team_prefix}_avg_last_hits"] = df[last_hits_cols].mean(axis=1)
    df[f"{team_prefix}_avg_denies"] = df[denies_cols].mean(axis=1)
    df[f"{team_prefix}_avg_hero_damage"] = df[hero_damage_cols].mean(axis=1)

    # Team GPM and XPM and Net Worth: Average GPM and XPM per team
    gpm_cols = [f"{team_prefix}_player_{i}_gold_per_min" for i in range(1, 6)]
    xpm_cols = [f"{team_prefix}_player_{i}_xp_per_min" for i in range(1, 6)]
    net_worth_cols = [f"{team_prefix}_player_{i}_net_worth" for i in range(1, 6)]
    player_level_cols = [f"{team_prefix}_player_{i}_level" for i in range(1, 6)]

    df[f"{team_prefix}_avg_gpm"] = df[gpm_cols].mean(axis=1)
    df[f"{team_prefix}_avg_xpm"] = df[xpm_cols].mean(axis=1)
    df[f"{team_prefix}_avg_net_worth"] = df[net_worth_cols].mean(axis=1)
    df[f"{team_prefix}_avg_player_level"] = df[player_level_cols].mean(axis=1)

    # Team OBSERVER and SENTRY: Sum OBSERVER and SENTRY per team
    obs_cols = [f"{team_prefix}_player_{i}_obs_placed" for i in range(1, 6)]
    sen_cols = [f"{team_prefix}_player_{i}_sen_placed" for i in range(1, 6)]

    df[f"{team_prefix}_sum_obs"] = df[obs_cols].sum(axis=1)
    df[f"{team_prefix}_sum_sen"] = df[sen_cols].sum(axis=1)

    # Team teamfight participation: Avg teamfight participation per team
    teamfight_participation_cols = [
        f"{team_prefix}_player_{i}_teamfight_participation" for i in range(1, 6)
    ]

    df[f"{team_prefix}_avg_teamfight_participation_cols"] = df[
        teamfight_participation_cols
    ].mean(axis=1)

    # Drop the original columns used to create these features
    df.drop(
        columns=hero_winrate_cols
        + gpm_cols
        + xpm_cols
        + kills_cols
        + deaths_cols
        + assists_cols
        + obs_cols
        + sen_cols
        + teamfight_participation_cols
        + net_worth_cols
        + roshans_killed_cols
        + last_hits_cols
        + denies_cols
        + hero_damage_cols
        + tower_damage_cols
        + player_level_cols,
        inplace=True,
    )

    return df


def calculate_player_kda(df, team_prefix):
    """
    Function to calculate KDA (Kill-Death-Assist ratio) for each player.
    """
    df[f"{team_prefix}_avg_kda"] = (
        df[f"{team_prefix}_avg_kills"] + df[f"{team_prefix}_avg_assists"]
    ) / df[f"{team_prefix}_avg_deaths"].replace(
        0, 1
    )  # Avoid division by zero
    # Drop kills, deaths, and assists for each player
    df.drop(
        columns=[
            f"{team_prefix}_avg_kills",
            f"{team_prefix}_avg_deaths",
            f"{team_prefix}_avg_assists",
        ],
        inplace=True,
    )

    return df


def prepare_data(df, scaler_file_path="scaler.pkl"):
    # Apply feature engineering for both Radiant and Dire teams
    df = calculate_team_features(df, "radiant")
    df = calculate_team_features(df, "dire")

    # Calculate KDA for each player (for both teams)
    df = calculate_player_kda(df, "radiant")
    df = calculate_player_kda(df, "dire")

    # Create a new column for the match target: 1 if radiant_win is True, else 0
    try:
        df["radiant_win"] = df["radiant_win"].astype(int)
    except KeyError:
        pass

    df.drop(
        columns=[
            "match_id",
            "radiant_team_id",
            "radiant_team_name",
            "dire_team_id",
            "dire_team_name",
            # Drop player names to anonymize data
            *[f"radiant_player_{i}_name" for i in range(1, 6)],
            *[f"radiant_player_{i}_id" for i in range(1, 6)],
            *[f"radiant_player_{i}_hero_id" for i in range(1, 6)],
            *[f"radiant_player_{i}_hero_name" for i in range(1, 6)],
            *[f"dire_player_{i}_name" for i in range(1, 6)],
            *[f"dire_player_{i}_id" for i in range(1, 6)],
            *[f"dire_player_{i}_hero_id" for i in range(1, 6)],
            *[f"dire_player_{i}_hero_name" for i in range(1, 6)],
        ],
        inplace=True,
    )

    columns_to_normalize = df.columns.difference(["match_id", "radiant_win"])

    # Initialize the MinMaxScaler
    if os.path.exists(scaler_file_path):
        scaler = joblib.load(scaler_file_path)
        print(f"Loaded existing scaler from {scaler_file_path}")
    else:
        # Initialize a new MinMaxScaler if no saved state exists
        scaler = MinMaxScaler()
        scaler.fit(df[columns_to_normalize])
        joblib.dump(scaler, scaler_file_path)
        print("No existing scaler found, creating a new one.")

    # Apply Min-Max normalization
    df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
    return df
