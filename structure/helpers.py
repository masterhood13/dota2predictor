# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

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

    # Team Player Win Rate: Average win rate of the players for the team
    player_winrate_cols = [f"{team_prefix}_player_{i}_winrate" for i in range(1, 6)]
    df[f"{team_prefix}_avg_player_winrate"] = df[player_winrate_cols].mean(axis=1)

    # Team Kills, Deaths, Assists
    kills_cols = [f"{team_prefix}_player_{i}_kills" for i in range(1, 6)]
    deaths_cols = [f"{team_prefix}_player_{i}_deaths" for i in range(1, 6)]
    assists_cols = [f"{team_prefix}_player_{i}_assists" for i in range(1, 6)]

    df[f"{team_prefix}_total_kills"] = df[kills_cols].sum(axis=1)
    df[f"{team_prefix}_total_deaths"] = df[deaths_cols].sum(axis=1)
    df[f"{team_prefix}_total_assists"] = df[assists_cols].sum(axis=1)

    # Team GPM and XPM: Average GPM and XPM per team
    gpm_cols = [f"{team_prefix}_player_{i}_gold_per_min" for i in range(1, 6)]
    xpm_cols = [f"{team_prefix}_player_{i}_xp_per_min" for i in range(1, 6)]

    df[f"{team_prefix}_avg_gpm"] = df[gpm_cols].mean(axis=1)
    df[f"{team_prefix}_avg_xpm"] = df[xpm_cols].mean(axis=1)

    # Drop the original columns used to create these features
    df.drop(
        columns=hero_winrate_cols + player_winrate_cols + gpm_cols + xpm_cols,
        inplace=True,
    )

    return df


def calculate_player_kda(df, team_prefix):
    """
    Function to calculate KDA (Kill-Death-Assist ratio) for each player.
    """
    for i in range(1, 6):
        df[f"{team_prefix}_player_{i}_kda"] = (
            df[f"{team_prefix}_player_{i}_kills"]
            + df[f"{team_prefix}_player_{i}_assists"]
        ) / df[f"{team_prefix}_player_{i}_deaths"].replace(
            0, 1
        )  # Avoid division by zero
        # Drop kills, deaths, and assists for each player
        df.drop(
            columns=[
                f"{team_prefix}_player_{i}_kills",
                f"{team_prefix}_player_{i}_deaths",
                f"{team_prefix}_player_{i}_assists",
            ],
            inplace=True,
        )

    return df


def prepare_data(df):
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
            *[f"radiant_player_{i}_hero_name" for i in range(1, 6)],
            *[f"dire_player_{i}_name" for i in range(1, 6)],
            *[f"dire_player_{i}_id" for i in range(1, 6)],
            *[f"dire_player_{i}_hero_name" for i in range(1, 6)],
        ],
        inplace=True,
    )

    columns_to_normalize = df.columns.difference(["match_id", "radiant_win"])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Apply Min-Max normalization
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df
