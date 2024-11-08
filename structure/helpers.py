# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import os
import re

import joblib
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def remove_special_chars(text):
    return re.sub(r"[^A-Za-z0-9\s]+", "", text)


def remove_zero_columns(df):
    """
    Remove columns from a DataFrame that match predefined patterns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame with specific columns removed.
    """
    player_attributes = [
        "avg_hero_winrate",
        "avg_teamfight_participation_cols",
        "sum_obs",
        "sum_sen",
        "avg_roshans_killed",
        "avg_hero_damage",
        "avg_tower_damage",
    ]

    # Generate columns for both Radiant and Dire teams (5 players each)
    columns_to_remove = [
        f"{team}_{attr}" for team in ["radiant", "dire"] for attr in player_attributes
    ]
    df_cleaned = df.drop(
        columns=[col for col in columns_to_remove if col in df.columns]
    )

    return df_cleaned


def find_dict_in_list(dicts, key, value):
    logger.info(f"Searching for dictionary with {key} = {value}")
    try:
        result = next(item for item in dicts if item[key] == value)
        logger.info("Dictionary found")
        return result
    except (KeyError, StopIteration):
        logger.warning("Dictionary not found or key error occurred")
        return {"sum": None, "n": 0}


# Feature Engineering Functions
def calculate_team_features(df, team_prefix):
    """
    Function to calculate team-based features for a given prefix (radiant or dire).
    """
    logger.info(f"Calculating team-based features for {team_prefix}")

    # Team Hero Win Rate: Average win rate of the heroes for the team
    hero_winrate_cols = [f"{team_prefix}_player_{i}_hero_winrate" for i in range(1, 6)]
    logger.debug(f"{team_prefix} hero winrate columns: {hero_winrate_cols}")
    df[f"{team_prefix}_avg_hero_winrate"] = df[hero_winrate_cols].mean(axis=1)

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

    logger.debug(f"{team_prefix} kills columns: {kills_cols}")
    logger.debug(f"{team_prefix} deaths columns: {deaths_cols}")
    logger.debug(f"{team_prefix} assists columns: {assists_cols}")

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

    logger.debug(f"{team_prefix} GPM columns: {gpm_cols}")
    logger.debug(f"{team_prefix} XPM columns: {xpm_cols}")
    logger.debug(f"{team_prefix} net worth columns: {net_worth_cols}")

    df[f"{team_prefix}_avg_gpm"] = df[gpm_cols].mean(axis=1)
    df[f"{team_prefix}_avg_xpm"] = df[xpm_cols].mean(axis=1)
    df[f"{team_prefix}_avg_net_worth"] = df[net_worth_cols].mean(axis=1)
    df[f"{team_prefix}_avg_player_level"] = df[player_level_cols].mean(axis=1)

    # Team OBSERVER and SENTRY: Sum OBSERVER and SENTRY per team
    obs_cols = [f"{team_prefix}_player_{i}_obs_placed" for i in range(1, 6)]
    sen_cols = [f"{team_prefix}_player_{i}_sen_placed" for i in range(1, 6)]

    logger.debug(f"{team_prefix} observer columns: {obs_cols}")
    logger.debug(f"{team_prefix} sentry columns: {sen_cols}")

    df[f"{team_prefix}_sum_obs"] = df[obs_cols].sum(axis=1)
    df[f"{team_prefix}_sum_sen"] = df[sen_cols].sum(axis=1)

    # Team teamfight participation: Avg teamfight participation per team
    teamfight_participation_cols = [
        f"{team_prefix}_player_{i}_teamfight_participation" for i in range(1, 6)
    ]
    logger.debug(
        f"{team_prefix} teamfight participation columns: {teamfight_participation_cols}"
    )

    df[f"{team_prefix}_avg_teamfight_participation_cols"] = df[
        teamfight_participation_cols
    ].mean(axis=1)

    # Drop the original columns used to create these features
    logger.info(f"Dropping original columns used to create features for {team_prefix}")
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

    logger.info(f"Feature calculation completed for {team_prefix}")
    return df


def calculate_player_kda(df, team_prefix):
    logger.info(f"Calculating KDA for {team_prefix}")
    try:
        df[f"{team_prefix}_avg_kda"] = (
            df[f"{team_prefix}_avg_kills"] + df[f"{team_prefix}_avg_assists"]
        ) / df[f"{team_prefix}_avg_deaths"].replace(
            0, 1
        )  # Avoid division by zero

        # Dropping columns used to calculate KDA
        df.drop(
            columns=[
                f"{team_prefix}_avg_kills",
                f"{team_prefix}_avg_deaths",
                f"{team_prefix}_avg_assists",
            ],
            inplace=True,
        )
        logger.info(f"KDA calculated and relevant columns dropped for {team_prefix}")

    except Exception as e:
        logger.error(f"Error calculating KDA for {team_prefix}: {e}")

    return df


def prepare_match_prediction_data(df, scaler_file_path="scaler.pkl"):
    logger.info("Preparing match prediction data")
    try:
        df = calculate_team_features(df, "radiant")
        df = calculate_team_features(df, "dire")

        df = calculate_player_kda(df, "radiant")
        df = calculate_player_kda(df, "dire")

        try:
            df["radiant_win"] = df["radiant_win"].astype(int)
        except KeyError:
            logger.warning("radiant_win column missing")

        df.drop(
            columns=[
                "match_id",
                "radiant_team_id",
                "radiant_team_name",
                "dire_team_id",
                "dire_team_name",
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

        if os.path.exists(scaler_file_path):
            scaler = joblib.load(scaler_file_path)
            logger.info(f"Loaded existing scaler from {scaler_file_path}")
        else:
            scaler = MinMaxScaler()
            scaler.fit(df[columns_to_normalize])
            joblib.dump(scaler, scaler_file_path)
            logger.info("No existing scaler found, created and saved a new one")

        df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        logger.info("Normalization applied")

    except Exception as e:
        logger.error(f"Error in prepare_match_prediction_data: {e}")

    return df


def create_hero_features(df, team_prefix):
    logger.info(f"Creating hero features for {team_prefix}")
    try:
        hero_columns = [
            f"{team_prefix}_hero_{i}_{n}_counter_pick"
            for i in range(1, 6)
            for n in range(1, 6)
        ]
        hero_winrate_columns = [
            f"{team_prefix}_player_{i}_hero_winrate" for i in range(1, 6)
        ]

        df[f"{team_prefix}_avg_counter_pick"] = df[hero_columns].mean(axis=1)
        df[f"{team_prefix}_avg_hero_winrate"] = df[hero_winrate_columns].mean(axis=1)

        df.drop(columns=hero_columns + hero_winrate_columns, inplace=True)
        logger.info(f"Hero features created and columns dropped for {team_prefix}")

    except Exception as e:
        logger.error(f"Error in create_hero_features for {team_prefix}: {e}")

    return df


def prepare_hero_pick_data(df):
    logger.info("Preparing hero pick data")
    try:
        df = create_hero_features(df, "radiant")
        df = create_hero_features(df, "dire")

        try:
            df["radiant_win"] = df["radiant_win"].astype(int)
        except KeyError:
            logger.warning("radiant_win column missing")

        df.drop(
            columns=[
                "match_id",
                "radiant_team_id",
                "radiant_team_name",
                "dire_team_id",
                "dire_team_name",
                *[f"radiant_player_{i}_hero_id" for i in range(1, 6)],
                *[f"radiant_player_{i}_hero_name" for i in range(1, 6)],
                *[f"dire_player_{i}_hero_id" for i in range(1, 6)],
                *[f"dire_player_{i}_hero_name" for i in range(1, 6)],
            ],
            inplace=True,
        )
        logger.info("Hero pick data prepared and relevant columns dropped")

    except Exception as e:
        logger.error(f"Error in prepare_hero_pick_data: {e}")

    return df
