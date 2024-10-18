import unittest

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from structure.helpers import (
    calculate_team_features,
    calculate_player_kda,
    prepare_data,
)


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Setting up test data
        data = {
            "match_id": [1],
            "radiant_team_id": [101],
            "radiant_team_name": ["Radiant Team"],
            "dire_team_id": [201],
            "dire_team_name": ["Dire Team"],
            "radiant_win": [1],  # 1 for True (Radiant win)
            # Radiant Team Player Information
            "radiant_player_1_name": ["Player1_Radiant"],
            "radiant_player_2_name": ["Player2_Radiant"],
            "radiant_player_3_name": ["Player3_Radiant"],
            "radiant_player_4_name": ["Player4_Radiant"],
            "radiant_player_5_name": ["Player5_Radiant"],
            "radiant_player_1_id": [10101],
            "radiant_player_2_id": [10102],
            "radiant_player_3_id": [10103],
            "radiant_player_4_id": [10104],
            "radiant_player_5_id": [10105],
            "radiant_player_1_hero_name": ["Hero1"],
            "radiant_player_2_hero_name": ["Hero2"],
            "radiant_player_3_hero_name": ["Hero3"],
            "radiant_player_4_hero_name": ["Hero4"],
            "radiant_player_5_hero_name": ["Hero5"],
            "radiant_player_1_hero_id": [1],
            "radiant_player_2_hero_id": [1],
            "radiant_player_3_hero_id": [1],
            "radiant_player_4_hero_id": [1],
            "radiant_player_5_hero_id": [1],
            "radiant_player_1_level": [10],
            "radiant_player_2_level": [10],
            "radiant_player_3_level": [10],
            "radiant_player_4_level": [10],
            "radiant_player_5_level": [10],
            # Radiant Team Player Statistics
            "radiant_player_1_kills": [5],
            "radiant_player_1_deaths": [2],
            "radiant_player_1_assists": [3],
            "radiant_player_1_gold_per_min": [500],
            "radiant_player_1_xp_per_min": [450],
            "radiant_player_1_net_worth": [12000],
            "radiant_player_1_roshans_killed": [0],
            "radiant_player_1_last_hits": [150],
            "radiant_player_1_denies": [10],
            "radiant_player_1_hero_damage": [12000],
            "radiant_player_1_tower_damage": [2000],
            "radiant_player_1_obs_placed": [3],
            "radiant_player_1_sen_placed": [2],
            "radiant_player_1_teamfight_participation": [0.6],
            "radiant_player_1_hero_winrate": [0.55],  # Added hero winrate for player 1
            "radiant_player_2_kills": [4],
            "radiant_player_2_deaths": [3],
            "radiant_player_2_assists": [6],
            "radiant_player_2_gold_per_min": [550],
            "radiant_player_2_xp_per_min": [480],
            "radiant_player_2_net_worth": [12500],
            "radiant_player_2_roshans_killed": [1],
            "radiant_player_2_last_hits": [140],
            "radiant_player_2_denies": [15],
            "radiant_player_2_hero_damage": [15000],
            "radiant_player_2_tower_damage": [2500],
            "radiant_player_2_obs_placed": [5],
            "radiant_player_2_sen_placed": [3],
            "radiant_player_2_teamfight_participation": [0.8],
            "radiant_player_2_hero_winrate": [0.60],  # Added hero winrate for player 2
            "radiant_player_3_kills": [3],
            "radiant_player_3_deaths": [1],
            "radiant_player_3_assists": [4],
            "radiant_player_3_gold_per_min": [600],
            "radiant_player_3_xp_per_min": [490],
            "radiant_player_3_net_worth": [13000],
            "radiant_player_3_roshans_killed": [0],
            "radiant_player_3_last_hits": [160],
            "radiant_player_3_denies": [12],
            "radiant_player_3_hero_damage": [10000],
            "radiant_player_3_tower_damage": [3000],
            "radiant_player_3_obs_placed": [4],
            "radiant_player_3_sen_placed": [3],
            "radiant_player_3_teamfight_participation": [0.7],
            "radiant_player_3_hero_winrate": [0.45],  # Added hero winrate for player 3
            "radiant_player_4_kills": [2],
            "radiant_player_4_deaths": [4],
            "radiant_player_4_assists": [5],
            "radiant_player_4_gold_per_min": [450],
            "radiant_player_4_xp_per_min": [400],
            "radiant_player_4_net_worth": [11500],
            "radiant_player_4_roshans_killed": [0],
            "radiant_player_4_last_hits": [130],
            "radiant_player_4_denies": [7],
            "radiant_player_4_hero_damage": [9000],
            "radiant_player_4_tower_damage": [1500],
            "radiant_player_4_obs_placed": [2],
            "radiant_player_4_sen_placed": [1],
            "radiant_player_4_teamfight_participation": [0.5],
            "radiant_player_4_hero_winrate": [0.50],  # Added hero winrate for player 4
            "radiant_player_5_kills": [6],
            "radiant_player_5_deaths": [0],
            "radiant_player_5_assists": [8],
            "radiant_player_5_gold_per_min": [700],
            "radiant_player_5_xp_per_min": [550],
            "radiant_player_5_net_worth": [14000],
            "radiant_player_5_roshans_killed": [2],
            "radiant_player_5_last_hits": [170],
            "radiant_player_5_denies": [20],
            "radiant_player_5_hero_damage": [20000],
            "radiant_player_5_tower_damage": [3500],
            "radiant_player_5_obs_placed": [6],
            "radiant_player_5_sen_placed": [4],
            "radiant_player_5_teamfight_participation": [0.9],
            "radiant_player_5_hero_winrate": [0.65],  # Added hero winrate for player 5
            # Dire Team Player Information
            "dire_player_1_name": ["Player1_Dire"],
            "dire_player_2_name": ["Player2_Dire"],
            "dire_player_3_name": ["Player3_Dire"],
            "dire_player_4_name": ["Player4_Dire"],
            "dire_player_5_name": ["Player5_Dire"],
            "dire_player_1_id": [20101],
            "dire_player_2_id": [20102],
            "dire_player_3_id": [20103],
            "dire_player_4_id": [20104],
            "dire_player_5_id": [20105],
            "dire_player_1_hero_name": ["Hero6"],
            "dire_player_2_hero_name": ["Hero7"],
            "dire_player_3_hero_name": ["Hero8"],
            "dire_player_4_hero_name": ["Hero9"],
            "dire_player_5_hero_name": ["Hero10"],
            "dire_player_1_hero_id": [1],
            "dire_player_2_hero_id": [1],
            "dire_player_3_hero_id": [1],
            "dire_player_4_hero_id": [1],
            "dire_player_5_hero_id": [1],
            "dire_player_1_level": [10],
            "dire_player_2_level": [10],
            "dire_player_3_level": [10],
            "dire_player_4_level": [10],
            "dire_player_5_level": [10],
            # Dire Team Player Statistics
            "dire_player_1_kills": [2],
            "dire_player_1_deaths": [5],
            "dire_player_1_assists": [1],
            "dire_player_1_gold_per_min": [400],
            "dire_player_1_xp_per_min": [350],
            "dire_player_1_net_worth": [9000],
            "dire_player_1_roshans_killed": [1],
            "dire_player_1_last_hits": [100],
            "dire_player_1_denies": [5],
            "dire_player_1_hero_damage": [8000],
            "dire_player_1_tower_damage": [1000],
            "dire_player_1_obs_placed": [1],
            "dire_player_1_sen_placed": [1],
            "dire_player_1_teamfight_participation": [0.4],
            "dire_player_1_hero_winrate": [0.40],  # Added hero winrate for player 1
            "dire_player_2_kills": [3],
            "dire_player_2_deaths": [4],
            "dire_player_2_assists": [2],
            "dire_player_2_gold_per_min": [450],
            "dire_player_2_xp_per_min": [370],
            "dire_player_2_net_worth": [8500],
            "dire_player_2_roshans_killed": [0],
            "dire_player_2_last_hits": [90],
            "dire_player_2_denies": [3],
            "dire_player_2_hero_damage": [7000],
            "dire_player_2_tower_damage": [800],
            "dire_player_2_obs_placed": [1],
            "dire_player_2_sen_placed": [0],
            "dire_player_2_teamfight_participation": [0.5],
            "dire_player_2_hero_winrate": [0.35],  # Added hero winrate for player 2
            "dire_player_3_kills": [1],
            "dire_player_3_deaths": [6],
            "dire_player_3_assists": [0],
            "dire_player_3_gold_per_min": [350],
            "dire_player_3_xp_per_min": [330],
            "dire_player_3_net_worth": [7800],
            "dire_player_3_roshans_killed": [0],
            "dire_player_3_last_hits": [80],
            "dire_player_3_denies": [1],
            "dire_player_3_hero_damage": [4000],
            "dire_player_3_tower_damage": [600],
            "dire_player_3_obs_placed": [0],
            "dire_player_3_sen_placed": [0],
            "dire_player_3_teamfight_participation": [0.2],
            "dire_player_3_hero_winrate": [0.30],  # Added hero winrate for player 3
            "dire_player_4_kills": [2],
            "dire_player_4_deaths": [3],
            "dire_player_4_assists": [1],
            "dire_player_4_gold_per_min": [420],
            "dire_player_4_xp_per_min": [390],
            "dire_player_4_net_worth": [8200],
            "dire_player_4_roshans_killed": [0],
            "dire_player_4_last_hits": [70],
            "dire_player_4_denies": [2],
            "dire_player_4_hero_damage": [5000],
            "dire_player_4_tower_damage": [800],
            "dire_player_4_obs_placed": [1],
            "dire_player_4_sen_placed": [0],
            "dire_player_4_teamfight_participation": [0.2],
            "dire_player_4_hero_winrate": [0.36],  # Added hero winrate for player 4
            "dire_player_5_kills": [4],
            "dire_player_5_deaths": [1],
            "dire_player_5_assists": [5],
            "dire_player_5_gold_per_min": [600],
            "dire_player_5_xp_per_min": [460],
            "dire_player_5_net_worth": [10800],
            "dire_player_5_roshans_killed": [1],
            "dire_player_5_last_hits": [120],
            "dire_player_5_denies": [6],
            "dire_player_5_hero_damage": [9000],
            "dire_player_5_tower_damage": [2000],
            "dire_player_5_obs_placed": [4],
            "dire_player_5_sen_placed": [2],
            "dire_player_5_teamfight_participation": [0.6],
            "dire_player_5_hero_winrate": [0.50],  # Added hero winrate for player 5
            # Additional statistics can be added here as needed, like match duration, total gold, etc.
        }

        self.df = pd.DataFrame(data)

    def test_calculate_team_features(self):
        df_radiant = calculate_team_features(self.df.copy(), "radiant")
        df_dire = calculate_team_features(self.df.copy(), "dire")

        # Check for new features
        self.assertIn("radiant_avg_hero_winrate", df_radiant.columns)
        self.assertIn("radiant_avg_kills", df_radiant.columns)
        self.assertIn("radiant_avg_last_hits", df_radiant.columns)
        self.assertIn("radiant_avg_roshans_killed", df_radiant.columns)
        self.assertIn("radiant_avg_hero_damage", df_radiant.columns)

        self.assertIn("dire_avg_hero_winrate", df_dire.columns)
        self.assertIn("dire_avg_kills", df_dire.columns)
        self.assertIn("dire_avg_last_hits", df_dire.columns)
        self.assertIn("dire_avg_roshans_killed", df_dire.columns)
        self.assertIn("dire_avg_hero_damage", df_dire.columns)

        # Validate values for Radiant team
        self.assertEqual(df_radiant["radiant_avg_kills"].iloc[0], 4)
        self.assertEqual(df_radiant["radiant_avg_roshans_killed"].iloc[0], 0.6)
        self.assertEqual(df_radiant["radiant_avg_hero_damage"].iloc[0], 13200)

        # Validate values for Dire team
        self.assertEqual(df_dire["dire_avg_kills"].iloc[0], 2.4)
        self.assertEqual(df_dire["dire_avg_roshans_killed"].iloc[0], 0.4)
        self.assertEqual(df_dire["dire_avg_hero_damage"].iloc[0], 6600)

    def test_calculate_player_kda(self):
        radiant_df_with_kda = calculate_player_kda(
            calculate_team_features(self.df.copy(), "radiant"), "radiant"
        )
        dire_df_with_kda = calculate_player_kda(
            calculate_team_features(self.df.copy(), "dire"), "dire"
        )

        # Check for KDA columns
        self.assertIn("radiant_avg_kda", radiant_df_with_kda.columns)
        self.assertIn("dire_avg_kda", dire_df_with_kda.columns)

        # Validate KDA values for Radiant team
        self.assertAlmostEqual(
            radiant_df_with_kda["radiant_avg_kda"].iloc[0], 4.6, places=1
        )

        # Validate KDA values for Dire team
        self.assertAlmostEqual(dire_df_with_kda["dire_avg_kda"].iloc[0], 1.1, places=1)

    def test_prepare_data(self):
        prepared_df = prepare_data(self.df.copy(), "test_scaler.pkl")

        # Check for expected columns after preparation
        expected_columns = [
            "radiant_win",
            "radiant_avg_hero_winrate",
            "radiant_avg_roshans_killed",
            "radiant_avg_last_hits",
            "radiant_avg_denies",
            "radiant_avg_hero_damage",
            "radiant_avg_player_level",
            "radiant_avg_gpm",
            "radiant_avg_xpm",
            "radiant_avg_net_worth",
            "radiant_sum_obs",
            "radiant_sum_sen",
            "radiant_avg_teamfight_participation_cols",
            "dire_avg_hero_winrate",
            "dire_avg_roshans_killed",
            "dire_avg_last_hits",
            "dire_avg_denies",
            "dire_avg_hero_damage",
            "dire_avg_player_level",
            "dire_avg_gpm",
            "dire_avg_xpm",
            "dire_avg_net_worth",
            "dire_sum_obs",
            "dire_sum_sen",
            "dire_avg_teamfight_participation_cols",
            "radiant_avg_kda",
            "dire_avg_kda",
        ]
        self.assertEqual(len(expected_columns), len(prepared_df.columns))
        for col in expected_columns:
            self.assertIn(col, prepared_df.columns)

        # Check that the radiant_win column is properly transformed
        self.assertEqual(
            prepared_df["radiant_win"].iloc[0], 1
        )  # True should be converted to 1

        # Validate the normalization of features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(
            prepared_df.drop(columns=["radiant_win"])
        )

        # Check if features are indeed normalized
        for i in range(1, len(scaled_features[0])):
            self.assertGreaterEqual(scaled_features[0][i], 0)
            self.assertLessEqual(scaled_features[0][i], 1)


if __name__ == "__main__":
    unittest.main()
