import unittest

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from structure.helpers import (
    calculate_team_features,
    calculate_player_kda,
    prepare_match_prediction_data,
    find_dict_in_list,
    prepare_hero_pick_data,
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

        data_hero_pick = {
            "match_id": [7484575133],
            "radiant_team_id": [9247354],
            "radiant_team_name": ["Team Falcons"],
            "dire_team_id": [7422789],
            "dire_team_name": ["9Pandas"],
            "radiant_win": [True],
            # Radiant Player 1
            "radiant_player_1_hero_id": [113],
            "radiant_player_1_hero_name": ["Arc Warden"],
            "radiant_player_1_hero_winrate": [0.5087719298245614],
            "radiant_hero_1_1_counter_pick": [0.3888888888888889],
            "radiant_hero_1_2_counter_pick": [0.2777777777777778],
            "radiant_hero_1_3_counter_pick": [0.3888888888888889],
            "radiant_hero_1_4_counter_pick": [0.3333333333333333],
            "radiant_hero_1_5_counter_pick": [0.5454545454545454],
            # Radiant Player 2
            "radiant_player_2_hero_id": [54],
            "radiant_player_2_hero_name": ["Lifestealer"],
            "radiant_player_2_hero_winrate": [0.4595744680851064],
            "radiant_hero_2_1_counter_pick": [0.5675675675675675],
            "radiant_hero_2_2_counter_pick": [0.4507042253521127],
            "radiant_hero_2_3_counter_pick": [0.5384615384615384],
            "radiant_hero_2_4_counter_pick": [0.39285714285714285],
            "radiant_hero_2_5_counter_pick": [0.3877551020408163],
            # Radiant Player 3
            "radiant_player_3_hero_id": [13],
            "radiant_player_3_hero_name": ["Puck"],
            "radiant_player_3_hero_winrate": [0.5209125475285171],
            "radiant_hero_3_1_counter_pick": [0.4322033898305085],
            "radiant_hero_3_2_counter_pick": [0.5833333333333334],
            "radiant_hero_3_3_counter_pick": [0.4142857142857143],
            "radiant_hero_3_4_counter_pick": [0.5],
            "radiant_hero_3_5_counter_pick": [0.3939393939393939],
            # Radiant Player 4
            "radiant_player_4_hero_id": [4],
            "radiant_player_4_hero_name": ["Bloodseeker"],
            "radiant_player_4_hero_winrate": [0.5949367088607594],
            "radiant_hero_4_1_counter_pick": [0.4318181818181818],
            "radiant_hero_4_2_counter_pick": [0.53125],
            "radiant_hero_4_3_counter_pick": [0.42857142857142855],
            "radiant_hero_4_4_counter_pick": [0.6],
            "radiant_hero_4_5_counter_pick": [0.47368421052631576],
            # Radiant Player 5
            "radiant_player_5_hero_id": [26],
            "radiant_player_5_hero_name": ["Lion"],
            "radiant_player_5_hero_winrate": [0.4669421487603306],
            "radiant_hero_5_1_counter_pick": [0.3931034482758621],
            "radiant_hero_5_2_counter_pick": [0.4028776978417266],
            "radiant_hero_5_3_counter_pick": [0.46956521739130436],
            "radiant_hero_5_4_counter_pick": [0.46788990825688076],
            "radiant_hero_5_5_counter_pick": [0.5301204819277109],
            # Dire Player 1
            "dire_player_1_hero_id": [43],
            "dire_player_1_hero_name": ["Death Prophet"],
            "dire_player_1_hero_winrate": [0.4017094017094017],
            "dire_hero_1_1_counter_pick": [0.5899280575539568],
            "dire_hero_1_2_counter_pick": [0.5909090909090909],
            "dire_hero_1_3_counter_pick": [0.5892857142857143],
            "dire_hero_1_4_counter_pick": [0.4375],
            "dire_hero_1_5_counter_pick": [0.5555555555555556],
            # Dire Player 2
            "dire_player_2_hero_id": [102],
            "dire_player_2_hero_name": ["Abaddon"],
            "dire_player_2_hero_winrate": [0.5133689839572193],
            "dire_hero_2_1_counter_pick": [0.6],
            "dire_hero_2_2_counter_pick": [0.5352112676056338],
            "dire_hero_2_3_counter_pick": [0.5714285714285714],
            "dire_hero_2_4_counter_pick": [0.5238095238095238],
            "dire_hero_2_5_counter_pick": [0.5555555555555556],
            # Dire Player 3
            "dire_player_3_hero_id": [138],
            "dire_player_3_hero_name": ["Muerta"],
            "dire_player_3_hero_winrate": [0.49714285714285716],
            "dire_hero_3_1_counter_pick": [0.559322033898305],
            "dire_hero_3_2_counter_pick": [0.42342342342342343],
            "dire_hero_3_3_counter_pick": [0.5229357798165137],
            "dire_hero_3_4_counter_pick": [0.5454545454545454],
            "dire_hero_3_5_counter_pick": [0.36363636363636365],
            # Dire Player 4
            "dire_player_4_hero_id": [76],
            "dire_player_4_hero_name": ["Outworld Devourer"],
            "dire_player_4_hero_winrate": [0.49324324324324326],
            "dire_hero_4_1_counter_pick": [0.5217391304347826],
            "dire_hero_4_2_counter_pick": [0.48484848484848486],
            "dire_hero_4_3_counter_pick": [0.4461538461538462],
            "dire_hero_4_4_counter_pick": [0.35],
            "dire_hero_4_5_counter_pick": [0.6],
            # Dire Player 5
            "dire_player_5_hero_id": [50],
            "dire_player_5_hero_name": ["Dazzle"],
            "dire_player_5_hero_winrate": [0.4397163120567376],
            "dire_hero_5_1_counter_pick": [0.40625],
            "dire_hero_5_2_counter_pick": [0.4578313253012048],
            "dire_hero_5_3_counter_pick": [0.5918367346938775],
            "dire_hero_5_4_counter_pick": [0.47368421052631576],
            "dire_hero_5_5_counter_pick": [0.6666666666666666],
        }

        self.df = pd.DataFrame(data)

        self.df_hero_pick = pd.DataFrame(data_hero_pick)

    def test_find_dict_in_list(self):
        dicts = [{"id": 1, "sum": 100, "n": 5}, {"id": 2, "sum": 200, "n": 3}]

        # Test when key-value pair is found
        result = find_dict_in_list(dicts, "id", 1)
        self.assertEqual(result, {"id": 1, "sum": 100, "n": 5})

        # Test when key-value pair is not found
        result = find_dict_in_list(dicts, "id", 3)
        self.assertEqual(result, {"sum": None, "n": 0})

        # Test when key doesn't exist
        result = find_dict_in_list(dicts, "unknown_key", 1)
        self.assertEqual(result, {"sum": None, "n": 0})

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
        prepared_df = prepare_match_prediction_data(self.df.copy(), "test_scaler.pkl")

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

    def test_prepare_hero_pick_data(self):
        df_prepared = prepare_hero_pick_data(self.df_hero_pick.copy())

        # Check if the new aggregated columns are created
        self.assertIn("radiant_avg_counter_pick", df_prepared.columns)
        self.assertIn("dire_avg_counter_pick", df_prepared.columns)

        # Check that player hero columns are dropped
        for i in range(1, 6):
            self.assertNotIn(f"radiant_player_{i}_hero_id", df_prepared.columns)
            self.assertNotIn(f"dire_player_{i}_hero_id", df_prepared.columns)

        # Check if radiant_win is properly handled
        self.assertEqual(df_prepared["radiant_win"].iloc[0], 1)

    def test_prepare_data_no_radiant_win(self):
        df_no_win = self.df.drop(columns=["radiant_win"]).copy()
        prepared_df = prepare_match_prediction_data(df_no_win, "test_scaler.pkl")

        # Check if radiant_win column is added correctly
        self.assertNotIn("radiant_win", prepared_df.columns)

    def test_prepare_hero_pick_data_no_radiant_win(self):
        df_no_win = self.df_hero_pick.drop(columns=["radiant_win"]).copy()
        df_prepared = prepare_hero_pick_data(df_no_win)

        # Check if radiant_win column is added correctly
        self.assertNotIn("radiant_win", df_prepared.columns)

    def test_calculate_team_features_with_extreme_values(self):
        df_extreme = self.df.copy()
        # Assign extreme values
        df_extreme["radiant_player_1_kills"] = 0
        df_extreme["radiant_player_1_deaths"] = 0
        df_extreme["radiant_player_1_assists"] = 100

        df_radiant = calculate_team_features(df_extreme, "radiant")

        # Validate that the averages are calculated correctly
        self.assertEqual(df_radiant["radiant_avg_kills"].iloc[0], 3)
        self.assertEqual(df_radiant["radiant_avg_assists"].iloc[0], 24.6)
        self.assertEqual(df_radiant["radiant_avg_deaths"].iloc[0], 1.6)
