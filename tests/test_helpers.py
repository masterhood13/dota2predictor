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
        data = {
            "match_id": [1],
            "radiant_team_id": [101],
            "radiant_team_name": ["Radiant Team"],
            "dire_team_id": [201],
            "dire_team_name": ["Dire Team"],
            "radiant_win": [1],  # 1 for True (Radiant win)
            "radiant_player_1_id": [1],
            "radiant_player_1_name": ["Player1"],
            "radiant_player_1_hero_id": [301],
            "radiant_player_1_hero_name": ["HeroA"],
            "radiant_player_1_hero_winrate": [0.6],
            "radiant_player_1_winrate": [0.7],
            "radiant_player_1_kills": [5],
            "radiant_player_1_deaths": [2],
            "radiant_player_1_assists": [3],
            "radiant_player_1_gold_per_min": [500],
            "radiant_player_1_xp_per_min": [450],
            "radiant_player_2_id": [2],
            "radiant_player_2_name": ["Player2"],
            "radiant_player_2_hero_id": [302],
            "radiant_player_2_hero_name": ["HeroB"],
            "radiant_player_2_hero_winrate": [0.65],
            "radiant_player_2_winrate": [0.75],
            "radiant_player_2_kills": [6],
            "radiant_player_2_deaths": [1],
            "radiant_player_2_assists": [2],
            "radiant_player_2_gold_per_min": [600],
            "radiant_player_2_xp_per_min": [500],
            "radiant_player_3_id": [3],
            "radiant_player_3_name": ["Player3"],
            "radiant_player_3_hero_id": [303],
            "radiant_player_3_hero_name": ["HeroC"],
            "radiant_player_3_hero_winrate": [0.7],
            "radiant_player_3_winrate": [0.6],
            "radiant_player_3_kills": [7],
            "radiant_player_3_deaths": [2],
            "radiant_player_3_assists": [4],
            "radiant_player_3_gold_per_min": [550],
            "radiant_player_3_xp_per_min": [460],
            "radiant_player_4_id": [4],
            "radiant_player_4_name": ["Player4"],
            "radiant_player_4_hero_id": [304],
            "radiant_player_4_hero_name": ["HeroD"],
            "radiant_player_4_hero_winrate": [0.55],
            "radiant_player_4_winrate": [0.65],
            "radiant_player_4_kills": [2],
            "radiant_player_4_deaths": [3],
            "radiant_player_4_assists": [1],
            "radiant_player_4_gold_per_min": [450],
            "radiant_player_4_xp_per_min": [400],
            "radiant_player_5_id": [5],
            "radiant_player_5_name": ["Player5"],
            "radiant_player_5_hero_id": [305],
            "radiant_player_5_hero_name": ["HeroE"],
            "radiant_player_5_hero_winrate": [0.5],
            "radiant_player_5_winrate": [0.55],
            "radiant_player_5_kills": [3],
            "radiant_player_5_deaths": [1],
            "radiant_player_5_assists": [2],
            "radiant_player_5_gold_per_min": [480],
            "radiant_player_5_xp_per_min": [420],
            "dire_player_1_id": [6],
            "dire_player_1_name": ["DirePlayer1"],
            "dire_player_1_hero_id": [306],
            "dire_player_1_hero_name": ["DireHeroA"],
            "dire_player_1_hero_winrate": [0.4],
            "dire_player_1_winrate": [0.45],
            "dire_player_1_kills": [2],
            "dire_player_1_deaths": [5],
            "dire_player_1_assists": [1],
            "dire_player_1_gold_per_min": [400],
            "dire_player_1_xp_per_min": [350],
            "dire_player_2_id": [7],
            "dire_player_2_name": ["DirePlayer2"],
            "dire_player_2_hero_id": [307],
            "dire_player_2_hero_name": ["DireHeroB"],
            "dire_player_2_hero_winrate": [0.35],
            "dire_player_2_winrate": [0.55],
            "dire_player_2_kills": [3],
            "dire_player_2_deaths": [3],
            "dire_player_2_assists": [0],
            "dire_player_2_gold_per_min": [420],
            "dire_player_2_xp_per_min": [370],
            "dire_player_3_id": [8],
            "dire_player_3_name": ["DirePlayer3"],
            "dire_player_3_hero_id": [308],
            "dire_player_3_hero_name": ["DireHeroC"],
            "dire_player_3_hero_winrate": [0.6],
            "dire_player_3_winrate": [0.6],
            "dire_player_3_kills": [4],
            "dire_player_3_deaths": [2],
            "dire_player_3_assists": [3],
            "dire_player_3_gold_per_min": [490],
            "dire_player_3_xp_per_min": [410],
            "dire_player_4_id": [9],
            "dire_player_4_name": ["DirePlayer4"],
            "dire_player_4_hero_id": [309],
            "dire_player_4_hero_name": ["DireHeroD"],
            "dire_player_4_hero_winrate": [0.45],
            "dire_player_4_winrate": [0.5],
            "dire_player_4_kills": [1],
            "dire_player_4_deaths": [4],
            "dire_player_4_assists": [1],
            "dire_player_4_gold_per_min": [430],
            "dire_player_4_xp_per_min": [380],
            "dire_player_5_id": [10],
            "dire_player_5_name": ["DirePlayer5"],
            "dire_player_5_hero_id": [310],
            "dire_player_5_hero_name": ["DireHeroE"],
            "dire_player_5_hero_winrate": [0.55],
            "dire_player_5_winrate": [0.65],
            "dire_player_5_kills": [0],
            "dire_player_5_deaths": [2],
            "dire_player_5_assists": [2],
            "dire_player_5_gold_per_min": [460],
            "dire_player_5_xp_per_min": [400],
        }

        self.df = pd.DataFrame(data)

    def test_calculate_team_features(self):
        df_radiant = calculate_team_features(self.df.copy(), "radiant")
        df_dire = calculate_team_features(self.df.copy(), "dire")

        # Check for new features
        self.assertIn("radiant_avg_hero_winrate", df_radiant.columns)
        self.assertIn("radiant_avg_player_winrate", df_radiant.columns)
        self.assertIn("radiant_total_kills", df_radiant.columns)
        self.assertIn("radiant_total_deaths", df_radiant.columns)
        self.assertIn("radiant_total_assists", df_radiant.columns)
        self.assertIn("radiant_avg_gpm", df_radiant.columns)
        self.assertIn("radiant_avg_xpm", df_radiant.columns)

        self.assertIn("dire_avg_hero_winrate", df_dire.columns)
        self.assertIn("dire_avg_player_winrate", df_dire.columns)
        self.assertIn("dire_total_kills", df_dire.columns)
        self.assertIn("dire_total_deaths", df_dire.columns)
        self.assertIn("dire_total_assists", df_dire.columns)
        self.assertIn("dire_avg_gpm", df_dire.columns)
        self.assertIn("dire_avg_xpm", df_dire.columns)

        # Validate values for Radiant team
        self.assertAlmostEqual(
            df_radiant["radiant_avg_hero_winrate"].iloc[0], 0.6, places=1
        )
        self.assertAlmostEqual(
            df_radiant["radiant_avg_player_winrate"].iloc[0], 0.65, places=1
        )
        self.assertEqual(df_radiant["radiant_total_kills"].iloc[0], 23)
        self.assertEqual(df_radiant["radiant_total_deaths"].iloc[0], 9)
        self.assertEqual(df_radiant["radiant_total_assists"].iloc[0], 12)

        # Validate values for Dire team
        self.assertAlmostEqual(df_dire["dire_avg_hero_winrate"].iloc[0], 0.47, places=1)
        self.assertAlmostEqual(
            df_dire["dire_avg_player_winrate"].iloc[0], 0.55, places=1
        )
        self.assertEqual(df_dire["dire_total_kills"].iloc[0], 10)
        self.assertEqual(df_dire["dire_total_deaths"].iloc[0], 16)
        self.assertEqual(df_dire["dire_total_assists"].iloc[0], 7)

    def test_calculate_player_kda(self):
        df_with_kda = calculate_player_kda(self.df.copy(), "radiant")
        df_with_kda = calculate_player_kda(df_with_kda, "dire")

        # Check for KDA columns
        self.assertIn("radiant_player_1_kda", df_with_kda.columns)
        self.assertIn("dire_player_1_kda", df_with_kda.columns)

        # Validate KDA values for Radiant team
        self.assertAlmostEqual(
            df_with_kda["radiant_player_1_kda"].iloc[0], 4.0, places=1
        )
        self.assertAlmostEqual(
            df_with_kda["radiant_player_2_kda"].iloc[0], 8.0, places=1
        )
        self.assertAlmostEqual(
            df_with_kda["radiant_player_3_kda"].iloc[0], 5.5, places=1
        )
        self.assertAlmostEqual(
            df_with_kda["radiant_player_4_kda"].iloc[0], 1.0, places=1
        )
        self.assertAlmostEqual(
            df_with_kda["radiant_player_5_kda"].iloc[0], 5.0, places=1
        )

        # Validate KDA values for Dire team
        self.assertAlmostEqual(df_with_kda["dire_player_1_kda"].iloc[0], 0.6, places=1)
        self.assertAlmostEqual(df_with_kda["dire_player_2_kda"].iloc[0], 1.0, places=2)
        self.assertAlmostEqual(df_with_kda["dire_player_3_kda"].iloc[0], 3.5, places=1)
        self.assertAlmostEqual(df_with_kda["dire_player_4_kda"].iloc[0], 0.5, places=2)
        self.assertAlmostEqual(df_with_kda["dire_player_5_kda"].iloc[0], 1.0, places=1)

    def test_prepare_data(self):
        prepared_df = prepare_data(self.df.copy())

        # Check for expected columns after preparation
        expected_columns = [
            "radiant_avg_hero_winrate",
            "radiant_avg_player_winrate",
            "radiant_total_kills",
            "radiant_total_deaths",
            "radiant_total_assists",
            "dire_avg_hero_winrate",
            "dire_avg_player_winrate",
            "dire_total_kills",
            "dire_total_deaths",
            "dire_total_assists",
            "radiant_win",
        ]

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
