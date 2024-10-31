# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import numpy as np

from db.database_operations import (
    insert_match_result,
    update_actual_result,
    get_history_data_as_dataframe,
    convert_to_native_type,
)


class TestDatabaseOperations(unittest.TestCase):

    @patch("db.database_operations.get_database_session")
    def test_insert_match_result(self, mock_get_session):
        # Mocking the session and its methods
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        match_id = 8012600015
        model_prediction = 1
        kwargs = {
            "radiant_avg_hero_winrate": np.float64(0.56),
            "radiant_avg_roshans_killed": np.float64(0.05),
            "radiant_avg_last_hits": np.float64(0.12),
            "radiant_avg_denies": np.float64(0.15),
            "radiant_avg_hero_damage": np.float64(0.34),
            "radiant_avg_gpm": np.float64(0.60),
            "radiant_avg_xpm": np.float64(0.58),
            "radiant_avg_net_worth": np.float64(0.45),
            "radiant_avg_player_level": np.float64(0.48),
            "radiant_sum_obs": np.float64(0.60),
            "radiant_sum_sen": np.float64(0.55),
            "radiant_avg_teamfight_participation_cols": np.float64(0.50),
            "dire_avg_hero_winrate": np.float64(0.45),
            "dire_avg_roshans_killed": np.float64(0.05),
            "dire_avg_last_hits": np.float64(0.10),
            "dire_avg_denies": np.float64(0.18),
            "dire_avg_hero_damage": np.float64(0.20),
            "dire_avg_gpm": np.float64(0.30),
            "dire_avg_xpm": np.float64(0.40),
            "dire_avg_net_worth": np.float64(0.55),
            "dire_avg_player_level": np.float64(0.47),
            "dire_sum_obs": np.float64(0.20),
            "dire_sum_sen": np.float64(0.15),
            "dire_avg_teamfight_participation_cols": np.float64(0.60),
            "radiant_avg_kda": np.float64(0.30),
            "dire_avg_kda": np.float64(0.20),
        }

        # Ensure that the session query returns None (no existing record)
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Call the function
        insert_match_result(match_id, model_prediction, **kwargs)

        # Assertions
        mock_session.add.assert_called_once()
        self.assertEqual(mock_session.add.call_args[0][0].match_id, match_id)
        self.assertEqual(
            mock_session.add.call_args[0][0].model_prediction, model_prediction
        )

    @patch("db.database_operations.get_database_session")
    def test_update_actual_result(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        match_id = 8012600015
        actual_result = 1

        # Mocking the query return value
        mock_record = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_record
        )

        update_actual_result(match_id, actual_result)

        # Assertions
        mock_record.actual_result = actual_result
        mock_session.commit.assert_called_once()

    @patch("db.database_operations.get_database_session")
    def test_get_history_data_as_dataframe(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mocking return value for session query
        mock_record = MagicMock()
        mock_record.__dict__ = {
            "match_id": 8012600015,
            "model_prediction": 1,
            "actual_result": None,
            "timestamp": datetime.utcnow(),
            "radiant_avg_hero_winrate": 0.56,
            "radiant_avg_roshans_killed": 0.05,
            "radiant_avg_last_hits": 0.12,
            "radiant_avg_denies": 0.15,
            "radiant_avg_hero_damage": 0.34,
            "radiant_avg_gpm": 0.60,
            "radiant_avg_xpm": 0.58,
            "radiant_avg_net_worth": 0.45,
            "radiant_avg_player_level": 0.48,
            "radiant_sum_obs": 0.60,
            "radiant_sum_sen": 0.55,
            "radiant_avg_teamfight_participation_cols": 0.50,
            "dire_avg_hero_winrate": 0.45,
            "dire_avg_roshans_killed": 0.05,
            "dire_avg_last_hits": 0.10,
            "dire_avg_denies": 0.18,
            "dire_avg_hero_damage": 0.20,
            "dire_avg_gpm": 0.30,
            "dire_avg_xpm": 0.40,
            "dire_avg_net_worth": 0.55,
            "dire_avg_player_level": 0.47,
            "dire_sum_obs": 0.20,
            "dire_sum_sen": 0.15,
            "dire_avg_teamfight_participation_cols": 0.60,
            "radiant_avg_kda": 0.30,
            "dire_avg_kda": 0.20,
        }

        mock_session.query.return_value.all.return_value = [mock_record]

        df = get_history_data_as_dataframe()

        # Assertions
        self.assertEqual(len(df), 1)  # Expecting 1 row
        self.assertEqual(df["match_id"][0], 8012600015)  # Check match_id

    def test_convert_to_native_type(self):
        # Test converting numpy int64 to int
        self.assertEqual(convert_to_native_type(np.int64(42)), 42)
        self.assertEqual(convert_to_native_type(np.int32(42)), 42)

        # Test converting numpy float64 to float
        self.assertEqual(convert_to_native_type(np.float64(42.0)), 42.0)
        self.assertEqual(convert_to_native_type(np.float32(42.0)), 42.0)

        # Test passing a regular int
        self.assertEqual(convert_to_native_type(42), 42)

        # Test passing a regular float
        self.assertEqual(convert_to_native_type(42.0), 42.0)
