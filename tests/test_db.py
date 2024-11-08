# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from db.database_operations import (
    insert_match_result,
    update_actual_result,
    get_history_data_as_dataframe,
    convert_to_native_type,
    fetch_and_update_actual_results,
    calculate_win_rate,
    get_current_last_trained_row_id,
    update_or_create_last_trained_row_id,
)
from db.setup import History, ModelTrainingMetadata


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

    @patch("db.database_operations.get_database_session")
    @patch("db.database_operations.requests.get")
    def test_fetch_and_update_actual_results(self, mock_get, mock_get_session):
        # Mocking the session and its methods
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Prepare mock data
        match_to_update = History(match_id=12345, actual_result=None)
        mock_session.query.return_value.filter.return_value.all.return_value = [
            match_to_update
        ]

        # Mocking the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"radiant_win": True}
        mock_get.return_value = mock_response

        # Call the function
        fetch_and_update_actual_results()

        # Assertions
        mock_session.commit.assert_called_once()
        self.assertEqual(
            match_to_update.actual_result, 1
        )  # Assuming True translates to 1

    @patch("db.database_operations.get_database_session")
    def test_fetch_and_update_no_matches(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mocking no entries with actual_result as None
        mock_session.query.return_value.filter.return_value.all.return_value = []

        # Call the function
        fetch_and_update_actual_results()

        # Assertions
        mock_session.commit.assert_not_called()  # No commit should happen

    @patch("db.database_operations.get_database_session")
    @patch("db.database_operations.requests.get")
    def test_fetch_and_update_api_failure(self, mock_get, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        match_to_update = History(match_id=12345, actual_result=None)
        mock_session.query.return_value.filter.return_value.all.return_value = [
            match_to_update
        ]

        # Mocking the API response with failure
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Call the function
        fetch_and_update_actual_results()

        # Assertions
        mock_session.commit.assert_not_called()  # No commit should happen
        self.assertIsNone(
            match_to_update.actual_result
        )  # actual_result should remain None

    @patch("db.database_operations.get_database_session")
    @patch("db.database_operations.requests.get")
    def test_fetch_and_update_no_actual_result_in_response(
        self, mock_get, mock_get_session
    ):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        match_to_update = History(match_id=12345, actual_result=None)
        mock_session.query.return_value.filter.return_value.all.return_value = [
            match_to_update
        ]

        # Mocking the API response with no actual result
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No "radiant_win" field
        mock_get.return_value = mock_response

        # Call the function
        fetch_and_update_actual_results()

        # Assertions
        mock_session.commit.assert_not_called()  # No commit should happen
        self.assertIsNone(
            match_to_update.actual_result
        )  # actual_result should remain None

    @patch("db.database_operations.get_database_session")
    def test_calculate_win_rate(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Prepare mock data
        history_data = [
            History(actual_result=1, model_prediction=1),  # Correct
            History(actual_result=0, model_prediction=1),  # Incorrect
            History(actual_result=1, model_prediction=0),  # Incorrect
            History(actual_result=1, model_prediction=1),  # Correct
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = (
            history_data
        )

        # Call the function
        win_rate, total_predictions = calculate_win_rate()

        # Assertions
        self.assertEqual(total_predictions, 4)
        self.assertEqual(win_rate, 0.5)  # 2 out of 4 are correct

    @patch("db.database_operations.get_database_session")
    def test_calculate_win_rate_no_predictions(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mocking no predictions
        mock_session.query.return_value.filter.return_value.all.return_value = []

        # Call the function
        win_rate, total_predictions = calculate_win_rate()

        # Assertions
        self.assertEqual(total_predictions, 0)
        self.assertEqual(win_rate, 0)

    @patch("db.database_operations.get_database_session")
    def test_calculate_win_rate_database_error(self, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mocking a database error
        mock_session.query.side_effect = SQLAlchemyError("Database error")

        # Call the function
        win_rate, total_predictions = calculate_win_rate()

        # Assertions
        self.assertEqual(total_predictions, 0)
        self.assertIsNone(win_rate)

    @patch("db.database_operations.get_database_session")
    def test_update_or_create_new_entry(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query(ModelTrainingMetadata).first.return_value = None

        # Call the function to test
        new_row_id = 123
        update_or_create_last_trained_row_id(new_row_id)

        # Verify a new entry was created
        mock_session.add.assert_called_once()
        metadata_entry = mock_session.add.call_args[0][0]
        self.assertEqual(metadata_entry.last_trained_row_id, new_row_id)
        mock_session.commit.assert_called_once()

    @patch("db.database_operations.get_database_session")
    def test_update_existing_entry(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Create a mock metadata entry with an existing row ID
        mock_metadata_entry = ModelTrainingMetadata(last_trained_row_id=100)
        mock_session.query(ModelTrainingMetadata).first.return_value = (
            mock_metadata_entry
        )

        # Call the function to test
        new_row_id = 456
        update_or_create_last_trained_row_id(new_row_id)

        # Verify the existing entry was updated
        self.assertEqual(mock_metadata_entry.last_trained_row_id, new_row_id)
        mock_session.commit.assert_called_once()

    @patch("db.database_operations.get_database_session")
    def test_get_existing_last_trained_row_id(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Mock a metadata entry with a specific row ID
        mock_metadata_entry = ModelTrainingMetadata(last_trained_row_id=789)
        mock_session.query(ModelTrainingMetadata).first.return_value = (
            mock_metadata_entry
        )

        # Call the function to test
        result = get_current_last_trained_row_id()

        # Verify the correct row ID was returned
        self.assertEqual(result, 789)

    @patch("db.database_operations.get_database_session")
    def test_get_no_last_trained_row_id(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Return None for no existing entry
        mock_session.query(ModelTrainingMetadata).first.return_value = None

        # Call the function to test
        result = get_current_last_trained_row_id()

        # Verify that 0 is returned when no metadata is found
        self.assertEqual(result, 0)

    @patch("db.database_operations.get_database_session")
    def test_update_or_create_exception_handling(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Simulate an exception on commit
        mock_session.commit.side_effect = Exception("Database error")

        # Call the function to test
        update_or_create_last_trained_row_id(123)

        # Verify rollback was called
        mock_session.rollback.assert_called_once()

    @patch("db.database_operations.get_database_session")
    def test_get_current_last_trained_row_id_exception_handling(self, mock_get_session):
        # Mock the session and its behavior
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Simulate an exception during query
        mock_session.query(ModelTrainingMetadata).first.side_effect = Exception(
            "Database error"
        )

        # Call the function to test
        result = get_current_last_trained_row_id()

        # Verify that 0 is returned on exception
        self.assertEqual(result, 0)
