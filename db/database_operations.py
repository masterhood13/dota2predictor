# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from config import DATABASE_CONFIG, opendota_key
from db.setup import History


logger = logging.getLogger(__name__)


def get_database_session():
    """Creates and returns a database session."""
    # Create the database URL
    db_url = f"postgresql://{DATABASE_CONFIG['USER']}:{DATABASE_CONFIG['PASSWORD']}@{DATABASE_CONFIG['HOST']}:{DATABASE_CONFIG['PORT']}/{DATABASE_CONFIG['DB_NAME']}"

    logger.info("Creating a new database session...")
    # Create the database engine
    engine = create_engine(db_url)

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a session
    session = Session()
    logger.info("Database session created successfully.")
    return session


def convert_to_native_type(value):
    """Convert numpy data types to native Python data types for compatibility with SQLAlchemy."""
    if isinstance(value, (np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    else:
        return value


def insert_match_result(match_id, model_prediction, **kwargs):
    """Inserts a new match result into the history table."""
    session = get_database_session()
    try:
        # Ensure match_id and model_prediction are integers
        match_id = int(match_id)
        model_prediction = int(model_prediction)

        # Check for an existing record with the same match_id
        existing_record = (
            session.query(History).filter(History.match_id == match_id).first()
        )
        if existing_record:
            logger.info(
                f"Record with match_id {match_id} already exists. Skipping insertion."
            )
            return  # Skip insertion if match_id already exists

        # Convert all kwargs to native Python types to avoid numpy types
        sanitized_data = {k: convert_to_native_type(v) for k, v in kwargs.items()}

        # Create a new History record with validated fields
        new_record = History(
            match_id=match_id,
            model_prediction=model_prediction,
            actual_result=None,  # Set default for actual_result if needed
            timestamp=datetime.utcnow(),
            **sanitized_data,  # Use sanitized kwargs
        )

        session.add(new_record)
        session.commit()
        logger.info(f"Data inserted successfully for match_id={match_id}")

    except ValueError as ve:
        logger.error(f"Value error: {ve}. Check types for match_id or prediction.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error inserting data for match_id={match_id}: {e}")
    finally:
        session.close()
        logger.info(
            f"Database session closed after inserting data for match_id={match_id}"
        )


def update_actual_result(match_id, actual_result):
    """Updates only the actual result for a given match_id in the history table."""
    session = get_database_session()
    try:
        # Query the History table for the specific match_id
        record = session.query(History).filter(History.match_id == match_id).first()
        if record:
            # Update the actual_result
            record.actual_result = actual_result
            session.commit()  # Commit the transaction
            logger.info(
                "Actual result updated successfully: match_id=%s, actual_result=%s",
                match_id,
                actual_result,
            )
        else:
            logger.warning("No record found for match_id: %s", match_id)
    except Exception as e:
        session.rollback()  # Rollback in case of error
        logger.error("Error updating actual result: %s", e)
    finally:
        session.close()  # Close the session
        logger.info("Database session closed after updating actual result.")


def get_history_data_as_dataframe():
    """Retrieves all data from the history table as a DataFrame."""
    session = get_database_session()
    try:
        # Query all data from the History table
        results = session.query(History).all()

        # Convert query results to a list of dictionaries
        data = [record.__dict__ for record in results]

        # Remove SQLAlchemy metadata (e.g., _sa_instance_state)
        for record in data:
            record.pop("_sa_instance_state", None)

        # Convert the data to a DataFrame and return it
        df = pd.DataFrame(data)
        logger.info("History data retrieved successfully as DataFrame")
        return df
    except Exception as e:
        logger.error(f"Error retrieving data as DataFrame: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    finally:
        session.close()
        logger.info("Database session closed after retrieving data.")


def fetch_and_update_actual_results():
    """Fetches actual results from OpenDota for matches with None actual_result and updates the database."""
    session = get_database_session()
    try:
        # Query to find all entries with actual_result as None
        matches_to_update = (
            session.query(History).filter(History.actual_result.is_(None)).all()
        )

        logger.info(f"Found {len(matches_to_update)} matches with None actual_result.")

        for match in matches_to_update:
            match_id = match.match_id
            logger.info(f"Fetching data for match_id={match_id} from OpenDota API...")

            # Fetch match data from OpenDota API
            response = requests.get(
                f"https://api.opendota.com/api/matches/{match_id}?api_key={opendota_key}"
            )

            if response.status_code == 200:
                match_data = response.json()
                actual_result = match_data.get(
                    "radiant_win"
                )  # Example field; adjust based on actual data structure

                if actual_result is not None:
                    logger.info(
                        f"Updating actual_result for match_id={match_id} to {actual_result}."
                    )
                    match.actual_result = int(
                        actual_result
                    )  # Assuming actual_result is binary (1 for win, 0 for loss)
                    session.commit()  # Commit the changes
                else:
                    logger.warning(f"No actual result found for match_id={match_id}.")
            else:
                logger.error(
                    f"Failed to fetch data for match_id={match_id}: HTTP {response.status_code}"
                )

    except SQLAlchemyError as e:
        logger.error(f"Database error occurred: {e}")
        session.rollback()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        session.close()
        logger.info("Database session closed after processing.")


def calculate_win_rate():
    """Calculates the win rate based on actual_result and model_prediction from the history table.

    Returns:
        tuple: A tuple containing the win rate (float) and the total number of predictions (int).
    """
    session = get_database_session()
    try:
        # Query to fetch all relevant data
        results = (
            session.query(History)
            .filter(
                History.actual_result.isnot(None), History.model_prediction.isnot(None)
            )
            .all()
        )

        total_predictions = len(results)
        correct_predictions = 0

        # Calculate the number of correct predictions
        for record in results:
            if record.actual_result == record.model_prediction:
                correct_predictions += 1

        # Calculate win rate
        if total_predictions > 0:
            win_rate = correct_predictions / total_predictions
        else:
            win_rate = 0  # Avoid division by zero

        logger.info(
            f"Total Predictions: {total_predictions}, Correct Predictions: {correct_predictions}, Win Rate: {win_rate:.2%}"
        )

        # Return both win rate and total predictions
        return win_rate, total_predictions

    except Exception as e:
        logger.error(f"Error calculating win rate: {e}")
        return None, 0  # Return None and 0 predictions on error

    finally:
        session.close()
        logger.info("Database session closed after calculating win rate.")
