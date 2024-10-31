# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_CONFIG
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
