# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

from sqlalchemy import create_engine, Column, Integer, DateTime, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(BigInteger, nullable=False)
    model_prediction = Column(Integer, nullable=False)
    actual_result = Column(Integer, nullable=True)
    timestamp = Column(DateTime, nullable=False)

    radiant_avg_hero_winrate = Column(Float, nullable=False)
    radiant_avg_roshans_killed = Column(Float, nullable=False)
    radiant_avg_last_hits = Column(Float, nullable=False)
    radiant_avg_denies = Column(Float, nullable=False)
    radiant_avg_hero_damage = Column(Float, nullable=False)
    radiant_avg_gpm = Column(Float, nullable=False)
    radiant_avg_xpm = Column(Float, nullable=False)
    radiant_avg_net_worth = Column(Float, nullable=False)
    radiant_avg_player_level = Column(Float, nullable=False)
    radiant_sum_obs = Column(Float, nullable=False)
    radiant_sum_sen = Column(Float, nullable=False)
    radiant_avg_teamfight_participation_cols = Column(Float, nullable=False)

    dire_avg_hero_winrate = Column(Float, nullable=False)
    dire_avg_roshans_killed = Column(Float, nullable=False)
    dire_avg_last_hits = Column(Float, nullable=False)
    dire_avg_denies = Column(Float, nullable=False)
    dire_avg_hero_damage = Column(Float, nullable=False)
    dire_avg_gpm = Column(Float, nullable=False)
    dire_avg_xpm = Column(Float, nullable=False)
    dire_avg_net_worth = Column(Float, nullable=False)
    dire_avg_player_level = Column(Float, nullable=False)
    dire_sum_obs = Column(Float, nullable=False)
    dire_sum_sen = Column(Float, nullable=False)
    dire_avg_teamfight_participation_cols = Column(Float, nullable=False)

    radiant_avg_kda = Column(Float, nullable=False)
    dire_avg_kda = Column(Float, nullable=False)


def create_database_and_tables(DATABASE_CONFIG):
    # Create the database URL
    db_url = f"postgresql://{DATABASE_CONFIG['USER']}:{DATABASE_CONFIG['PASSWORD']}@{DATABASE_CONFIG['HOST']}:{DATABASE_CONFIG['PORT']}/{DATABASE_CONFIG['DB_NAME']}"

    logger.info("Connecting to the database...")
    try:
        # Create the database engine
        engine = create_engine(db_url)

        # Create the tables in the database
        Base.metadata.create_all(engine)

        logger.info("Database and tables created successfully.")
    except Exception as e:
        logger.error("Error occurred while creating database and tables: %s", e)
