# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import os
import logging.config
from dotenv import load_dotenv

from db.setup import create_database_and_tables

logging.config.fileConfig("logging.conf")

# Load environment variables from .env file
load_dotenv()

steam_api_key = os.getenv("STEAM_API_KEY")
opendota_key = os.getenv("OPENDOTA_KEY")
telegram_key = os.getenv("TELEGRAM_KEY")

DATABASE_CONFIG = {
    "DB_NAME": os.getenv("POSTGRES_DB", "mydatabase"),
    "USER": os.getenv("POSTGRES_USER", "myuser"),
    "PASSWORD": os.getenv("POSTGRES_PASSWORD", "mypassword"),
    "HOST": os.getenv("DB_HOST", "localhost"),
    "PORT": os.getenv("DB_PORT", "5432"),
}

create_database_and_tables(DATABASE_CONFIG)
