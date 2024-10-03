# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.



import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

steam_api_key = os.getenv("STEAM_API_KEY")
opendota_key = os.getenv("OPENDOTA_KEY")
telegram_key = os.getenv("TELEGRAM_KEY")
