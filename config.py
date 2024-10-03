import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

steam_api_key = os.getenv("STEAM_API_KEY")
opendota_key = os.getenv("OPENDOTA_KEY")
telegram_key = os.getenv("TELEGRAM_KEY")
