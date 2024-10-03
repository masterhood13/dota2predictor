# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.



import requests


class OpenDotaApi:
    def __init__(self):
        self.MAIN_URL = "https://api.opendota.com/api/"
        # self.opendota_key = OPENDOTA_KEY
        self.limit = 30
        self.lobby_type = 2

    def set_premium_leagues(self):
        leagues = requests.get(url=f"{self.MAIN_URL}/leagues").json()
        return leagues
