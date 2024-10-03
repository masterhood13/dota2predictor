# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.



from time import sleep
import pandas as pd
import requests
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import opendota_key, steam_api_key
from ml.model import Dota2MatchPredictor, MainML
from structure.helpers import find_dict_in_list, prepare_data


class Dota2API:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={self.api_key}"

    def get_live_tournaments(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            live_matches = response.json().get("result", {}).get("games", [])
            print(len(live_matches))
            return self.build_tournaments(live_matches)
        else:
            print(f"Error: {response.status_code}")
            return []

    def build_tournaments(self, matches_data):
        tournaments = {}

        for match_data in matches_data:
            league_id = match_data.get("league_id")
            league_name = match_data.get("league_name", "Unknown Tournament")

            if league_id not in tournaments:
                tournaments[league_id] = Tournament(league_id, league_name)

            # Check for the presence of radiant_team and dire_team
            radiant_team_data = match_data.get("radiant_team")
            dire_team_data = match_data.get("dire_team")
            players_data = match_data.get("players", [])
            # Check if any player has hero_id == 0
            if any(player["hero_id"] == 0 for player in players_data):
                print(
                    f"Skipping match {match_data.get('match_id')} due to a player having hero_id = 0"
                )
                continue

            if radiant_team_data and dire_team_data:  # Ensure both teams are present
                radiant_team = self.build_team(radiant_team_data, 0, players_data)
                dire_team = self.build_team(dire_team_data, 1, players_data)

                match = Match(
                    match_id=match_data.get("match_id"),
                    radiant_team_id=radiant_team.team_id,
                    dire_team_id=dire_team.team_id,
                    league_id=league_id,
                )
                match.dire_team = dire_team
                match.radiant_team = radiant_team

                tournaments[league_id].add_match(match)

        return list(tournaments.values())

    def build_team(self, team_data, team_side, players_data):
        if team_data is None:
            return None  # Return None if team_data is not provided

        team = Team(
            team_name=team_data.get("team_name", "Unknown"),
            team_id=team_data.get("team_id", 0),
        )
        for player_data in players_data:
            if player_data.get("team") == team_side:
                player = Player(
                    account_id=player_data.get("account_id"),
                    name=player_data.get("name", "Unknown"),
                    hero_id=player_data.get("hero_id", 0),
                    team=team_side,
                )
                team.add_player(player)
        return team


class CallbackTriggers:
    dota2_get_current_matches_trigger = "cb_dota2"
    player_trigger = "cb_player"
    hero_trigger = "cb_hero"
    side_trigger = "cb_side"


class Icons:
    playerIcon = "\U0001F468"
    direIcon = "\U0001F47F"
    radiantIcon = "\U0001F607"


class Buttons:
    dota2_get_current_matches_button = InlineKeyboardButton(
        "Get current Dota2 matches",
        callback_data=CallbackTriggers.dota2_get_current_matches_trigger,
    )

    dota2_restart_button = InlineKeyboardButton(
        "No matches found try again later",
        callback_data=CallbackTriggers.dota2_get_current_matches_trigger,
    )

    dire_button = InlineKeyboardButton(
        f"Dire {Icons.direIcon}", callback_data=CallbackTriggers.side_trigger
    )
    radiant_button = InlineKeyboardButton(
        f"Radiant {Icons.radiantIcon}", callback_data=CallbackTriggers.side_trigger
    )

    @staticmethod
    def team_button(team_name):
        return InlineKeyboardButton(
            text=str(team_name),
            callback_data=f"{CallbackTriggers.player_trigger},'{team_name}']",
        )

    @staticmethod
    def hero_button(hero_name):
        return InlineKeyboardButton(
            text=str(hero_name),
            callback_data=f'{CallbackTriggers.hero_trigger},"{hero_name}"]',
        )

    @staticmethod
    def hero_teamplay_button(hero_name, player):
        return InlineKeyboardButton(
            text=str(hero_name),
            callback_data=f"{CallbackTriggers.hero_trigger},\"{hero_name}\",'{player}']",
        )

    @staticmethod
    def predict_button(predict):
        return InlineKeyboardButton(
            text=f"Radiant win prob: {predict}",
            callback_data=CallbackTriggers.side_trigger,
        )

    @staticmethod
    def player_button(player):
        return InlineKeyboardButton(
            text=f"{str(player)} {Icons.playerIcon}",
            callback_data=f"{CallbackTriggers.player_trigger},'{player}']",
        )

    @staticmethod
    def tournament_button(tournament_name):
        return InlineKeyboardButton(
            text=f"Tournament: {tournament_name}",
            callback_data=CallbackTriggers.side_trigger,
        )


class Hero:
    def __init__(self, hero_id):
        self.hero_id = hero_id
        self.features = self.get_hero_features()
        self.name = self.features["name"] if self.features else "Unknown Hero"

        if self.features and self.features["pro_pick"] > 0:
            self.winrate = self.features["pro_win"] / self.features["pro_pick"]
        else:
            self.winrate = 0

    def get_hero_features(self):
        url = f"https://api.opendota.com/api/heroStats?api_key={opendota_key}"
        response = requests.get(url)

        if response.status_code == 200:
            heroes = response.json()
            for hero in heroes:
                if hero["id"] == self.hero_id:
                    return {
                        "hero_id": hero["id"],
                        "name": hero["localized_name"],
                        "pro_win": hero.get("pro_win", 0),
                        "pro_pick": hero.get("pro_pick", 0),
                    }
        else:
            print(f"Error fetching data: {response.status_code}")
            return None

    def __repr__(self):
        return f"Hero(ID: {self.hero_id}, Name: {self.name}, Features: {self.features})"


class Player:
    def __init__(self, account_id, name, hero_id, team):
        self.account_id = account_id
        self.name = name
        self.team = team
        self.hero = Hero(hero_id)
        self.player_data = self.get_player_data()

        player_data = self.get_player_total_data()

        kills = find_dict_in_list(player_data, "field", "kills")
        self.kills = kills["sum"] / kills["n"] if kills["n"] > 0 else 0
        deaths = find_dict_in_list(player_data, "field", "deaths")
        self.deaths = deaths["sum"] / deaths["n"] if deaths["n"] > 0 else 0
        assists = find_dict_in_list(player_data, "field", "assists")
        self.assists = assists["sum"] / assists["n"] if assists["n"] > 0 else 0
        gold_per_min = find_dict_in_list(player_data, "field", "gold_per_min")
        self.gold_per_min = (
            gold_per_min["sum"] / gold_per_min["n"] if gold_per_min["n"] > 0 else 0
        )
        xp_per_min = find_dict_in_list(player_data, "field", "xp_per_min")
        self.xp_per_min = (
            xp_per_min["sum"] / xp_per_min["n"] if xp_per_min["n"] > 0 else 0
        )
        last_hits = find_dict_in_list(player_data, "field", "last_hits")
        self.last_hits = last_hits["sum"] / last_hits["n"] if last_hits["n"] > 0 else 0
        denies = find_dict_in_list(player_data, "field", "denies")
        self.denies = denies["sum"] / denies["n"] if denies["n"] > 0 else 0

    def get_player_total_data(self):
        """Fetch player total data with indefinite retries until success."""
        url = f"https://api.opendota.com/api/players/{self.account_id}/totals?api_key={opendota_key}&hero_id={self.hero.hero_id}&limit=30"

        while True:  # Retry loop
            try:
                response = requests.get(url)

                if response.status_code == 200:
                    return response.json()  # Successful response, exit loop
                else:
                    print(
                        f"Error fetching player data: {response.status_code}. Retrying..."
                    )

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying...")

            sleep(2)

    def get_player_data(self):
        # Fetch general win/loss data
        url = f"https://api.opendota.com/api/players/{self.account_id}/wl?api_key={opendota_key}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            player_stats = {
                "win_rate": (
                    data.get("win") / (data.get("win") + data.get("lose"))
                    if (data.get("win") + data.get("lose")) > 0
                    else 0
                ),
            }

            hero_url = f"https://api.opendota.com/api/players/{self.account_id}/heroes?api_key={opendota_key}&limit=30"
            hero_response = requests.get(hero_url)

            if hero_response.status_code == 200:
                hero_data = hero_response.json()
                for hero in hero_data:
                    if hero["hero_id"] == self.hero.hero_id:
                        # Calculate the hero's win rate
                        if hero["games"] > 0:
                            self.hero_win_rate = hero["win"] / hero["games"]
                        else:
                            self.hero_win_rate = 0

                        break
            else:
                print(f"Error fetching hero data: {hero_response.status_code}")

            return player_stats
        else:
            print(f"Error fetching player data: {response.status_code}")
            return None

    def __repr__(self):
        return f"Player({self.name}, Hero : {self.hero.name}, Team: {self.team}, Data: {self.player_data})"


class Team:
    def __init__(self, team_name: str, team_id: int):
        self.team_name = team_name
        self.team_id = team_id
        self.players = []

    def add_player(self, player):
        self.players.append(player)

    def __repr__(self):
        return f"Team({self.team_name}, ID: {self.team_id}, Players: {self.players})"


class Match:
    def __init__(
        self,
        match_id: int,
        radiant_team_id: int,
        dire_team_id: int,
        league_id: int,
        radiant_win=None,
    ):
        self.match_id = match_id
        self.radiant_team_id = radiant_team_id
        self.dire_team_id = dire_team_id
        self.radiant_team = None
        self.dire_team = None
        self.league_id = league_id
        self.radiant_win = radiant_win

    def get_match_data(self):
        url = f"https://api.opendota.com/api/matches/{self.match_id}?api_key={opendota_key}"
        response = requests.get(url)

        if response.status_code == 200:
            match_info = response.json()
            radiant_team = Team(
                match_info["radiant_name"], match_info["radiant_team_id"]
            )
            dire_team = Team(match_info["dire_name"], match_info["dire_team_id"])
            self.radiant_win = match_info["radiant_win"]
            for player in match_info["players"]:
                if player["isRadiant"]:
                    player = Player(
                        player["account_id"],
                        player["name"],
                        player["hero_id"],
                        radiant_team.team_name,
                    )
                    radiant_team.add_player(player)
                else:
                    player = Player(
                        player["account_id"],
                        player["name"],
                        player["hero_id"],
                        dire_team.team_name,
                    )
                    dire_team.add_player(player)
            self.radiant_team = radiant_team
            self.dire_team = dire_team

    def get_match_data_for_prediction(self):
        if len(self.radiant_team.players) == 5 and len(self.dire_team.players) == 5:
            # Create a single row with match and player data
            match_data = {
                "match_id": self.match_id,
                "radiant_team_id": self.radiant_team.team_id,
                "radiant_team_name": self.radiant_team.team_name,
                "dire_team_id": self.dire_team.team_id,
                "dire_team_name": self.dire_team.team_name,
            }

            # Add radiant team player data (5 players)
            for i, player in enumerate(self.radiant_team.players):
                match_data[f"radiant_player_{i + 1}_id"] = player.account_id
                match_data[f"radiant_player_{i + 1}_name"] = player.name
                match_data[f"radiant_player_{i + 1}_hero_id"] = player.hero.hero_id
                match_data[f"radiant_player_{i + 1}_hero_name"] = player.hero.name
                match_data[f"radiant_player_{i + 1}_hero_winrate"] = player.hero.winrate
                match_data[f"radiant_player_{i + 1}_winrate"] = player.player_data[
                    "win_rate"
                ]
                match_data[f"radiant_player_{i + 1}_kills"] = player.kills
                match_data[f"radiant_player_{i + 1}_deaths"] = player.deaths
                match_data[f"radiant_player_{i + 1}_assists"] = player.assists
                match_data[f"radiant_player_{i + 1}_gold_per_min"] = player.gold_per_min
                match_data[f"radiant_player_{i + 1}_xp_per_min"] = player.xp_per_min

            # Add dire team player data (5 players)
            for i, player in enumerate(self.dire_team.players):
                match_data[f"dire_player_{i + 1}_id"] = player.account_id
                match_data[f"dire_player_{i + 1}_name"] = player.name
                match_data[f"dire_player_{i + 1}_hero_id"] = player.hero.hero_id
                match_data[f"dire_player_{i + 1}_hero_name"] = player.hero.name
                match_data[f"dire_player_{i + 1}_hero_winrate"] = player.hero.winrate
                match_data[f"dire_player_{i + 1}_winrate"] = player.player_data[
                    "win_rate"
                ]
                match_data[f"dire_player_{i + 1}_kills"] = player.kills
                match_data[f"dire_player_{i + 1}_deaths"] = player.deaths
                match_data[f"dire_player_{i + 1}_assists"] = player.assists
                match_data[f"dire_player_{i + 1}_gold_per_min"] = player.gold_per_min
                match_data[f"dire_player_{i + 1}_xp_per_min"] = player.xp_per_min
        df = pd.DataFrame([match_data])
        df = prepare_data(df)
        top_features = df.columns.tolist()
        return df, top_features

    def __repr__(self):
        # Prepare the Radiant team players
        radiant_players = "\n".join(
            [
                f"    Player: {player.name} (Hero : {player.hero.name})"
                for player in self.radiant_team.players
            ]
        )

        # Prepare the Dire team players
        dire_players = "\n".join(
            [
                f"    Player: {player.name} (Hero : {player.hero.name})"
                for player in self.dire_team.players
            ]
        )

        # Format the result
        return (
            f"Match ID: {self.match_id}\n"
            f"League ID: {self.league_id}\n"
            f"Radiant Team: {self.radiant_team.team_name}\n"
            f"Radiant Players:\n{radiant_players}\n"
            f"Dire Team: {self.dire_team.team_name}\n"
            f"Dire Players:\n{dire_players}\n"
            f"Radiant Win: {'Yes' if self.radiant_win else 'No'}"
        )


class Tournament:
    def __init__(self, league_id: int, name: str):
        self.league_id = league_id
        self.name = name
        self.matches = []

    def add_match(self, match):
        self.matches.append(match)

    def get_league_matches(self):
        url = f"https://api.opendota.com/api/leagues/{self.league_id}/matches?api_key={opendota_key}"
        response = requests.get(url)
        if response.status_code == 200:
            for match_info in response.json():
                match_id = match_info["match_id"]
                radiant_team_id = match_info["radiant_team_id"]
                dire_team_id = match_info["dire_team_id"]
                radiant_win = match_info["radiant_win"]
                match = Match(
                    match_id, radiant_team_id, dire_team_id, self.league_id, radiant_win
                )
                match.get_match_data()
                self.add_match(match)
        else:
            print(
                f"Error fetching matches for league {self.league_id}: {response.status_code}"
            )

    def __repr__(self):
        return f"Tournament({self.name}, ID: {self.league_id})"


class Markups:
    def __init__(self):
        self.markup = InlineKeyboardMarkup()
        self.markup.row_width = 8

    def gen_main_markup(self, current_user_id, current_channel_id):
        self.markup.add(Buttons.dota2_get_current_matches_button)
        return self.markup

    def gen_dota2_matches_markup(self):
        dota_api = Dota2API(steam_api_key)
        tournaments = dota_api.get_live_tournaments()
        self.markup = InlineKeyboardMarkup()

        for tournament in tournaments:
            for match in tournament.matches:
                self.markup.add(
                    Buttons.tournament_button(
                        f"{tournament.name}_{tournament.league_id}"
                    )
                )
                self.markup.add(Buttons.dire_button)
                self.markup.add(
                    Buttons.team_button(
                        f"{match.dire_team.team_name}_{match.dire_team.team_id}"
                    )
                )
                for player in match.dire_team.players:
                    self.markup.add(
                        Buttons.player_button(player=player.name),
                        Buttons.hero_teamplay_button(
                            player=player.account_id, hero_name=player.hero.name
                        ),
                    )

                self.markup.add(Buttons.radiant_button)
                self.markup.add(
                    Buttons.team_button(
                        f"{match.radiant_team.team_name}_{match.radiant_team.team_id}"
                    )
                )
                for player in match.radiant_team.players:
                    self.markup.add(
                        Buttons.player_button(player=player.name),
                        Buttons.hero_teamplay_button(
                            player=player.account_id, hero_name=player.hero.name
                        ),
                    )

                df, top_features = match.get_match_data_for_prediction()

                predictor = MainML(
                    df=df,  # The DataFrame after preparation
                    model_path="./my_pytorch_model.pth",  # Path where the model will be saved
                    top_features=None,  # Use all features if None is passed
                    n_hidden=[100, 50],  # Hidden layers configuration
                    drop_p=0.2,  # Dropout rate
                    random_state=17,  # Seed for reproducibility
                )

                prediction = predictor.predict_new_data(df)
                # Add the prediction button to the markup
                self.markup.add(Buttons.predict_button(prediction[0]))
        if not (self.markup.keyboard):
            self.markup.add(Buttons.dota2_restart_button)


        return self.markup
