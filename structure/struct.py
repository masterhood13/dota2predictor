# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

from time import sleep
import pandas as pd
import requests
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import opendota_key, steam_api_key
from ml.model import MainML
from structure.helpers import prepare_data

main_ml = MainML(None, "xgb_model.pkl")
main_ml.load_model()


class Dota2API:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={self.api_key}&dpc=true"

    def fetch_live_matches(self):
        """Fetch live matches data from the Dota 2 API."""
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            return response.json().get("result", {}).get("games", [])
        except requests.RequestException as e:
            print(f"Error fetching live matches: {e}")
            return []

    def get_live_tournaments(self):
        """Fetch and build a list of live tournaments."""
        live_matches = self.fetch_live_matches()
        return self.build_tournaments(live_matches)

    def get_match_as_buttons(self, markup):
        """Build and add buttons for each live match."""
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if self.is_valid_match(match_data):
                radiant_team, dire_team = match_data.get(
                    "radiant_team"
                ), match_data.get("dire_team")
                if radiant_team and dire_team:
                    markup.add(
                        Buttons.match_button(
                            dire_team_name=dire_team.get("team_name", "Unknown"),
                            radiant_team_name=radiant_team.get("team_name", "Unknown"),
                            match_id=match_data.get("match_id"),
                        )
                    )
        markup.add(Buttons.dota2_restart_button)
        return markup

    def build_single_match(self, match_id):
        """Build a single match object given the match_id."""
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if str(match_data.get("match_id")) == str(match_id) and self.is_valid_match(
                match_data
            ):
                return self.create_match_object(match_data)
        return None

    def build_tournaments(self, matches_data):
        """Organize matches into tournaments."""
        tournaments = {}
        for match_data in matches_data:
            league_id, league_name = match_data.get("league_id"), match_data.get(
                "league_name", "Unknown Tournament"
            )

            if league_id not in tournaments:
                tournaments[league_id] = Tournament(league_id, league_name)

            if self.is_valid_match(match_data):
                match = self.create_match_object(match_data)
                if match:
                    tournaments[league_id].add_match(match)

        return list(tournaments.values())

    def build_team(self, team_data, team_side, players_data):
        """Build a team object with players."""
        if not team_data:
            return None

        team = Team(
            team_name=team_data.get("team_name", "Unknown"),
            team_id=team_data.get("team_id", 0),
        )

        team_players = [
            self.build_player(player_data)
            for player_data in players_data
            if player_data.get("team") == team_side
        ]
        for player in team_players:
            if player:
                team.add_player(player)

        return team

    def build_player(self, player_data):
        """Build a player object."""
        return Player(
            account_id=player_data.get("account_id"),
            name=player_data.get("name", "Unknown"),
            hero_id=player_data.get("hero_id", 0),
            team=player_data.get("team"),
        )

    def create_match_object(self, match_data):
        """Create a Match object from match data."""
        radiant_team_data, dire_team_data = match_data.get(
            "radiant_team"
        ), match_data.get("dire_team")
        players_data = match_data.get("players", [])

        radiant_team = self.build_team(radiant_team_data, 0, players_data)
        dire_team = self.build_team(dire_team_data, 1, players_data)

        if radiant_team and dire_team:
            match = Match(
                match_id=match_data.get("match_id"),
                radiant_team_id=radiant_team.team_id,
                dire_team_id=dire_team.team_id,
                league_id=match_data.get("league_id"),
            )
            match.dire_team = dire_team
            match.radiant_team = radiant_team
            return match
        return None

    def is_valid_match(self, match_data):
        """Check if the match is valid (no player with hero_id = 0 on teams)."""
        players_data = match_data.get("players", [])
        invalid_players = [
            player
            for player in players_data
            if player["hero_id"] == 0 and player["team"] in (0, 1)
        ]

        if invalid_players:
            print(
                f"Skipping match {match_data.get('match_id')} due to invalid players."
            )
            return False
        return True


class CallbackTriggers:
    dota2_get_current_matches_trigger = "cb_dota2"
    predict_by_id_trigger = "cb_match_by_id"
    match_trigger = "['cb_match_t'"


class Icons:
    playerIcon = "\U0001F468"
    direIcon = "\U0001F47F"
    radiantIcon = "\U0001F607"
    reload = "\U0001F503"


class Buttons:
    dota2_get_current_matches_button = InlineKeyboardButton(
        "Predict all matches at once",
        callback_data=CallbackTriggers.dota2_get_current_matches_trigger,
    )

    predict_by_id_button = InlineKeyboardButton(
        "Select available match to predict",
        callback_data=CallbackTriggers.predict_by_id_trigger,
    )

    dota2_restart_button = InlineKeyboardButton(
        f"Reload results {Icons.reload}",
        callback_data=CallbackTriggers.predict_by_id_trigger,
    )

    @staticmethod
    def match_button(dire_team_name, radiant_team_name, match_id):
        return InlineKeyboardButton(
            text=f"{match_id} | {Icons.direIcon}{dire_team_name} VS {Icons.radiantIcon}{radiant_team_name}",
            callback_data=f'{CallbackTriggers.match_trigger},"{match_id}"]',
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
    def __init__(self, account_id, name, hero_id, team, player_data=None):
        self.account_id = account_id
        self.team = team
        self.hero = Hero(hero_id)
        self.name = name
        if player_data:
            self.teamfight_participation = player_data.get("teamfight_participation", 0)
            self.obs_placed = player_data.get("obs_placed", 0)
            self.sen_placed = player_data.get("sen_placed", 0)
            self.net_worth = player_data.get("net_worth", 0)
            self.kills = player_data.get("kills", 0)
            self.deaths = player_data.get("deaths", 0)
            self.assists = player_data.get("assists", 0)
            self.roshans_killed = player_data.get("roshans_killed", 0)
            self.last_hits = player_data.get("last_hits", 0)
            self.denies = player_data.get("denies", 0)
            self.gold_per_min = player_data.get("gold_per_min", 0)
            self.xp_per_min = player_data.get("xp_per_min", 0)
            self.level = player_data.get("level", 0)
            self.hero_damage = player_data.get("hero_damage", 0)
            self.tower_damage = player_data.get("tower_damage", 0)
            self.hero_healing = player_data.get("hero_healing", 0)
        else:
            self.teamfight_participation = 0
            self.obs_placed = 0
            self.sen_placed = 0
            self.net_worth = 0
            self.kills = 0
            self.deaths = 0
            self.assists = 0
            self.roshans_killed = 0
            self.last_hits = 0
            self.denies = 0
            self.gold_per_min = 0
            self.xp_per_min = 0
            self.level = 0
            self.hero_damage = 0
            self.tower_damage = 0
            self.hero_healing = 0
            self.get_player_total_data()

    def get_player_total_data(self):
        """Fetch player total data with retries on match data retrieval."""
        recent_matches = self.fetch_recent_matches()

        # Initialize counters for averages
        participation_count = obs_count = sen_count = net_worth_count = 0
        kills_count = deaths_count = assists_count = roshan_count = 0
        last_hits_count = denies_count = gpm_count = xpm_count = level_count = 0
        hero_damage_count = tower_damage_count = healing_count = 0

        # Iterate through recent matches
        for match in recent_matches:
            match_id = match["match_id"]
            match_data = self.fetch_match_data_with_retries(match_id)

            if match_data is None:
                print(f"Skipping match {match_id} after 5 attempts")
                continue  # Skip the match if it couldn't be retrieved

            # Get player data
            player_data = self.get_player_data(match_data)

            if player_data:
                # Safely accumulate values if the keys exist and track counts
                participation_count = self.accumulate_value(
                    player_data, "teamfight_participation", participation_count
                )
                obs_count = self.accumulate_value(player_data, "obs_placed", obs_count)
                sen_count = self.accumulate_value(player_data, "sen_placed", sen_count)
                net_worth_count = self.accumulate_value(
                    player_data, "net_worth", net_worth_count
                )
                kills_count = self.accumulate_value(player_data, "kills", kills_count)
                deaths_count = self.accumulate_value(
                    player_data, "deaths", deaths_count
                )
                assists_count = self.accumulate_value(
                    player_data, "assists", assists_count
                )
                roshan_count = self.accumulate_value(
                    player_data, "roshans_killed", roshan_count
                )
                last_hits_count = self.accumulate_value(
                    player_data, "last_hits", last_hits_count
                )
                denies_count = self.accumulate_value(
                    player_data, "denies", denies_count
                )
                gpm_count = self.accumulate_value(
                    player_data, "gold_per_min", gpm_count
                )
                xpm_count = self.accumulate_value(player_data, "xp_per_min", xpm_count)
                level_count = self.accumulate_value(player_data, "level", level_count)
                hero_damage_count = self.accumulate_value(
                    player_data, "hero_damage", hero_damage_count
                )
                tower_damage_count = self.accumulate_value(
                    player_data, "tower_damage", tower_damage_count
                )
                healing_count = self.accumulate_value(
                    player_data, "hero_healing", healing_count
                )

        # Safely divide by the number of successful additions for each field
        self.teamfight_participation = self.calculate_average(
            self.teamfight_participation, participation_count
        )
        self.obs_placed = self.calculate_average(self.obs_placed, obs_count)
        self.sen_placed = self.calculate_average(self.sen_placed, sen_count)
        self.net_worth = self.calculate_average(self.net_worth, net_worth_count)
        self.kills = self.calculate_average(self.kills, kills_count)
        self.deaths = self.calculate_average(self.deaths, deaths_count)
        self.assists = self.calculate_average(self.assists, assists_count)
        self.roshans_killed = self.calculate_average(self.roshans_killed, roshan_count)
        self.last_hits = self.calculate_average(self.last_hits, last_hits_count)
        self.denies = self.calculate_average(self.denies, denies_count)
        self.gold_per_min = self.calculate_average(self.gold_per_min, gpm_count)
        self.xp_per_min = self.calculate_average(self.xp_per_min, xpm_count)
        self.level = self.calculate_average(self.level, level_count)
        self.hero_damage = self.calculate_average(self.hero_damage, hero_damage_count)
        self.tower_damage = self.calculate_average(
            self.tower_damage, tower_damage_count
        )
        self.hero_healing = self.calculate_average(self.hero_healing, healing_count)

    def fetch_recent_matches(self):
        """Fetch recent matches for the player."""
        response = requests.get(
            f"https://api.opendota.com/api/players/{self.account_id}/matches?api_key={opendota_key}&limit=10&hero_id={self.hero.hero_id}&lobby_type=1"
        )
        return response.json() if response.status_code == 200 else []

    def fetch_match_data_with_retries(self, match_id):
        """Fetch match data with retries."""
        retries = 0
        max_retries = 5

        while retries < max_retries:
            response = requests.get(
                f"https://api.opendota.com/api/matches/{match_id}?api_key={opendota_key}"
            )
            if response.status_code == 200:
                return response.json()  # Successful response
            else:
                retries += 1
                print(
                    f"Retrying... attempt {retries} (Status code: {response.status_code})"
                )
                sleep(2)  # Sleep for 2 seconds before retrying

        return None  # Return None if all retries fail

    def get_player_data(self, match_data):
        """Extract player data from match data."""
        players = match_data.get("players", [])
        return next(
            (
                player
                for player in players
                if player.get("account_id") == self.account_id
            ),
            None,
        )

    def accumulate_value(self, player_data, key, count):
        """Accumulate value for a given key and return updated count."""
        if key in player_data:
            try:
                setattr(
                    self, key, getattr(self, key) + player_data[key]
                )  # Update the stat
                count += 1  # Increment count
            except TypeError:
                count = 1
        return count

    def calculate_average(self, total, count):
        """Calculate average, returning 0 if count is 0."""
        return total / count if count > 0 else 0

    def __repr__(self):
        return (
            f"Player(Name: {self.name}, Hero: {self.hero.name}, Team: {self.team}, "
            f"Teamfight Participation: {self.teamfight_participation * 100:.1f}%, "
            f"Observers Placed: {self.obs_placed}, Sentries Placed: {self.sen_placed}, "
            f"K/D/A: {self.kills}/{self.deaths}/{self.assists}, "
            f"Net Worth: {self.net_worth:,}, Roshan Killed: {self.roshans_killed}, Last Hits: {self.last_hits}, "
            f"Denies: {self.denies}, GPM: {self.gold_per_min}, XPM: {self.xp_per_min}, "
            f"Level: {self.level}, Hero Damage: {self.hero_damage:,}, Tower Damage: {self.tower_damage:,}, "
            f"Hero Healing: {self.hero_healing:,})"
        )


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
                        account_id=player["account_id"],
                        hero_id=player["hero_id"],
                        name=player["name"],
                        team=radiant_team.team_name,
                        player_data=player,
                    )
                    radiant_team.add_player(player)
                else:
                    player = Player(
                        account_id=player["account_id"],
                        hero_id=player["hero_id"],
                        name=player["name"],
                        team=dire_team.team_name,
                        player_data=player,
                    )
                    dire_team.add_player(player)
                print(player)
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
                # match_data[f"radiant_player_{i + 1}_winrate"] = player.player_data["win_rate"]
                match_data[f"radiant_player_{i + 1}_kills"] = player.kills
                match_data[f"radiant_player_{i + 1}_deaths"] = player.deaths
                match_data[f"radiant_player_{i + 1}_assists"] = player.assists
                match_data[f"radiant_player_{i + 1}_gold_per_min"] = player.gold_per_min
                match_data[f"radiant_player_{i + 1}_xp_per_min"] = player.xp_per_min

                match_data[f"radiant_player_{i + 1}_teamfight_participation"] = (
                    player.teamfight_participation
                )
                match_data[f"radiant_player_{i + 1}_obs_placed"] = player.obs_placed
                match_data[f"radiant_player_{i + 1}_sen_placed"] = player.sen_placed
                match_data[f"radiant_player_{i + 1}_net_worth"] = player.net_worth
                match_data[f"radiant_player_{i + 1}_roshans_killed"] = (
                    player.roshans_killed
                )
                match_data[f"radiant_player_{i + 1}_last_hits"] = player.last_hits
                match_data[f"radiant_player_{i + 1}_denies"] = player.denies
                match_data[f"radiant_player_{i + 1}_level"] = player.level
                match_data[f"radiant_player_{i + 1}_hero_damage"] = player.hero_damage
                match_data[f"radiant_player_{i + 1}_tower_damage"] = player.tower_damage

            # Add dire team player data (5 players)
            for i, player in enumerate(self.dire_team.players):
                match_data[f"dire_player_{i + 1}_id"] = player.account_id
                match_data[f"dire_player_{i + 1}_name"] = player.name
                match_data[f"dire_player_{i + 1}_hero_id"] = player.hero.hero_id
                match_data[f"dire_player_{i + 1}_hero_name"] = player.hero.name
                match_data[f"dire_player_{i + 1}_hero_winrate"] = player.hero.winrate
                # match_data[f"dire_player_{i + 1}_winrate"] = player.player_data["win_rate"]
                match_data[f"dire_player_{i + 1}_kills"] = player.kills
                match_data[f"dire_player_{i + 1}_deaths"] = player.deaths
                match_data[f"dire_player_{i + 1}_assists"] = player.assists
                match_data[f"dire_player_{i + 1}_gold_per_min"] = player.gold_per_min
                match_data[f"dire_player_{i + 1}_xp_per_min"] = player.xp_per_min

                match_data[f"dire_player_{i + 1}_teamfight_participation"] = (
                    player.teamfight_participation
                )
                match_data[f"dire_player_{i + 1}_obs_placed"] = player.obs_placed
                match_data[f"dire_player_{i + 1}_sen_placed"] = player.sen_placed
                match_data[f"dire_player_{i + 1}_net_worth"] = player.net_worth
                match_data[f"dire_player_{i + 1}_roshans_killed"] = (
                    player.roshans_killed
                )
                match_data[f"dire_player_{i + 1}_last_hits"] = player.last_hits
                match_data[f"dire_player_{i + 1}_denies"] = player.denies
                match_data[f"dire_player_{i + 1}_level"] = player.level
                match_data[f"dire_player_{i + 1}_hero_damage"] = player.hero_damage
                match_data[f"dire_player_{i + 1}_tower_damage"] = player.tower_damage

        df = pd.DataFrame([match_data])
        df = prepare_data(df, "scaler.pkl")
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
                print(match_info)
                match_id = match_info["match_id"]
                radiant_team_id = match_info["radiant_team_id"]
                dire_team_id = match_info["dire_team_id"]
                radiant_win = match_info["radiant_win"]
                match = Match(
                    match_id, radiant_team_id, dire_team_id, self.league_id, radiant_win
                )
                try:
                    match.get_match_data()
                    self.add_match(match)
                except (TypeError, KeyError):
                    # done ofr skip broken data
                    pass
        else:
            print(
                f"Error fetching matches for league {self.league_id}: {response.status_code}"
            )

    def __repr__(self):
        return f"Tournament({self.name}, ID: {self.league_id})"


class Markups:
    def __init__(self, bot):
        self.markup = InlineKeyboardMarkup()
        self.markup.row_width = 8
        self.bot = bot

    def gen_main_markup(self, current_user_id, current_channel_id):
        self.markup.add(Buttons.dota2_get_current_matches_button)
        self.markup.add(Buttons.predict_by_id_button)
        return self.markup

    def gen_dota2_matches_markup(self, call):
        dota_api = Dota2API(steam_api_key)
        self.bot.send_message(
            chat_id=call.message.chat.id,
            text="<b>Task started. This may take around 5 minutes. Please wait...</b>",
            parse_mode="HTML",
        )
        tournaments = dota_api.get_live_tournaments()
        for tournament in tournaments:
            for match in tournament.matches:
                message = ""  # Start with a header
                message += f"<b>Tournament:</b> {tournament.name}\n"
                message += f"<b>League ID:</b> {tournament.league_id}\n\n"  # Include league ID for reference
                message += f"<b>Match ID:</b> {match.match_id}\n"
                message += f"<b>Dire Team {Icons.direIcon} :</b> {match.dire_team.team_name} (ID: {match.dire_team.team_id})\n"
                message += "<b>Players:</b>\n"

                # List Dire team players
                for player in match.dire_team.players:
                    message += f"   - {player.name} {Icons.playerIcon}(Hero: {player.hero.name})\n"

                message += f"\n<b>Radiant Team {Icons.radiantIcon}:</b> {match.radiant_team.team_name} (ID: {match.radiant_team.team_id})\n"
                message += "<b>Players:</b>\n"

                # List Radiant team players
                for player in match.radiant_team.players:
                    message += f"   - {player.name} {Icons.playerIcon}(Hero: {player.hero.name})\n"

                # Prepare match data for prediction
                df, top_features = match.get_match_data_for_prediction()
                prediction = main_ml.predict(df)
                print(prediction)

                # Add the prediction to the message
                message += f"\n<b>Prediction:</b> {'Radiant Wins' if prediction[0] == 1 else 'Dire Wins'}\n"
                message += "<b>----------------------------------------</b>\n"  # Separator line in bold
                self.bot.send_message(
                    chat_id=call.message.chat.id, text=message, parse_mode="HTML"
                )
        self.bot.send_message(
            chat_id=call.message.chat.id, text="<b>DONE</b>", parse_mode="HTML"
        )

    def gen_match_markup_by_id(self, call):
        dota_api = Dota2API(steam_api_key)
        self.markup = dota_api.get_match_as_buttons(self.markup)
        return self.markup

    def make_prediction_for_selected_match(self, call, match_id):
        self.bot.send_message(
            chat_id=call.message.chat.id,
            text="Task started. This may take around 5 minutes. Please wait...",
        )
        dota_api = Dota2API(steam_api_key)
        match = dota_api.build_single_match(match_id=match_id)
        message = ""
        message += f"<b>Match ID:</b> {match.match_id}\n"
        message += f"<b>Dire Team {Icons.direIcon} :</b> {match.dire_team.team_name} (ID: {match.dire_team.team_id})\n"
        message += "<b>Players:</b>\n"

        # List Dire team players
        for player in match.dire_team.players:
            message += (
                f"   - {player.name} {Icons.playerIcon}(Hero: {player.hero.name})\n"
            )

        message += f"\n<b>Radiant Team {Icons.radiantIcon}:</b> {match.radiant_team.team_name} (ID: {match.radiant_team.team_id})\n"
        message += "<b>Players:</b>\n"

        # List Radiant team players
        for player in match.radiant_team.players:
            message += (
                f"   - {player.name} {Icons.playerIcon}(Hero: {player.hero.name})\n"
            )

        # Prepare match data for prediction
        df, top_features = match.get_match_data_for_prediction()
        prediction = main_ml.predict(df)
        # Add the prediction to the message
        message += f"\n<b>Prediction:</b> {'Radiant Wins' if prediction[0] == 1 else 'Dire Wins'}\n"
        message += "<b>----------------------------------------</b>\n"  # Separator line in bold
        self.bot.send_message(
            chat_id=call.message.chat.id, text=message, parse_mode="HTML"
        )
