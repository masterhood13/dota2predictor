# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.
from time import sleep, strftime
import pandas as pd
import requests
import logging
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from config import opendota_key, steam_api_key
from db.database_operations import insert_match_result
from ml.model import MainML
from structure.helpers import (
    prepare_match_prediction_data,
    prepare_hero_pick_data,
    remove_special_chars,
)

logger = logging.getLogger(__name__)


class Dota2API:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = f"https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/?key={self.api_key}&dpc=true"
        logger.info("Dota2API initialized with provided API key.")

    def fetch_live_matches(self):
        """Fetch live matches data from the Dota 2 API."""
        logger.info("Fetching live matches from the Dota 2 API.")
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            logger.info("Successfully fetched live matches.")
            return response.json().get("result", {}).get("games", [])
        except requests.RequestException as e:
            logger.error(f"Error fetching live matches: {e}")
            return []

    def get_live_tournaments(self):
        """Fetch and build a list of live tournaments."""
        logger.info("Fetching live tournaments.")
        live_matches = self.fetch_live_matches()
        tournaments = self.build_tournaments(live_matches)
        logger.info("Live tournaments fetched successfully.")
        return tournaments

    def get_match_as_buttons(self, markup):
        """Build and add buttons for each live match."""
        logger.info("Building match buttons for live matches.")
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
                    logger.info(
                        f"Added button for match ID: {match_data.get('match_id')}"
                    )
        markup.add(Buttons.dota2_restart_button)
        logger.info("All match buttons added to markup.")
        return markup

    def get_hero_match_as_buttons(self, markup):
        """Build and add buttons for each live match."""
        logger.info("Building hero match buttons for live matches.")
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if self.is_valid_match(match_data):
                radiant_team, dire_team = match_data.get(
                    "radiant_team"
                ), match_data.get("dire_team")
                if radiant_team and dire_team:
                    markup.add(
                        Buttons.hero_match_button(
                            dire_team_name=dire_team.get("team_name", "Unknown"),
                            radiant_team_name=radiant_team.get("team_name", "Unknown"),
                            match_id=match_data.get("match_id"),
                        )
                    )
                    logger.info(
                        f"Added hero button for match ID: {match_data.get('match_id')}"
                    )
        markup.add(Buttons.dota2_restart_button)
        logger.info("All hero match buttons added to markup.")
        return markup

    def get_dota_plus_match_as_buttons(self, markup):
        """Build and add buttons for each live match."""
        logger.info("Building dota plus match buttons for live matches.")
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if self.is_valid_match(match_data):
                radiant_team, dire_team = match_data.get(
                    "radiant_team"
                ), match_data.get("dire_team")
                if radiant_team and dire_team:
                    markup.add(
                        Buttons.dota_plus_match_button(
                            dire_team_name=dire_team.get("team_name", "Unknown"),
                            radiant_team_name=radiant_team.get("team_name", "Unknown"),
                            match_id=match_data.get("match_id"),
                        )
                    )
                    logger.info(
                        f"Added dota plus button for match ID: {match_data.get('match_id')}"
                    )
        markup.add(Buttons.dota2_restart_button)
        logger.info("All dota plus match buttons added to markup.")
        return markup

    def build_single_match(self, match_id):
        """Build a single match object given the match_id."""
        logger.info(f"Building single match object for match ID: {match_id}.")
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if str(match_data.get("match_id")) == str(match_id) and self.is_valid_match(
                match_data
            ):
                logger.info(f"Match ID {match_id} found and is valid.")
                return self.create_match_object(match_data)
        logger.warning(f"Match ID {match_id} not found or invalid.")
        return None

    def get_single_match_online_data(self, match_id):
        logger.info(f"Get single match online data : {match_id}.")
        live_matches = self.fetch_live_matches()
        for match_data in live_matches:
            if str(match_data.get("match_id")) == str(match_id) and self.is_valid_match(
                match_data
            ):
                logger.info(f"Match ID {match_id} found and is valid.")
                return match_data
        logger.warning(f"Match ID {match_id} not found or invalid.")
        return None

    def build_tournaments(self, matches_data):
        """Organize matches into tournaments."""
        logger.info("Building tournaments from match data.")
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
                    logger.info(
                        f"Match ID {match.match_id} added to tournament {league_name}."
                    )

        logger.info(f"Total tournaments built: {len(tournaments)}.")
        return list(tournaments.values())

    def build_team(self, team_data, team_side, players_data):
        """Build a team object with players."""
        if not team_data:
            logger.warning("No team data provided for building a team.")
            return None

        logger.info(f"Building team for side {team_side}.")
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

        logger.info(f"Team {team.team_name} built with {len(team_players)} players.")
        return team

    def build_player(self, player_data):
        """Build a player object."""
        logger.info(
            f"Building player object for account ID: {player_data.get('account_id')}."
        )
        return Player(
            account_id=player_data.get("account_id"),
            name=player_data.get("name", "Unknown"),
            hero_id=player_data.get("hero_id", 0),
            team=player_data.get("team"),
        )

    def create_match_object(self, match_data):
        """Create a Match object from match data."""
        logger.info("Creating match object from match data.")
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
            logger.info(f"Match object created for match ID: {match.match_id}.")
            return match
        logger.warning("Could not create match object due to missing teams.")
        return None

    def is_valid_match(self, match_data):
        """Check if the match is valid (no player with hero_id = 0 on teams)."""
        logger.info(f"Validating match ID: {match_data.get('match_id')}.")
        players_data = match_data.get("players", [])
        invalid_players = [
            player
            for player in players_data
            if player["hero_id"] == 0 and player["team"] in (0, 1)
        ]

        if invalid_players:
            logger.warning(
                f"Skipping match {match_data.get('match_id')} due to invalid players."
            )
            return False
        logger.info(f"Match ID {match_data.get('match_id')} is valid.")
        return True


class CallbackTriggers:
    dota2_get_current_matches_trigger = "cb_dota2"
    predict_by_id_trigger = "cb_match_by_id"
    predict_pick_analyser_trigger = "cb_pick_analyser"
    dota_plus_trigger = "cb_dota_plus"
    match_trigger = "['cb_match_t'"
    hero_match_trigger = "['cb_hero_match_t'"
    dota_plus_match_trigger = "['cb_dota_plus_match_t'"
    get_history_of_predictions_trigger = "cb_history"


class Icons:
    playerIcon = "\U0001F468"
    direIcon = "\U0001F47F"
    radiantIcon = "\U0001F607"
    reload = "\U0001F503"
    history = "\U0001F4DC"
    statistic = "\U0001F4CA"
    stream = "\U0001F4F9"
    match = "\U0001F3C1"
    hero_strength = "\U0001F9BE"
    match_online = "\U0001F3AE"
    match_finished = "\U0001F3C6"
    match_tracking = "\U0001F525"


class Buttons:
    dota2_get_current_matches_button = InlineKeyboardButton(
        "Predict all matches at once",
        callback_data=CallbackTriggers.dota2_get_current_matches_trigger,
    )

    get_history_button = InlineKeyboardButton(
        f"Get predictions history {Icons.history}",
        callback_data=CallbackTriggers.get_history_of_predictions_trigger,
    )

    predict_by_id_button = InlineKeyboardButton(
        f"Predict match result {Icons.match}",
        callback_data=CallbackTriggers.predict_by_id_trigger,
    )

    predict_pick_analyser_button = InlineKeyboardButton(
        f"Analyse pick strength {Icons.hero_strength}",
        callback_data=CallbackTriggers.predict_pick_analyser_trigger,
    )

    dota_plus_button = InlineKeyboardButton(
        f"Track win probability during match (like Dota Plus) {Icons.stream}",
        callback_data=CallbackTriggers.dota_plus_trigger,
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

    @staticmethod
    def hero_match_button(dire_team_name, radiant_team_name, match_id):
        return InlineKeyboardButton(
            text=f"{match_id} | {Icons.direIcon}{dire_team_name} VS {Icons.radiantIcon}{radiant_team_name}",
            callback_data=f'{CallbackTriggers.hero_match_trigger},"{match_id}"]',
        )

    @staticmethod
    def dota_plus_match_button(dire_team_name, radiant_team_name, match_id):
        return InlineKeyboardButton(
            text=f"{match_id} | {Icons.direIcon}{dire_team_name} VS {Icons.radiantIcon}{radiant_team_name}",
            callback_data=f'{CallbackTriggers.dota_plus_match_trigger},"{match_id}"]',
        )


class Hero:
    def __init__(self, hero_id):
        self.hero_id = hero_id
        self.features = self.get_hero_features()
        self.name = self.features["name"] if self.features else "Unknown Hero"
        self.counter_picks = []

        if self.features and self.features["pro_pick"] > 0:
            self.winrate = self.features["pro_win"] / self.features["pro_pick"]
        else:
            self.winrate = 0

        logger.info(f"Initialized Hero: {self}")

    def get_hero_features(self):
        url = f"https://api.opendota.com/api/heroStats?api_key={opendota_key}"
        logger.info(f"Fetching hero features for Hero ID: {self.hero_id}")
        response = requests.get(url)

        if response.status_code == 200:
            heroes = response.json()
            for hero in heroes:
                if hero["id"] == self.hero_id:
                    logger.info(
                        f"Hero features retrieved for ID {self.hero_id}: {hero}"
                    )
                    return {
                        "hero_id": hero["id"],
                        "name": hero["localized_name"],
                        "pro_win": hero.get("pro_win", 0),
                        "pro_pick": hero.get("pro_pick", 0),
                    }
        else:
            logger.error(f"Error fetching hero features: {response.status_code}")
            return None

    def get_hero_matchups(self):
        url = f"https://api.opendota.com/api/heroes/{self.hero_id}/matchups?api_key={opendota_key}"
        logger.info(f"Fetching matchups for Hero ID: {self.hero_id}")
        response = requests.get(url)

        if response.status_code == 200:
            hero_matchups = response.json()
            logger.info(f"Matchups retrieved for Hero ID {self.hero_id}.")
            return hero_matchups
        else:
            logger.error(f"Error fetching hero matchups: {response.status_code}")
            return None

    def set_counter_pick_data(self, hero_against_ids):
        logger.info(f"Setting counter pick data for Hero ID: {self.hero_id}")
        hero_matchups = self.get_hero_matchups()
        if hero_matchups:
            for hero_matchup in hero_matchups:
                if hero_matchup["hero_id"] in hero_against_ids:
                    win_rate = (
                        hero_matchup["wins"] / hero_matchup["games_played"]
                        if hero_matchup["games_played"] > 0
                        else 0
                    )
                    self.counter_picks.append(
                        {"win_rate": win_rate, "hero_id": hero_matchup["hero_id"]}
                    )
                    logger.info(
                        f"Added counter pick for Hero ID: {hero_matchup['hero_id']} with win rate: {win_rate:.2f}"
                    )
        else:
            logger.warning(f"No matchups found for Hero ID: {self.hero_id}")

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
            # Initialize default values
            self.reset_stats()
            self.get_player_total_data()

        logger.info(f"Initialized Player: {self}")

    def reset_stats(self):
        """Reset all player statistics to zero."""
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

    def get_player_total_data(self):
        """Fetch player total data with retries on match data retrieval."""
        logger.info(
            f"Fetching total data for Player: {self.name} (ID: {self.account_id})"
        )
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
                logger.warning(f"Skipping match {match_id} after 5 attempts")
                continue  # Skip the match if it couldn't be retrieved

            # Get player data
            player_data = self.get_player_data(match_data)

            if player_data:
                logger.debug(
                    f"Processing match data for match ID {match_id}: {player_data}"
                )
                # Accumulate values safely
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

        logger.info(f"Completed data retrieval for Player: {self.name}")

    def fetch_recent_matches(self):
        """Fetch recent matches for the player."""
        logger.info(f"Fetching recent matches for Player ID: {self.account_id}")
        response = requests.get(
            f"https://api.opendota.com/api/players/{self.account_id}/matches?api_key={opendota_key}&limit=10&hero_id={self.hero.hero_id}&lobby_type=1"
        )
        if response.status_code == 200:
            logger.info(
                f"Recent matches fetched successfully for Player ID: {self.account_id}"
            )
            return response.json()
        else:
            logger.error(f"Error fetching recent matches: {response.status_code}")
            return []

    def fetch_match_data_with_retries(self, match_id):
        """Fetch match data with retries."""
        logger.info(f"Fetching match data for Match ID: {match_id}")
        retries = 0
        max_retries = 5

        while retries < max_retries:
            response = requests.get(
                f"https://api.opendota.com/api/matches/{match_id}?api_key={opendota_key}"
            )
            if response.status_code == 200:
                logger.info(f"Successfully fetched match data for Match ID: {match_id}")
                return response.json()  # Successful response
            else:
                retries += 1
                logger.warning(
                    f"Retrying... attempt {retries} for Match ID {match_id} (Status code: {response.status_code})"
                )
                sleep(2)  # Sleep for 2 seconds before retrying

        logger.error(
            f"Failed to fetch match data for Match ID {match_id} after {max_retries} attempts"
        )
        return None  # Return None if all retries fail

    def get_player_data(self, match_data):
        """Extract player data from match data."""
        players = match_data.get("players", [])
        player_data = next(
            (
                player
                for player in players
                if player.get("account_id") == self.account_id
            ),
            None,
        )
        if player_data:
            logger.info(
                f"Player data found for Account ID {self.account_id}: {player_data}"
            )
        else:
            logger.warning(
                f"No player data found for Account ID {self.account_id} in match data."
            )
        return player_data

    def accumulate_value(self, player_data, key, count):
        """Accumulate value for a given key and return updated count."""
        if key in player_data:
            try:
                setattr(
                    self, key, getattr(self, key) + player_data[key]
                )  # Update the stat
                count += 1  # Increment count
            except TypeError:
                logger.error(
                    f"TypeError while accumulating value for key {key}. Setting count to 1."
                )
                count = 1
        return count

    def calculate_average(self, total, count):
        """Calculate average, returning 0 if count is 0."""
        average = total / count if count > 0 else 0
        logger.debug(f"Calculated average: {average} (Total: {total}, Count: {count})")
        return average

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
        logger.info(f"Initialized Team: {self}")

    def add_player(self, player):
        self.players.append(player)
        logger.info(f"Added player {player.name} to team {self.team_name}")

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
        logger.info(f"Initialized Match: {self}")

    def get_match_data(self):
        logger.info(f"Fetching match data for match ID: {self.match_id}")
        url = f"https://api.opendota.com/api/matches/{self.match_id}?api_key={opendota_key}"
        response = requests.get(url)

        if response.status_code == 200:
            match_info = response.json()
            radiant_team = Team(
                match_info["radiant_name"], match_info["radiant_team_id"]
            )
            dire_team = Team(match_info["dire_name"], match_info["dire_team_id"])
            self.radiant_win = match_info["radiant_win"]
            logger.info(f"Match info retrieved: {match_info}")

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

                logger.info(
                    f"Added player: {player} to team: {radiant_team.team_name if player['isRadiant'] else dire_team.team_name}"
                )
            self.radiant_team = radiant_team
            self.dire_team = dire_team
            logger.info(f"Teams set: {self.radiant_team}, {self.dire_team}")
        else:
            logger.error(
                f"Failed to fetch match data: {response.status_code} - {response.text}"
            )

    def get_match_data_for_prediction(self):
        logger.info("Preparing match data for prediction.")
        if len(self.radiant_team.players) == 5 and len(self.dire_team.players) == 5:
            # Initialize the match_data dictionary
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

            # Convert to DataFrame
            df = pd.DataFrame([match_data])
            df = prepare_match_prediction_data(df, "scaler.pkl")
            logger.info("Match data prepared for prediction.")
            top_features = df.columns.tolist()
            return df, top_features
        else:
            logger.error("Both teams must have exactly 5 players.")
            raise ValueError("Both teams must have exactly 5 players.")

    @staticmethod
    def get_realtime_match_data_for_prediction_win_probability(match_steam_data):
        logger.info("Preparing match realtime data for probability prediction.")
        if (
            len(match_steam_data.get("scoreboard").get("radiant").get("players")) == 5
            and len(match_steam_data.get("scoreboard").get("dire").get("players")) == 5
        ):
            # Initialize the match_data dictionary
            match_data = {
                "match_id": match_steam_data.get("match_id"),
                "radiant_team_id": match_steam_data.get("radiant_team").get("team_id"),
                "radiant_team_name": match_steam_data.get("radiant_team").get(
                    "team_name"
                ),
                "dire_team_id": match_steam_data.get("dire_team").get("team_id"),
                "dire_team_name": match_steam_data.get("dire_team").get("team_name"),
            }

            # Add radiant team player data (5 players)
            for i, player in enumerate(
                match_steam_data.get("scoreboard").get("radiant").get("players")
            ):
                match_data[f"radiant_player_{i + 1}_id"] = player["account_id"]
                match_data[f"radiant_player_{i + 1}_name"] = 0
                match_data[f"radiant_player_{i + 1}_hero_id"] = player["hero_id"]
                match_data[f"radiant_player_{i + 1}_hero_name"] = 0
                match_data[f"radiant_player_{i + 1}_hero_winrate"] = 0
                match_data[f"radiant_player_{i + 1}_kills"] = player["kills"]
                match_data[f"radiant_player_{i + 1}_deaths"] = player["death"]
                match_data[f"radiant_player_{i + 1}_assists"] = player["assists"]
                match_data[f"radiant_player_{i + 1}_gold_per_min"] = player[
                    "gold_per_min"
                ]
                match_data[f"radiant_player_{i + 1}_xp_per_min"] = player["xp_per_min"]
                match_data[f"radiant_player_{i + 1}_teamfight_participation"] = 0
                match_data[f"radiant_player_{i + 1}_obs_placed"] = 0
                match_data[f"radiant_player_{i + 1}_sen_placed"] = 0
                match_data[f"radiant_player_{i + 1}_net_worth"] = player["net_worth"]
                match_data[f"radiant_player_{i + 1}_roshans_killed"] = 0
                match_data[f"radiant_player_{i + 1}_last_hits"] = player["last_hits"]
                match_data[f"radiant_player_{i + 1}_denies"] = player["denies"]
                match_data[f"radiant_player_{i + 1}_level"] = player["level"]
                match_data[f"radiant_player_{i + 1}_hero_damage"] = 0
                match_data[f"radiant_player_{i + 1}_tower_damage"] = 0

            # Add dire team player data (5 players)
            for i, player in enumerate(
                match_steam_data.get("scoreboard").get("dire").get("players")
            ):
                match_data[f"dire_player_{i + 1}_id"] = player["account_id"]
                match_data[f"dire_player_{i + 1}_name"] = 0
                match_data[f"dire_player_{i + 1}_hero_id"] = player["hero_id"]
                match_data[f"dire_player_{i + 1}_hero_name"] = 0
                match_data[f"dire_player_{i + 1}_hero_winrate"] = 0
                match_data[f"dire_player_{i + 1}_kills"] = player["kills"]
                match_data[f"dire_player_{i + 1}_deaths"] = player["death"]
                match_data[f"dire_player_{i + 1}_assists"] = player["assists"]
                match_data[f"dire_player_{i + 1}_gold_per_min"] = player["gold_per_min"]
                match_data[f"dire_player_{i + 1}_xp_per_min"] = player["xp_per_min"]
                match_data[f"dire_player_{i + 1}_teamfight_participation"] = 0
                match_data[f"dire_player_{i + 1}_obs_placed"] = 0
                match_data[f"dire_player_{i + 1}_sen_placed"] = 0
                match_data[f"dire_player_{i + 1}_net_worth"] = player["net_worth"]
                match_data[f"dire_player_{i + 1}_roshans_killed"] = 0
                match_data[f"dire_player_{i + 1}_last_hits"] = player["last_hits"]
                match_data[f"dire_player_{i + 1}_denies"] = player["denies"]
                match_data[f"dire_player_{i + 1}_level"] = player["level"]
                match_data[f"dire_player_{i + 1}_hero_damage"] = 0
                match_data[f"dire_player_{i + 1}_tower_damage"] = 0

            # Convert to DataFrame
            df = pd.DataFrame([match_data])
            df = prepare_match_prediction_data(df, "scaler.pkl")
            logger.info("Match data prepared for prediction.")
            top_features = df.columns.tolist()
            return df, top_features
        else:
            logger.error("Both teams must have exactly 5 players.")
            raise ValueError("Both teams must have exactly 5 players.")

    def get_hero_match_data_for_prediction(self):
        logger.info("Preparing hero match data for prediction.")
        if len(self.radiant_team.players) == 5 and len(self.dire_team.players) == 5:
            dire_hero_ids = [player.hero.hero_id for player in self.dire_team.players]
            radiant_hero_ids = [
                player.hero.hero_id for player in self.radiant_team.players
            ]

            [
                player.hero.set_counter_pick_data(radiant_hero_ids)
                for player in self.dire_team.players
            ]
            [
                player.hero.set_counter_pick_data(dire_hero_ids)
                for player in self.radiant_team.players
            ]
            match_data = {
                "match_id": self.match_id,
                "radiant_team_id": self.radiant_team.team_id,
                "radiant_team_name": self.radiant_team.team_name,
                "dire_team_id": self.dire_team.team_id,
                "dire_team_name": self.dire_team.team_name,
            }

            for i, player in enumerate(self.radiant_team.players):
                match_data[f"radiant_player_{i + 1}_hero_id"] = player.hero.hero_id
                match_data[f"radiant_player_{i + 1}_hero_name"] = player.hero.name
                match_data[f"radiant_player_{i + 1}_hero_winrate"] = player.hero.winrate
                for n, counter_pick in enumerate(player.hero.counter_picks):
                    match_data[f"radiant_hero_{i + 1}_{n + 1}_counter_pick"] = (
                        counter_pick["win_rate"]
                    )

            for i, player in enumerate(self.dire_team.players):
                match_data[f"dire_player_{i + 1}_hero_id"] = player.hero.hero_id
                match_data[f"dire_player_{i + 1}_hero_name"] = player.hero.name
                match_data[f"dire_player_{i + 1}_hero_winrate"] = player.hero.winrate
                for n, counter_pick in enumerate(player.hero.counter_picks):
                    match_data[f"dire_hero_{i + 1}_{n + 1}_counter_pick"] = (
                        counter_pick["win_rate"]
                    )

            df = pd.DataFrame([match_data])
            df = prepare_hero_pick_data(df)
            logger.info("Hero match data prepared for prediction.")
            top_features = df.columns.tolist()
            return df, top_features
        else:
            logger.error("Both teams must have exactly 5 players.")
            raise ValueError("Both teams must have exactly 5 players.")

    def set_hero_counter_picks(self):
        logger.info("Setting hero counter picks for players.")
        dire_hero_ids = [player.hero.hero_id for player in self.dire_team.players]
        radiant_hero_ids = [player.hero.hero_id for player in self.radiant_team.players]

        [
            player.hero.set_counter_pick_data(radiant_hero_ids)
            for player in self.dire_team.players
        ]
        [
            player.hero.set_counter_pick_data(dire_hero_ids)
            for player in self.radiant_team.players
        ]
        logger.info("Hero counter picks have been set.")

    def __repr__(self):
        radiant_players = "\n".join(
            [
                f"    Player: {player.name} (Hero: {player.hero.name})"
                for player in (self.radiant_team.players if self.radiant_team else [])
            ]
        )
        dire_players = "\n".join(
            [
                f"    Player: {player.name} (Hero: {player.hero.name})"
                for player in (self.dire_team.players if self.dire_team else [])
            ]
        )

        return (
            f"Match ID: {self.match_id}\n"
            f"League ID: {self.league_id}\n"
            f"Radiant Team: {self.radiant_team.team_name if self.radiant_team else 'Not Set'}\n"
            f"Radiant Players:\n{radiant_players or 'No Players'}\n"
            f"Dire Team: {self.dire_team.team_name if self.dire_team else 'Not Set'}\n"
            f"Dire Players:\n{dire_players or 'No Players'}\n"
            f"Radiant Win: {'Yes' if self.radiant_win else 'No'}"
        )


class Tournament:
    def __init__(self, league_id: int, name: str):
        self.league_id = league_id
        self.name = name
        self.matches = []
        logger.info(f"Tournament initialized: {self.name} (ID: {self.league_id})")

    def add_match(self, match):
        self.matches.append(match)
        logger.info(f"Match added: {match.match_id} to tournament {self.name}")

    def get_league_matches(self):
        url = f"https://api.opendota.com/api/leagues/{self.league_id}/matches?api_key={opendota_key}"
        logger.info(f"Fetching matches for league {self.league_id} from {url}")
        response = requests.get(url)
        if response.status_code == 200:
            for match_info in response.json():
                logger.debug(f"Match info received: {match_info}")
                match_id = match_info["match_id"]
                radiant_team_id = match_info["radiant_team_id"]
                dire_team_id = match_info["dire_team_id"]
                radiant_win = match_info["radiant_win"]

                match = Match(
                    match_id, radiant_team_id, dire_team_id, self.league_id, radiant_win
                )

                try:
                    match.get_match_data()
                    match.set_hero_counter_picks()
                    self.add_match(match)
                    logger.info(f"Match {match_id} successfully added to tournament.")
                except (TypeError, KeyError) as e:
                    logger.warning(
                        f"Error processing match {match_id}: {str(e)} - Skipping this match."
                    )
        else:
            logger.error(
                f"Error fetching matches for league {self.league_id}: {response.status_code}"
            )

    def __repr__(self):
        return f"Tournament({self.name}, ID: {self.league_id})"


class Markups:
    def __init__(self, bot):
        self.markup = InlineKeyboardMarkup()
        self.markup.row_width = 8
        self.bot = bot
        logger.info("Markups class initialized.")

    def gen_main_markup(self, current_user_id, current_channel_id):
        logger.info(
            f"Generating main markup for user {current_user_id} in channel {current_channel_id}."
        )
        self.markup.add(Buttons.get_history_button)
        self.markup.add(Buttons.dota2_get_current_matches_button)
        self.markup.add(Buttons.predict_by_id_button)
        self.markup.add(Buttons.predict_pick_analyser_button)
        self.markup.add(Buttons.dota_plus_button)
        return self.markup

    def gen_dota2_matches_markup(self, call):
        logger.info("Generating Dota2 matches markup.")
        dota_api = Dota2API(steam_api_key)
        self.bot.send_message(
            chat_id=call.message.chat.id,
            text="<b>Task started. This may take around 5 minutes. Please wait...</b>",
            parse_mode="HTML",
        )

        try:
            tournaments = dota_api.get_live_tournaments()
            for tournament in tournaments:
                for match in tournament.matches:
                    message = (
                        f"<b>Tournament:</b> {tournament.name}\n"
                        f"<b>League ID:</b> {tournament.league_id}\n\n"
                        f"<b>Match ID:</b> {match.match_id}\n"
                        f"<b>Dire Team {Icons.direIcon}:</b> {match.dire_team.team_name} (ID: {match.dire_team.team_id})\n"
                        "<b>Players:</b>\n"
                    )

                    # List Dire team players
                    for player in match.dire_team.players:
                        message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

                    message += (
                        f"\n<b>Radiant Team {Icons.radiantIcon}:</b> {match.radiant_team.team_name} (ID: {match.radiant_team.team_id})\n"
                        "<b>Players:</b>\n"
                    )

                    # List Radiant team players
                    for player in match.radiant_team.players:
                        message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

                    # Prepare match data for prediction
                    df, top_features = match.get_match_data_for_prediction()
                    main_ml = MainML(None, "xgb_model.pkl")
                    main_ml.load_model()
                    prediction, probabilities = main_ml.predict(df)

                    logger.debug(f"Prediction for match {match.match_id}: {prediction}")

                    # Add the prediction to the message
                    message += f"\n<b>Prediction:</b> {'Radiant Wins' if prediction[0] == 1 else 'Dire Wins'}\n"
                    radiant_prob = probabilities[0][1]  # Assuming class 1 is Radiant
                    dire_prob = probabilities[0][0]  # Assuming class 0 is Dire
                    message += f"<b>Probabilities:</b> Radiant: {radiant_prob:.2%}, Dire: {dire_prob:.2%}\n"
                    message += "<b>----------------------------------------</b>\n"  # Separator line in bold

                    # Log the message text
                    logger.info(
                        f"Sending message to chat {call.message.chat.id}: {message}"
                    )
                    self.bot.send_message(
                        chat_id=call.message.chat.id, text=message, parse_mode="HTML"
                    )
            self.bot.send_message(
                chat_id=call.message.chat.id, text="<b>DONE</b>", parse_mode="HTML"
            )
            logger.info("Dota2 matches markup generation completed.")
        except Exception as e:
            logger.error(f"Error while generating Dota2 matches markup: {str(e)}")

    def gen_match_markup_by_id(self, call):
        logger.info(f"Generating match markup by ID for call: {call}")
        dota_api = Dota2API(steam_api_key)
        self.markup = dota_api.get_match_as_buttons(self.markup)
        return self.markup

    def gen_hero_match_markup_by_id(self, call):
        logger.info(f"Generating hero match markup by ID for call: {call}")
        dota_api = Dota2API(steam_api_key)
        self.markup = dota_api.get_hero_match_as_buttons(self.markup)
        return self.markup

    def gen_dota_plus_match_markup_by_id(self, call):
        logger.info(f"Generating dota plus match markup by ID for call: {call}")
        dota_api = Dota2API(steam_api_key)
        self.markup = dota_api.get_dota_plus_match_as_buttons(self.markup)
        return self.markup

    def make_prediction_for_selected_match(self, call, match_id):
        logger.info(f"Making prediction for selected match ID: {match_id}")
        self.bot.send_message(
            chat_id=call.message.chat.id,
            text="Task started. This may take around 5 minutes. Please wait...",
        )
        dota_api = Dota2API(steam_api_key)
        match = dota_api.build_single_match(match_id=match_id)
        message = (
            f"<b>Match ID:</b> {match.match_id}\n"
            f"<b>Dire Team {Icons.direIcon}:</b> {match.dire_team.team_name} (ID: {match.dire_team.team_id})\n"
            "<b>Players:</b>\n"
        )

        # List Dire team players
        for player in match.dire_team.players:
            message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

        message += (
            f"\n<b>Radiant Team {Icons.radiantIcon}:</b> {match.radiant_team.team_name} (ID: {match.radiant_team.team_id})\n"
            "<b>Players:</b>\n"
        )

        # List Radiant team players
        for player in match.radiant_team.players:
            message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

        # Prepare match data for prediction
        df, top_features = match.get_match_data_for_prediction()
        main_ml = MainML(None, "xgb_model.pkl")
        main_ml.load_model()
        prediction, probabilities = main_ml.predict(df)

        row = df.iloc[0]  # Access the first row of the DataFrame

        model_prediction = prediction[0]

        radiant_avg_hero_winrate = row["radiant_avg_hero_winrate"]
        radiant_avg_roshans_killed = row["radiant_avg_roshans_killed"]
        radiant_avg_last_hits = row["radiant_avg_last_hits"]
        radiant_avg_denies = row["radiant_avg_denies"]
        radiant_avg_hero_damage = row["radiant_avg_hero_damage"]
        radiant_avg_gpm = row["radiant_avg_gpm"]
        radiant_avg_xpm = row["radiant_avg_xpm"]
        radiant_avg_net_worth = row["radiant_avg_net_worth"]
        radiant_avg_player_level = row["radiant_avg_player_level"]
        radiant_sum_obs = row["radiant_sum_obs"]
        radiant_sum_sen = row["radiant_sum_sen"]
        radiant_avg_teamfight_participation_cols = row[
            "radiant_avg_teamfight_participation_cols"
        ]

        dire_avg_hero_winrate = row["dire_avg_hero_winrate"]
        dire_avg_roshans_killed = row["dire_avg_roshans_killed"]
        dire_avg_last_hits = row["dire_avg_last_hits"]
        dire_avg_denies = row["dire_avg_denies"]
        dire_avg_hero_damage = row["dire_avg_hero_damage"]
        dire_avg_gpm = row["dire_avg_gpm"]
        dire_avg_xpm = row["dire_avg_xpm"]
        dire_avg_net_worth = row["dire_avg_net_worth"]
        dire_avg_player_level = row["dire_avg_player_level"]
        dire_sum_obs = row["dire_sum_obs"]
        dire_sum_sen = row["dire_sum_sen"]
        dire_avg_teamfight_participation_cols = row[
            "dire_avg_teamfight_participation_cols"
        ]

        radiant_avg_kda = row["radiant_avg_kda"]
        dire_avg_kda = row["dire_avg_kda"]

        # Call the insert function to add the match result to the database
        insert_match_result(
            match_id=match_id,
            model_prediction=model_prediction,
            radiant_avg_hero_winrate=radiant_avg_hero_winrate,
            radiant_avg_roshans_killed=radiant_avg_roshans_killed,
            radiant_avg_last_hits=radiant_avg_last_hits,
            radiant_avg_denies=radiant_avg_denies,
            radiant_avg_hero_damage=radiant_avg_hero_damage,
            radiant_avg_gpm=radiant_avg_gpm,
            radiant_avg_xpm=radiant_avg_xpm,
            radiant_avg_net_worth=radiant_avg_net_worth,
            radiant_avg_player_level=radiant_avg_player_level,
            radiant_sum_obs=radiant_sum_obs,
            radiant_sum_sen=radiant_sum_sen,
            radiant_avg_teamfight_participation_cols=radiant_avg_teamfight_participation_cols,
            dire_avg_hero_winrate=dire_avg_hero_winrate,
            dire_avg_roshans_killed=dire_avg_roshans_killed,
            dire_avg_last_hits=dire_avg_last_hits,
            dire_avg_denies=dire_avg_denies,
            dire_avg_hero_damage=dire_avg_hero_damage,
            dire_avg_gpm=dire_avg_gpm,
            dire_avg_xpm=dire_avg_xpm,
            dire_avg_net_worth=dire_avg_net_worth,
            dire_avg_player_level=dire_avg_player_level,
            dire_sum_obs=dire_sum_obs,
            dire_sum_sen=dire_sum_sen,
            dire_avg_teamfight_participation_cols=dire_avg_teamfight_participation_cols,
            radiant_avg_kda=radiant_avg_kda,
            dire_avg_kda=dire_avg_kda,
        )

        message += f"\n<b>Prediction:</b> {'Radiant Wins' if prediction[0] == 1 else 'Dire Wins'}\n"
        radiant_prob = probabilities[0][1]  # Assuming class 1 is Radiant
        dire_prob = probabilities[0][0]  # Assuming class 0 is Dire
        message += f"<b>Probabilities:</b> Radiant: {radiant_prob:.2%}, Dire: {dire_prob:.2%}\n"
        message += "<b>----------------------------------------</b>\n"  # Separator line in bold

        # Log the message text
        logger.info(f"Sending message to chat {call.message.chat.id}: {message}")
        self.bot.send_message(
            chat_id=call.message.chat.id, text=message, parse_mode="HTML"
        )
        logger.info(f"Prediction for match ID {match_id} sent successfully.")

    def make_hero_pick_prediction_for_selected_match(self, call, match_id):
        logger.info(f"Making hero pick prediction for match ID: {match_id}")
        self.bot.send_message(
            chat_id=call.message.chat.id,
            text="Task started. This may take around 5 minutes. Please wait...",
        )
        dota_api = Dota2API(steam_api_key)
        match = dota_api.build_single_match(match_id=match_id)
        message = (
            f"<b>Match ID:</b> {match.match_id}\n"
            f"<b>Dire Team {Icons.direIcon}:</b> {match.dire_team.team_name} (ID: {match.dire_team.team_id})\n"
            "<b>Players:</b>\n"
        )

        # List Dire team players
        for player in match.dire_team.players:
            message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

        message += (
            f"\n<b>Radiant Team {Icons.radiantIcon}:</b> {match.radiant_team.team_name} (ID: {match.radiant_team.team_id})\n"
            "<b>Players:</b>\n"
        )

        # List Radiant team players
        for player in match.radiant_team.players:
            message += f"   - {remove_special_chars(player.name)} {Icons.playerIcon}(Hero: {player.hero.name})\n"

        # Prepare match data for prediction
        df, top_features = match.get_hero_match_data_for_prediction()
        hero_pick_ml = MainML(None, "xgb_model_hero_pick.pkl")
        hero_pick_ml.load_model()
        prediction, _ = hero_pick_ml.predict(df)
        message += f"\n<b>Prediction:</b> {'Radiant pick is stronger' if prediction[0] == 1 else 'Dire pick is stronger'}\n"
        message += "<b>----------------------------------------</b>\n"  # Separator line in bold

        # Log the message text
        logger.info(f"Sending message to chat {call.message.chat.id}: {message}")
        self.bot.send_message(
            chat_id=call.message.chat.id, text=message, parse_mode="HTML"
        )
        logger.info(f"Hero pick prediction for match ID {match_id} sent successfully.")

    def follow_dota_plus_for_selected_match(self, call, match_id):
        logger.info(f"Follow dota plus for match ID: {match_id}")
        dota_api = Dota2API(steam_api_key)
        msg = self.bot.send_message(
            call.message.chat.id,
            f"{Icons.match_online} Match {match_id} is live! {Icons.match_tracking} Tracking win probability...",
        )
        main_ml = MainML(None, "xgb_model.pkl")
        main_ml.load_model()

        while True:
            match_data = dota_api.get_single_match_online_data(match_id=match_id)
            df, top_features = (
                Match.get_realtime_match_data_for_prediction_win_probability(match_data)
            )
            prediction, probabilities = main_ml.predict(df)
            # Check if the match is finished
            if not match_data:
                self.bot.edit_message_text(
                    chat_id=call.message.chat.id,
                    message_id=msg.message_id,
                    text=f"{Icons.match_finished} Match {match_id} finished!"
                    f"Radiant Team {Icons.radiantIcon}| {match_data.get('radiant_team').get('team_name')} vs Dire Team {Icons.direIcon}:|  {match_data.get('dire_team').get('team_name')}\n"
                    f"Final in game {match_data.get('scoreboard').get('duration')/ 60:.2f}\n"
                    f"Last update time: {strftime('%H:%M:%S')}"
                    f"Final win probability: Probabilities: Radiant: {probabilities[0][1]:.2%}, Dire: {probabilities[0][0] :.2%}\n",
                )
                break  # Exit the loop when the match is finished

            self.bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=msg.message_id,
                text=f"{Icons.match_online} Match is live! {Icons.match_tracking} Tracking win probability...\n"
                f"Radiant Team {Icons.radiantIcon}| {match_data.get('radiant_team').get('team_name')} vs Dire Team {Icons.direIcon}:|  {match_data.get('dire_team').get('team_name')}\n"
                f"Probabilities: Radiant: {probabilities[0][1]:.2%}, Dire: {probabilities[0][0] :.2%}\n"
                f"Time in game {match_data.get('scoreboard').get('duration')/ 60:.2f}\n"
                f"Last update time: {strftime('%H:%M:%S')}",
            )

            sleep(60)
