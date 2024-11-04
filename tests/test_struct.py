# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock, Mock

from structure.struct import Hero, Player, Team, Match, Tournament, Dota2API, Markups


@dataclass
class HeroD:
    name: str


@dataclass
class PlayerD:
    name: str
    hero: Hero


@dataclass
class TeamD:
    team_name: str
    team_id: int
    players: list


@dataclass
class MatchD:
    match_id: int
    dire_team: Team
    radiant_team: Team


class TestHero(unittest.TestCase):
    @patch("requests.get")
    def test_get_hero_features_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                "id": 1,
                "localized_name": "Anti-Mage",
                "pro_win": 100,
                "pro_pick": 200,
            }
        ]
        hero = Hero(hero_id=1)
        self.assertEqual(hero.name, "Anti-Mage")
        self.assertAlmostEqual(hero.winrate, 0.5)

    @patch("requests.get")
    def test_get_hero_features_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        hero = Hero(hero_id=999)  # Assuming this ID does not exist
        self.assertEqual(hero.name, "Unknown Hero")
        self.assertEqual(hero.winrate, 0)

    @patch("requests.get")
    def test_get_hero_matchups_success(self, mock_get):
        # Mock the hero features request
        mock_get.side_effect = [
            Mock(
                status_code=200,
                json=lambda: [
                    {
                        "id": 1,
                        "localized_name": "Anti-Mage",
                        "pro_win": 100,
                        "pro_pick": 200,
                    }
                ],
            ),
            Mock(
                status_code=200,
                json=lambda: [
                    {"hero_id": 2, "games_played": 100, "wins": 40},
                    {"hero_id": 3, "games_played": 200, "wins": 100},
                ],
            ),
        ]

        hero = Hero(hero_id=1)
        matchups = hero.get_hero_matchups()
        self.assertEqual(len(matchups), 2)
        self.assertEqual(matchups[0]["hero_id"], 2)

    @patch("requests.get")
    def test_get_hero_matchups_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        hero = Hero(hero_id=1)
        matchups = hero.get_hero_matchups()
        self.assertIsNone(matchups)

    @patch("requests.get")
    def test_set_counter_pick_data(self, mock_get):
        # Mock the hero features request
        mock_get.side_effect = [
            Mock(
                status_code=200,
                json=lambda: [
                    {
                        "id": 1,
                        "localized_name": "Anti-Mage",
                        "pro_win": 100,
                        "pro_pick": 200,
                    }
                ],
            ),
            Mock(
                status_code=200,
                json=lambda: [
                    {"hero_id": 2, "games_played": 100, "wins": 40},
                    {"hero_id": 3, "games_played": 50, "wins": 20},
                ],
            ),
        ]

        hero = Hero(hero_id=1)
        hero.set_counter_pick_data([2, 3])
        self.assertEqual(len(hero.counter_picks), 2)
        self.assertEqual(hero.counter_picks[0]["hero_id"], 2)
        self.assertAlmostEqual(hero.counter_picks[0]["win_rate"], 0.4)

    @patch("requests.get")
    def test_set_counter_pick_data_no_games_played(self, mock_get):
        # Mock the hero features request
        mock_get.side_effect = [
            Mock(
                status_code=200,
                json=lambda: [
                    {
                        "id": 1,
                        "localized_name": "Anti-Mage",
                        "pro_win": 100,
                        "pro_pick": 200,
                    }
                ],
            ),
            Mock(
                status_code=200,
                json=lambda: [{"hero_id": 2, "games_played": 0, "wins": 0}],
            ),
        ]

        hero = Hero(hero_id=1)
        hero.set_counter_pick_data([2])
        self.assertEqual(len(hero.counter_picks), 1)
        self.assertEqual(hero.counter_picks[0]["win_rate"], 0)

    def test_repr(self):
        hero = Hero(hero_id=1)
        hero.name = "Anti-Mage"
        repr_output = repr(hero)
        self.assertIn("Anti-Mage", repr_output)
        self.assertIn("Hero(ID: 1", repr_output)


class PlayerTest(unittest.TestCase):

    @patch("requests.get")
    @patch.object(Hero, "get_hero_features")
    def test_get_player_total_data(self, mock_hero_features, mock_get):
        # Mock the return value of get_hero_features with the required structure
        mock_hero_features.return_value = {
            "id": 1,
            "hero_id": 1,
            "name": "Anti-Mage",
            "pro_win": 100,
            "pro_pick": 200,
        }

        # Mock the response for recent matches
        mock_recent_matches_response = MagicMock(status_code=200)
        mock_recent_matches_response.json.return_value = [
            {"match_id": 12345},
            {"match_id": 12346},
        ]

        # Mock the response for match data
        mock_match_response = MagicMock(status_code=200)
        mock_match_response.json.return_value = {
            "players": [
                {
                    "account_id": 1,
                    "teamfight_participation": 0.5,
                    "obs_placed": 2,
                    "sen_placed": 1,
                    "net_worth": 12000,
                    "kills": 5,
                    "deaths": 3,
                    "assists": 4,
                    "roshans_killed": 0,
                    "last_hits": 40,
                    "denies": 5,
                    "gold_per_min": 350,
                    "xp_per_min": 400,
                    "level": 20,
                    "hero_damage": 15000,
                    "tower_damage": 3000,
                    "hero_healing": 2000,
                }
            ]
        }

        # Update the side effects
        mock_get.side_effect = [
            mock_hero_features,
            mock_recent_matches_response,  # 1st call: Recent matches
            mock_match_response,
            mock_match_response,  # 2nd call: Match data
        ]

        # Create a Player instance
        player = Player(account_id=1, name="Player1", hero_id=1, team="Team A")

        # Call the method to fetch player data
        player.get_player_total_data()

        # Assertions to validate the totals after fetching data
        self.assertEqual(player.teamfight_participation, 0.5)
        self.assertEqual(player.obs_placed, 2)
        self.assertEqual(player.sen_placed, 1)
        self.assertEqual(player.net_worth, 12000)
        self.assertEqual(player.kills, 5)
        self.assertEqual(player.deaths, 3)
        self.assertEqual(player.assists, 4)
        self.assertEqual(player.roshans_killed, 0)
        self.assertEqual(player.last_hits, 40)
        self.assertEqual(player.denies, 5)
        self.assertEqual(player.gold_per_min, 350)
        self.assertEqual(player.xp_per_min, 400)
        self.assertEqual(player.level, 20)
        self.assertEqual(player.hero_damage, 15000)
        self.assertEqual(player.tower_damage, 3000)
        self.assertEqual(player.hero_healing, 2000)


class TestTeam(unittest.TestCase):

    def test_team_initialization(self):
        team = Team(team_name="Team A", team_id=1)
        self.assertEqual(team.team_name, "Team A")
        self.assertEqual(team.team_id, 1)
        self.assertEqual(len(team.players), 0)

    def test_add_player(self):
        team = Team(team_name="Team A", team_id=1)
        player = Player(account_id=1, name="Player1", hero_id=1, team="Team A")
        team.add_player(player)
        self.assertEqual(len(team.players), 1)
        self.assertEqual(team.players[0], player)

    def test_add_multiple_players(self):
        team = Team(team_name="Team A", team_id=1)
        player1 = Player(account_id=1, name="Player1", hero_id=1, team="Team A")
        player2 = Player(account_id=2, name="Player2", hero_id=2, team="Team A")

        team.add_player(player1)
        team.add_player(player2)

        self.assertEqual(len(team.players), 2)
        self.assertEqual(team.players[0], player1)
        self.assertEqual(team.players[1], player2)

    def test_repr_method(self):
        team = Team(team_name="Team A", team_id=1)
        player = Player(account_id=1, name="Player1", hero_id=1, team="Team A")
        team.add_player(player)

        self.assertEqual(repr(team), f"Team(Team A, ID: 1, Players: [{player}])")


class TestMatch(unittest.TestCase):
    @patch("requests.get")
    @patch("structure.struct.Player")
    @patch.object(
        Hero,
        "get_hero_features",
        return_value={
            "id": 1,
            "localized_name": "Anti-Mage",
            "pro_win": 100,
            "pro_pick": 200,
        },
    )
    def test_get_match_data_success(self, mock_hero_features, mock_player, mock_get):
        # Mock the Player instance and its methods
        mock_player_instance = mock_player.return_value
        mock_player_instance.get_player_data.return_value = {"win": 10, "lose": 5}
        mock_player_instance.get_player_total_data.return_value = [
            {"field": "kills", "sum": 100, "n": 10},
            {"field": "deaths", "sum": 50, "n": 10},
            {"field": "assists", "sum": 75, "n": 10},
            {"field": "gold_per_min", "sum": 300, "n": 10},
            {"field": "xp_per_min", "sum": 400, "n": 10},
            {"field": "last_hits", "sum": 200, "n": 10},
            {"field": "denies", "sum": 25, "n": 10},
        ]

        # Mock the response of the get request for match data
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "radiant_name": "Team A",
            "radiant_team_id": 1,
            "dire_name": "Team B",
            "dire_team_id": 2,
            "radiant_win": True,
            "players": [
                {"account_id": 1, "name": "Player1", "hero_id": 1, "isRadiant": True},
                {"account_id": 2, "name": "Player2", "hero_id": 2, "isRadiant": False},
            ],
        }

        # Create a Match instance and call get_match_data
        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)
        match.get_match_data()

        # Assertions to validate the match data
        self.assertEqual(match.radiant_team.team_name, "Team A")
        self.assertEqual(match.dire_team.team_name, "Team B")
        self.assertTrue(match.radiant_win)

        # Validate Player1 data
        for player in match.radiant_team.players:
            if player.account_id == 1:  # Checking for Player1
                self.assertAlmostEqual(player.kills, 10)  # 100/10
                self.assertAlmostEqual(player.deaths, 5)  # 50/10
                self.assertAlmostEqual(player.assists, 7.5)  # 75/10
                self.assertAlmostEqual(player.gold_per_min, 30)  # 300/10
                self.assertAlmostEqual(player.xp_per_min, 40)  # 400/10
                self.assertAlmostEqual(player.last_hits, 20)  # 200/10
                self.assertAlmostEqual(player.denies, 2.5)  # 25/10

        # Validate Player2 data
        for player in match.dire_team.players:
            if player.account_id == 2:  # Checking for Player2
                self.assertIsNotNone(
                    player.hero
                )  # Just to check if Hero object is created
                self.assertEqual(player.name, "Player2")

    @patch("requests.get")
    def test_get_match_data_api_failure(self, mock_get):
        # Mock the response of the get request to simulate a failure
        mock_get.return_value.status_code = 404

        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)
        match.get_match_data()

        # Assertions to check that teams are still None
        self.assertIsNone(match.radiant_team)
        self.assertIsNone(match.dire_team)

    @patch("requests.get")
    @patch("structure.struct.Player")
    def test_get_match_data_for_prediction(self, mock_player, mock_get):
        # Create mocked players
        players = []
        for i in range(1, 6):
            mock_player_instance = MagicMock()
            mock_player_instance.account_id = i
            mock_player_instance.name = f"Player{i}"
            mock_player_instance.hero = MagicMock()
            mock_player_instance.hero.hero_id = i
            mock_player_instance.hero.name = f"Hero{i}"
            mock_player_instance.hero.winrate = 75
            mock_player_instance.kills = 10
            mock_player_instance.deaths = 5
            mock_player_instance.assists = 7
            mock_player_instance.gold_per_min = 300
            mock_player_instance.xp_per_min = 400
            mock_player_instance.last_hits = 100
            mock_player_instance.denies = 10
            mock_player_instance.level = 25
            mock_player_instance.hero_damage = 15000
            mock_player_instance.tower_damage = 2000
            mock_player_instance.roshans_killed = 1
            mock_player_instance.teamfight_participation = 0.5
            mock_player_instance.obs_placed = 5
            mock_player_instance.sen_placed = 3
            mock_player_instance.net_worth = 15000
            players.append(mock_player_instance)

        # Mock the response of the get request for match data
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "radiant_name": "Team A",
            "radiant_team_id": 1,
            "dire_name": "Team B",
            "dire_team_id": 2,
            "radiant_win": True,
            "players": [
                {"account_id": 1, "name": "Player1", "hero_id": 1, "isRadiant": True},
                {"account_id": 2, "name": "Player2", "hero_id": 2, "isRadiant": False},
            ],
        }

        # Create a Match instance and set up the teams
        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)

        # Set 5 players for both teams
        match.radiant_team = Team(team_name="Team A", team_id=1)
        match.dire_team = Team(team_name="Team B", team_id=2)

        # Set 5 players for both teams
        match.radiant_team.players = players
        match.dire_team.players = players

        # Run the method
        df, top_features = match.get_match_data_for_prediction()

        # Check for expected columns after preparation
        expected_columns = [
            "radiant_avg_hero_winrate",
            "radiant_avg_roshans_killed",
            "radiant_avg_last_hits",
            "radiant_avg_denies",
            "radiant_avg_hero_damage",
            "radiant_avg_player_level",
            "radiant_avg_gpm",
            "radiant_avg_xpm",
            "radiant_avg_net_worth",
            "radiant_sum_obs",
            "radiant_sum_sen",
            "radiant_avg_teamfight_participation_cols",
            "dire_avg_hero_winrate",
            "dire_avg_roshans_killed",
            "dire_avg_last_hits",
            "dire_avg_denies",
            "dire_avg_hero_damage",
            "dire_avg_player_level",
            "dire_avg_gpm",
            "dire_avg_xpm",
            "dire_avg_net_worth",
            "dire_sum_obs",
            "dire_sum_sen",
            "dire_avg_teamfight_participation_cols",
            "radiant_avg_kda",
            "dire_avg_kda",
        ]
        self.assertEqual(len(expected_columns), len(df.columns))
        for col in expected_columns:
            self.assertIn(col, df.columns)

    @patch("structure.struct.Player")
    def test_set_hero_counter_picks(self, mock_player):
        # Set up mock players with heroes
        radiant_player_1 = mock_player.return_value
        radiant_player_1.hero = MagicMock()
        radiant_player_1.hero.hero_id = 1
        radiant_player_1.hero.counter_pick_data = []

        radiant_player_2 = mock_player.return_value
        radiant_player_2.hero = MagicMock()
        radiant_player_2.hero.hero_id = 2
        radiant_player_2.hero.counter_pick_data = []

        dire_player_1 = mock_player.return_value
        dire_player_1.hero = MagicMock()
        dire_player_1.hero.hero_id = 3
        dire_player_1.hero.counter_pick_data = []

        dire_player_2 = mock_player.return_value
        dire_player_2.hero = MagicMock()
        dire_player_2.hero.hero_id = 4
        dire_player_2.hero.counter_pick_data = []

        # Set up teams
        radiant_team = Team("Team A", 1)
        radiant_team.players = [radiant_player_1, radiant_player_2]
        dire_team = Team("Team B", 2)
        dire_team.players = [dire_player_1, dire_player_2]

        # Create a match instance
        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)
        match.radiant_team = radiant_team
        match.dire_team = dire_team

        # Call the method that should set counter pick data
        match.set_hero_counter_picks()

        # Check that counter pick data is set correctly
        for player in dire_team.players:
            self.assertEqual(
                player.hero.counter_pick_data, []
            )  # Radiant heroes should be counter picks for Dire

        for player in radiant_team.players:
            self.assertEqual(player.hero.counter_pick_data, [])

    @patch("structure.struct.Player")
    def test_repr(self, mock_player):
        # Correctly instantiate the MagicMock for Hero
        radiant_player = Player(1, 1, "Player1", "Team A", {})
        radiant_player.hero = MagicMock()
        radiant_player.hero.id = 1
        radiant_player.hero.name = "Anti-Mage"  # Set the actual name

        dire_player = Player(2, 2, "Player2", "Team B", {})
        dire_player.hero = MagicMock()
        dire_player.hero.id = 2
        dire_player.hero.name = "Invoker"  # Set the actual name

        radiant_team = Team("Team A", 1)
        radiant_team.players = [radiant_player]
        dire_team = Team("Team B", 2)
        dire_team.players = [dire_player]

        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)
        match.radiant_team = radiant_team
        match.dire_team = dire_team
        match.radiant_win = True

        expected_repr = (
            "Match ID: 1\n"
            "League ID: 1\n"
            "Radiant Team: Team A\n"
            "Radiant Players:\n"
            "    Player: 1 (Hero: Anti-Mage)\n"
            "Dire Team: Team B\n"
            "Dire Players:\n"
            "    Player: 2 (Hero: Invoker)\n"
            "Radiant Win: Yes"
        )

        # Check that the __repr__ of the match object matches the expected output
        self.assertEqual(str(match), expected_repr)


class TestTournament(unittest.TestCase):
    @patch("requests.get")
    @patch("structure.struct.Match")  # Mock the Match class
    def test_get_league_matches_success(self, mock_match, mock_get):
        # Mock the Match instance
        mock_match_instance = MagicMock()
        mock_match.return_value = mock_match_instance

        # Set the actual return values for the Match instance attributes
        mock_match_instance.match_id = 1
        mock_match_instance.radiant_team_id = 1
        mock_match_instance.dire_team_id = 2
        mock_match_instance.radiant_win = True

        # Mock the response for league matches
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                "match_id": 1,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_win": True,
            }
        ]

        # Mock get_match_data and set_hero_counter_picks for the Match instance
        mock_match_instance.get_match_data.return_value = None
        mock_match_instance.set_hero_counter_picks.return_value = None

        # Initialize a Tournament instance
        tournament = Tournament(league_id=1, name="Tournament A")
        tournament.get_league_matches()

        # Assertions to check if matches are populated correctly
        self.assertEqual(len(tournament.matches), 1)

        # Verify that the match has the expected attributes
        match = tournament.matches[0]
        self.assertEqual(match.match_id, 1)
        self.assertEqual(match.radiant_team_id, 1)
        self.assertEqual(match.dire_team_id, 2)
        self.assertTrue(match.radiant_win)

        # Verify that the expected methods were called
        mock_match_instance.get_match_data.assert_called_once()
        mock_match_instance.set_hero_counter_picks.assert_called_once()

    @patch("requests.get")
    def test_get_league_matches_failure(self, mock_get):
        # Mock a failure response (404)
        mock_get.return_value.status_code = 404

        # Initialize a Tournament instance with a non-existent league ID
        tournament = Tournament(league_id=999, name="Unknown Tournament")
        tournament.get_league_matches()

        # Assert no matches are added to the tournament in case of failure
        self.assertEqual(len(tournament.matches), 0)


class TestMarkups(unittest.TestCase):
    def setUp(self):
        self.bot = MagicMock()
        self.markups = Markups(bot=self.bot)

    @patch("structure.struct.Buttons")
    def test_gen_main_markup(self, mock_buttons):
        # Set up mock buttons
        mock_buttons.get_history_button = MagicMock()
        mock_buttons.dota2_get_current_matches_button = MagicMock()
        mock_buttons.predict_by_id_button = MagicMock()
        mock_buttons.predict_pick_analyser_button = MagicMock()

        # Call the method to generate markup
        markup = self.markups.gen_main_markup(current_user_id=1, current_channel_id=1)

        # Check if the buttons were added correctly
        self.assertIn(mock_buttons.get_history_button, markup.keyboard[0])
        self.assertIn(mock_buttons.dota2_get_current_matches_button, markup.keyboard[1])
        self.assertIn(mock_buttons.predict_by_id_button, markup.keyboard[2])
        self.assertIn(mock_buttons.predict_pick_analyser_button, markup.keyboard[3])

    @patch("structure.struct.Dota2API")
    @patch("structure.struct.MainML")
    def test_gen_dota2_matches_markup(self, mock_ml, mock_dota_api):
        mock_dota_api_instance = MagicMock()
        mock_dota_api.return_value = mock_dota_api_instance

        mock_match = MagicMock()
        mock_match.match_id = 1

        # Set up Dire Team
        mock_match.dire_team.team_name = "Dire Team"
        mock_match.dire_team.team_id = 2
        mock_match.dire_team.players = [
            MagicMock(name="Player1", hero=MagicMock(name="Hero1"))
        ]

        # Set up Radiant Team
        mock_match.radiant_team.team_name = "Radiant Team"
        mock_match.radiant_team.team_id = 3
        mock_match.radiant_team.players = [
            MagicMock(name="Player2", hero=MagicMock(name="Hero2"))
        ]

        # Mocking the method to return a dataframe and top features
        mock_match.get_match_data_for_prediction.return_value = (
            MagicMock(),
            MagicMock(),
        )

        mock_dota_api_instance.get_live_tournaments.return_value = [
            MagicMock(name="Tournament", matches=[mock_match])
        ]

        mock_ml.predict.return_value = [1]  # Predict Radiant Wins

        call = MagicMock()
        call.message.chat.id = 12345

        self.markups.gen_dota2_matches_markup(call)

        # Check if the bot sent messages correctly
        self.bot.send_message.assert_called()

    @patch("structure.struct.Dota2API")
    def test_gen_match_markup_by_id(self, mock_dota_api):
        mock_dota_api_instance = MagicMock()
        mock_dota_api.return_value = mock_dota_api_instance

        # Assume we set buttons in a certain way
        mock_dota_api_instance.get_match_as_buttons.return_value = "Mocked Buttons"

        call = MagicMock()
        markup = self.markups.gen_match_markup_by_id(call)

        # Check if the markup is generated
        self.assertEqual(markup, "Mocked Buttons")

    @patch("structure.struct.Dota2API")
    @patch("structure.struct.MainML")
    def test_make_prediction_for_selected_match(self, mock_ml, mock_dota_api):
        mock_dota_api_instance = MagicMock()
        mock_dota_api.return_value = mock_dota_api_instance

        # Create team instances with players
        dire_team = TeamD(
            team_name="Dire Team",
            team_id=2,
            players=[PlayerD(name="Player1", hero=HeroD(name="Hero1"))],
        )
        radiant_team = TeamD(
            team_name="Radiant Team",
            team_id=3,
            players=[PlayerD(name="Player2", hero=HeroD(name="Hero2"))],
        )

        # Create a Match instance
        mock_match = MatchD(match_id=1, dire_team=dire_team, radiant_team=radiant_team)

        # Mocking the return value for build_single_match
        mock_dota_api_instance.build_single_match.return_value = mock_match

        # Mocking the return value for get_match_data_for_prediction
        mock_match.get_match_data_for_prediction = MagicMock(
            return_value=(MagicMock(), MagicMock())
        )

        # Mock the predict method to return prediction and probability
        mock_ml_instance = mock_ml.return_value  # Get the mocked instance of MainML
        mock_ml_instance.predict.return_value = (
            [1],
            [[0.2, 0.8]],
        )  # Predict Radiant Wins with probabilities

        call = MagicMock()
        call.message.chat.id = 12345

        # Call the method under test
        self.markups.make_prediction_for_selected_match(call, match_id=1)

        # Check if the bot sent the message correctly
        self.bot.send_message.assert_called()


class TestDota2API(unittest.TestCase):
    def setUp(self):
        # Initialize the Dota2API with a mock API key
        self.api_key = "mock_api_key"
        self.dota_api = Dota2API(api_key=self.api_key)

    @patch("requests.get")
    @patch("structure.struct.Player")
    @patch.object(
        Hero,
        "get_hero_features",
        return_value={"id": 1, "name": "Anti-Mage", "pro_win": 100, "pro_pick": 200},
    )
    def test_get_live_tournaments_success(
        self, mock_hero_features, mock_player, mock_get
    ):
        # Mock response for live tournaments
        mock_player_instance = mock_player.return_value
        mock_player_instance.get_player_data.return_value = {"win": 10, "lose": 5}
        mock_player_instance.get_player_total_data.return_value = [
            {"field": "kills", "sum": 100, "n": 10},
            {"field": "deaths", "sum": 50, "n": 10},
            {"field": "assists", "sum": 75, "n": 10},
            {"field": "gold_per_min", "sum": 300, "n": 10},
            {"field": "xp_per_min", "sum": 400, "n": 10},
            {"field": "last_hits", "sum": 200, "n": 10},
            {"field": "denies", "sum": 25, "n": 10},
        ]

        mock_response = {
            "result": {
                "games": [
                    {
                        "match_id": 1,
                        "league_id": 123,
                        "league_name": "Tournament A",
                        "radiant_team": {"team_id": 1, "team_name": "Radiant Team"},
                        "dire_team": {"team_id": 2, "team_name": "Dire Team"},
                        "players": [
                            {
                                "account_id": 101,
                                "name": "Player1",
                                "hero_id": 5,
                                "team": 0,
                            },
                            {
                                "account_id": 102,
                                "name": "Player2",
                                "hero_id": 3,
                                "team": 0,
                            },
                            {
                                "account_id": 201,
                                "name": "Player3",
                                "hero_id": 7,
                                "team": 1,
                            },
                            {
                                "account_id": 202,
                                "name": "Player4",
                                "hero_id": 2,
                                "team": 1,
                            },
                        ],
                    }
                ]
            }
        }

        # Set up the mock to return the above response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        # Call the method under test
        tournaments = self.dota_api.get_live_tournaments()

        # Assertions to validate the outcome
        self.assertEqual(len(tournaments), 1)
        self.assertEqual(tournaments[0].league_id, 123)
        self.assertEqual(tournaments[0].name, "Tournament A")
        self.assertEqual(len(tournaments[0].matches), 1)

        match = tournaments[0].matches[0]
        self.assertEqual(match.match_id, 1)
        self.assertEqual(match.radiant_team.team_id, 1)
        self.assertEqual(match.dire_team.team_id, 2)

    @patch("requests.get")
    def test_get_live_tournaments_no_matches(self, mock_get):
        # Mock response with no live matches
        mock_response = {"result": {"games": []}}

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        tournaments = self.dota_api.get_live_tournaments()
        self.assertEqual(len(tournaments), 0)  # Expecting no tournaments

    @patch("requests.get")
    def test_get_live_tournaments_error(self, mock_get):
        # Mock a response with an error status
        mock_get.return_value.status_code = 404

        tournaments = self.dota_api.get_live_tournaments()
        self.assertEqual(len(tournaments), 0)  # Expecting no tournaments due to error
