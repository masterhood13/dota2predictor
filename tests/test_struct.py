import unittest
from unittest.mock import patch, MagicMock

from structure.struct import Hero, Player, Team, Match, Tournament, Dota2API


class TestHero(unittest.TestCase):
    @patch('requests.get')
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

    @patch('requests.get')
    def test_get_hero_features_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        hero = Hero(hero_id=999)  # Assuming this ID does not exist
        self.assertEqual(hero.name, "Unknown Hero")
        self.assertEqual(hero.winrate, 0)


class TestPlayer(unittest.TestCase):
    @patch('requests.get')
    @patch.object(Hero, 'get_hero_features')
    def test_player_initialization(self, mock_hero_features, mock_get):
        # Mock the return value of get_hero_features
        mock_hero_features.return_value = {
            "hero_id": 1,
            "name": "Anti-Mage",
            "pro_win": 100,
            "pro_pick": 200,
        }

        # Mock responses for player data API calls
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"win": 10, "lose": 5}),  # First call for get_player_data
            MagicMock(status_code=200, json=lambda: [                     # Second call for heroes data
                {"hero_id": 1, "win": 5, "games": 10}
            ]),
            MagicMock(status_code=200, json=lambda: [                     # Third call for totals data
                {"field": "kills", "sum": 0, "n": 1},
                {"field": "deaths", "sum": 0, "n": 1},
                {"field": "assists", "sum": 0, "n": 1},
                {"field": "gold_per_min", "sum": 0, "n": 1},
                {"field": "xp_per_min", "sum": 0, "n": 1},
                {"field": "last_hits", "sum": 0, "n": 1},
                {"field": "denies", "sum": 0, "n": 1},
            ])
        ]

        # Create a Player instance
        player = Player(account_id=1, name="Player1", hero_id=1, team="Team A")

        # Assertions to validate player initialization
        self.assertEqual(player.name, "Player1")
        self.assertEqual(player.hero.name, "Anti-Mage")
        self.assertAlmostEqual(player.kills, 0)  # Ensure kills are calculated correctly
        self.assertAlmostEqual(player.deaths, 0)  # Check if deaths is also zero
        self.assertAlmostEqual(player.assists, 0)  # Check if assists is also zero

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


class TestMatch(unittest.TestCase):
    @patch('requests.get')
    @patch('structure.struct.Player')
    @patch.object(Hero, 'get_hero_features', return_value={
        "id": 1,
        "name": "Anti-Mage",
        "pro_win": 100,
        "pro_pick": 200,
    })
    def test_get_match_data_success(self, mock_hero_features, mock_player, mock_get):
        # Mock the Player instance and its methods
        mock_player_instance = mock_player.return_value
        mock_player_instance.get_player_data.return_value = {
            "win": 10,
            "lose": 5
        }
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
            ]
        }

        # Create a Match instance and call get_match_data
        match = Match(match_id=1, radiant_team_id=1, dire_team_id=2, league_id=1)
        match.get_match_data()

        # Assertions to validate the match data
        self.assertEqual(match.radiant_team.team_name, "Team A")
        self.assertEqual(match.dire_team.team_name, "Team B")
        self.assertTrue(match.radiant_win)

        # Assuming players are stored in match.radiant_team.players or similar
        for player in match.radiant_team.players:
            if player.account_id == 1:  # Checking for Player1
                self.assertAlmostEqual(player.kills, 10)  # 100/10
                self.assertAlmostEqual(player.deaths, 5)  # 50/10
                self.assertAlmostEqual(player.assists, 7.5)  # 75/10
                self.assertAlmostEqual(player.gold_per_min, 30)  # 300/10
                self.assertAlmostEqual(player.xp_per_min, 40)  # 400/10
                self.assertAlmostEqual(player.last_hits, 20)  # 200/10
                self.assertAlmostEqual(player.denies, 2.5)  # 25/10


class TestTournament(unittest.TestCase):
    @patch('requests.get')
    @patch('structure.struct.Match')  # Mock the Match class
    def test_get_league_matches_success(self, mock_match, mock_get):
        # Mock the Match instance
        mock_match_instance = MagicMock()
        mock_match.return_value = mock_match_instance

        # Set up the mock response for league matches
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {"match_id": 1, "radiant_team_id": 1, "dire_team_id": 2, "radiant_win": True}
        ]

        # Set up the mock for get_match_data to not do the actual API call
        mock_match_instance.get_match_data.return_value = None  # No return needed for this test

        # Set the return values for the Match instance attributes
        mock_match_instance.match_id = 1
        mock_match_instance.radiant_team_id = 1
        mock_match_instance.dire_team_id = 2
        mock_match_instance.radiant_win = True

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

    @patch('requests.get')
    def test_get_league_matches_failure(self, mock_get):
        mock_get.return_value.status_code = 404
        tournament = Tournament(league_id=999, name="Unknown Tournament")
        tournament.get_league_matches()
        self.assertEqual(len(tournament.matches), 0)


class TestDota2API(unittest.TestCase):
    def setUp(self):
        # Initialize the Dota2API with a mock API key
        self.api_key = "mock_api_key"
        self.dota_api = Dota2API(api_key=self.api_key)


    @patch('requests.get')
    @patch('structure.struct.Player')
    @patch.object(Hero, 'get_hero_features', return_value={"id": 1,
        "name": "Anti-Mage",
        "pro_win": 100,
        "pro_pick": 200})
    def test_get_live_tournaments_success(self, mock_hero_features, mock_player, mock_get):
        # Mock response for live tournaments
        mock_player_instance = mock_player.return_value
        mock_player_instance.get_player_data.return_value = {
            "win": 10,
            "lose": 5
        }
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
                            {"account_id": 101, "name": "Player1", "hero_id": 5, "team": 0},
                            {"account_id": 102, "name": "Player2", "hero_id": 3, "team": 0},
                            {"account_id": 201, "name": "Player3", "hero_id": 7, "team": 1},
                            {"account_id": 202, "name": "Player4", "hero_id": 2, "team": 1},
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

    @patch('requests.get')
    def test_get_live_tournaments_no_matches(self, mock_get):
        # Mock response with no live matches
        mock_response = {
            "result": {
                "games": []
            }
        }

        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        tournaments = self.dota_api.get_live_tournaments()
        self.assertEqual(len(tournaments), 0)  # Expecting no tournaments

    @patch('requests.get')
    def test_get_live_tournaments_error(self, mock_get):
        # Mock a response with an error status
        mock_get.return_value.status_code = 404

        tournaments = self.dota_api.get_live_tournaments()
        self.assertEqual(len(tournaments), 0)  # Expecting no tournaments due to error


if __name__ == '__main__':
    unittest.main()
