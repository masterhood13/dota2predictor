import unittest
from unittest.mock import patch, MagicMock

from structure.struct import Hero, Player, Team, Match, Tournament, Dota2API


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


class TestPlayer(unittest.TestCase):
    @patch("requests.get")
    @patch.object(Hero, "get_hero_features")
    def test_player_initialization(self, mock_hero_features, mock_get):
        # Mock the return value of get_hero_features
        mock_hero_features.return_value = {
            "hero_id": 1,
            "name": "Anti-Mage",
            "pro_win": 100,
            "pro_pick": 200,
        }

        # Create a Player instance with mock data
        player_data = {
            "name": "Player1",
            "teamfight_participation": 0.75,
            "obs_placed": 5,
            "sen_placed": 3,
            "net_worth": 15000,
            "kills": 10,
            "deaths": 4,
            "assists": 8,
            "roshans_killed": 1,
            "last_hits": 50,
            "denies": 10,
            "gold_per_min": 400,
            "xp_per_min": 450,
            "level": 25,
            "hero_damage": 20000,
            "tower_damage": 5000,
            "hero_healing": 3000,
        }

        player = Player(
            account_id=1,
            name="Player1",
            hero_id=1,
            team="Team A",
            player_data=player_data,
        )

        # Assertions to validate player initialization
        self.assertEqual(player.name, "Player1")
        self.assertEqual(player.hero.name, "Anti-Mage")
        self.assertEqual(player.kills, 10)
        self.assertEqual(player.deaths, 4)
        self.assertEqual(player.assists, 8)
        self.assertEqual(player.net_worth, 15000)
        self.assertEqual(player.teamfight_participation, 0.75)
        self.assertEqual(player.obs_placed, 5)
        self.assertEqual(player.sen_placed, 3)
        self.assertEqual(player.roshans_killed, 1)
        self.assertEqual(player.last_hits, 50)
        self.assertEqual(player.denies, 10)
        self.assertEqual(player.gold_per_min, 400)
        self.assertEqual(player.xp_per_min, 450)
        self.assertEqual(player.level, 25)
        self.assertEqual(player.hero_damage, 20000)
        self.assertEqual(player.tower_damage, 5000)
        self.assertEqual(player.hero_healing, 3000)

        # Ensure the Player's string representation is correct
        expected_repr = (
            "Player(Name: Player1, Hero: Anti-Mage, Team: Team A, "
            "Teamfight Participation: 75.0%, Observers Placed: 5, Sentries Placed: 3, "
            "K/D/A: 10/4/8, Net Worth: 15,000, Roshan Killed: 1, Last Hits: 50, "
            "Denies: 10, GPM: 400, XPM: 450, Level: 25, Hero Damage: 20,000, "
            "Tower Damage: 5,000, Hero Healing: 3,000)"
        )
        self.assertEqual(repr(player), expected_repr)

    @patch("requests.get")
    @patch.object(Hero, "get_hero_features")
    def test_get_player_total_data(self, mock_hero_features, mock_get):
        # Mock the return value of get_hero_features with the required structure
        mock_hero_features.return_value = {
            "id": 1,  # Make sure to include the 'id' key
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

        # Test for the Dire team players
        for player in match.dire_team.players:
            if player.account_id == 2:  # Checking for Player2
                # You can add relevant assertions for Player2 here as needed
                self.assertIsNotNone(
                    player.hero
                )  # Just to check if Hero object is created
                self.assertEqual(player.name, "Player2")


class TestTournament(unittest.TestCase):
    @patch("requests.get")
    @patch("structure.struct.Match")  # Mock the Match class
    def test_get_league_matches_success(self, mock_match, mock_get):
        # Mock the Match instance
        mock_match_instance = MagicMock()
        mock_match.return_value = mock_match_instance

        # Set up the mock response for league matches
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [
            {
                "match_id": 1,
                "radiant_team_id": 1,
                "dire_team_id": 2,
                "radiant_win": True,
            }
        ]

        # Set up the mock for get_match_data to not do the actual API call
        mock_match_instance.get_match_data.return_value = (
            None  # No return needed for this test
        )

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

    @patch("requests.get")
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


if __name__ == "__main__":
    unittest.main()
