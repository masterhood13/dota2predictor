# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import pandas as pd
from structure.struct import Tournament
from structure.opendota import OpenDotaApi


def generate_dataset():
    api = OpenDotaApi()
    dataset = []

    premium_leagues = api.set_premium_leagues()

    last_big_leagues = [
        "ESL One Kuala Lumpur powered by Intel",
        "BetBoom Dacha Dubai 2024",
        "DreamLeague Season 22 powered by Intel",
        "Elite League Season 2 Main Event – presented by ESB",
        "ESL One Birmingham 2024 Powered by Intel",
        "DreamLeague Season 23 powered by Intel",
        "Riyadh Masters 2024 at Esports World Cup",
        "Clavision DOTA League S1 : Snow-Ruyi",
        "The International 2024",
        "PGL Wallachia 2024 Season 1",
        "The International 2023",
        "Riyadh Masters 2023 by Gamers8",
        "BetBoom Dacha",
        "DreamLeague Season 21 powered by Intel",
        "The Bali Major",
        "DreamLeague Season 20 powered by Intel",
        "ESL One The Berlin Major powered by Intel",
        "DreamLeague Season 19 powered by Intel",
        "Lima Major 2023",
        "The International 2022",
        "ESL One Malaysia 2022 powered by Intel",
        "PGL Arlington Major 2022",
        "Riyadh Masters by Gamers8",
        "ESL One Stockholm Major 2022 powered by Intel",
        "Gamers Galaxy: Dota 2 Invitational Series Dubai 2022",
        "The International 2021",
        "SAPPHIRE OGA DOTA PIT INVITATIONAL",
        "ESL One Fall 2021 powered by Intel",
        "ESL One Summer 2021 powered by Intel",
    ]

    for premium_league in premium_leagues:
        league_id = premium_league["leagueid"]
        league_name = premium_league["name"]

        if league_name in last_big_leagues:
            tournament = Tournament(league_id=league_id, name=league_name)
            tournament.get_league_matches()  # Load matches for the tournament

            # Extract data from each match in the tournament
            for match in tournament.matches:
                radiant_team = match.radiant_team
                dire_team = match.dire_team

                # Ensure we have 5 players on each team
                if len(radiant_team.players) == 5 and len(dire_team.players) == 5:
                    # Create a single row with match and player data
                    match_data = {
                        "match_id": match.match_id,
                        "radiant_team_id": radiant_team.team_id,
                        "radiant_team_name": radiant_team.team_name,
                        "dire_team_id": dire_team.team_id,
                        "dire_team_name": dire_team.team_name,
                        "radiant_win": match.radiant_win,  # True/False if Radiant team won
                    }

                    # Add radiant team player data (5 players)
                    for i, player in enumerate(radiant_team.players):
                        match_data[f"radiant_player_{i + 1}_hero_id"] = (
                            player.hero.hero_id
                        )
                        match_data[f"radiant_player_{i + 1}_hero_name"] = (
                            player.hero.name
                        )
                        match_data[f"radiant_player_{i + 1}_hero_winrate"] = (
                            player.hero.winrate
                        )
                        for n, counter_pick in enumerate(player.hero.counter_picks):
                            match_data[f"radiant_hero_{i + 1}_{n + 1}_counter_pick"] = (
                                counter_pick["win_rate"]
                            )

                    # Add dire team player data (5 players)
                    for i, player in enumerate(dire_team.players):
                        match_data[f"dire_player_{i + 1}_hero_id"] = player.hero.hero_id
                        match_data[f"dire_player_{i + 1}_hero_name"] = player.hero.name
                        match_data[f"dire_player_{i + 1}_hero_winrate"] = (
                            player.hero.winrate
                        )
                        for n, counter_pick in enumerate(player.hero.counter_picks):
                            match_data[f"dire_hero_{i + 1}_{n + 1}_counter_pick"] = (
                                counter_pick["win_rate"]
                            )

                    print(match_data)
                    # Append match data to dataset
                    dataset.append(match_data)

            df = pd.DataFrame(dataset)
            # Write DataFrame to a CSV file
            df.to_csv(
                f"premium_league_matches_{league_name.replace(' ', '_')}.csv",
                index=False,
            )
    print("Match dataset has been generated and saved to 'premium_league_matches.csv'.")


generate_dataset()
