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
        # "ESL One Kuala Lumpur powered by Intel",
        "BetBoom Dacha Dubai 2024",
        # "DreamLeague Season 22 powered by Intel",
        # "Elite League Season 2 Main Event – presented by ESB",
        # "ESL One Birmingham 2024 Powered by Intel",
        # "DreamLeague Season 23 powered by Intel",
        # "Riyadh Masters 2024 at Esports World Cup",
        # "Clavision DOTA League S1 : Snow-Ruyi",
        # "The International 2024",
        # "PGL Wallachia 2024 Season 1",
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
                        match_data[f"radiant_player_{i + 1}_id"] = player.account_id
                        match_data[f"radiant_player_{i + 1}_name"] = player.name
                        match_data[f"radiant_player_{i + 1}_hero_id"] = (
                            player.hero.hero_id
                        )
                        match_data[f"radiant_player_{i + 1}_hero_name"] = (
                            player.hero.name
                        )
                        match_data[f"radiant_player_{i + 1}_hero_winrate"] = (
                            player.hero.winrate
                        )
                        # match_data[f"radiant_player_{i + 1}_winrate"] = player.player_data["win_rate"]
                        match_data[f"radiant_player_{i + 1}_kills"] = player.kills
                        match_data[f"radiant_player_{i + 1}_deaths"] = player.deaths
                        match_data[f"radiant_player_{i + 1}_assists"] = player.assists
                        match_data[f"radiant_player_{i + 1}_gold_per_min"] = (
                            player.gold_per_min
                        )
                        match_data[f"radiant_player_{i + 1}_xp_per_min"] = (
                            player.xp_per_min
                        )

                        # New fields
                        match_data[
                            f"radiant_player_{i + 1}_teamfight_participation"
                        ] = player.teamfight_participation
                        match_data[f"radiant_player_{i + 1}_obs_placed"] = (
                            player.obs_placed
                        )
                        match_data[f"radiant_player_{i + 1}_sen_placed"] = (
                            player.sen_placed
                        )
                        match_data[f"radiant_player_{i + 1}_net_worth"] = (
                            player.net_worth
                        )
                        match_data[f"radiant_player_{i + 1}_roshans_killed"] = (
                            player.roshans_killed
                        )
                        match_data[f"radiant_player_{i + 1}_last_hits"] = (
                            player.last_hits
                        )
                        match_data[f"radiant_player_{i + 1}_denies"] = player.denies
                        match_data[f"radiant_player_{i + 1}_level"] = player.level
                        match_data[f"radiant_player_{i + 1}_hero_damage"] = (
                            player.hero_damage
                        )
                        match_data[f"radiant_player_{i + 1}_tower_damage"] = (
                            player.tower_damage
                        )

                    # Add dire team player data (5 players)
                    for i, player in enumerate(dire_team.players):
                        match_data[f"dire_player_{i + 1}_id"] = player.account_id
                        match_data[f"dire_player_{i + 1}_name"] = player.name
                        match_data[f"dire_player_{i + 1}_hero_id"] = player.hero.hero_id
                        match_data[f"dire_player_{i + 1}_hero_name"] = player.hero.name
                        match_data[f"dire_player_{i + 1}_hero_winrate"] = (
                            player.hero.winrate
                        )
                        # match_data[f"dire_player_{i + 1}_winrate"] = player.player_data["win_rate"]
                        match_data[f"dire_player_{i + 1}_kills"] = player.kills
                        match_data[f"dire_player_{i + 1}_deaths"] = player.deaths
                        match_data[f"dire_player_{i + 1}_assists"] = player.assists
                        match_data[f"dire_player_{i + 1}_gold_per_min"] = (
                            player.gold_per_min
                        )
                        match_data[f"dire_player_{i + 1}_xp_per_min"] = (
                            player.xp_per_min
                        )

                        # New fields
                        match_data[f"dire_player_{i + 1}_teamfight_participation"] = (
                            player.teamfight_participation
                        )
                        match_data[f"dire_player_{i + 1}_obs_placed"] = (
                            player.obs_placed
                        )
                        match_data[f"dire_player_{i + 1}_sen_placed"] = (
                            player.sen_placed
                        )
                        match_data[f"dire_player_{i + 1}_net_worth"] = player.net_worth
                        match_data[f"dire_player_{i + 1}_roshans_killed"] = (
                            player.roshans_killed
                        )
                        match_data[f"dire_player_{i + 1}_last_hits"] = player.last_hits
                        match_data[f"dire_player_{i + 1}_denies"] = player.denies
                        match_data[f"dire_player_{i + 1}_level"] = player.level
                        match_data[f"dire_player_{i + 1}_hero_damage"] = (
                            player.hero_damage
                        )
                        match_data[f"dire_player_{i + 1}_tower_damage"] = (
                            player.tower_damage
                        )

                    print(match_data)
                    # Append match data to dataset
                    dataset.append(match_data)

    df = pd.DataFrame(dataset)

    # Write DataFrame to a CSV file
    df.to_csv("premium_league_matches10.csv", index=False)
    print("Match dataset has been generated and saved to 'premium_league_matches.csv'.")


generate_dataset()
