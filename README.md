<!-- GitHub badges -->
[![Tests](https://github.com/masterhood13/dota2predictor/actions/workflows/python-unitests.yml/badge.svg?branch=main)](https://github.com/masterhood13/dota2predictor/actions)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/masterhood13/dota2predictor.svg)](https://hub.docker.com/r/masterhood13/dota2predictor)
[![codecov](https://codecov.io/gh/masterhood13/dota2predictor/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/masterhood13/dota2predictor)
---

# Dota 2 Match Result Predictor Telegram Bot

## Overview

This project is a Telegram bot that leverages a TensorFlow neural network model to predict the outcomes of Dota 2 matches. The bot provides users with real-time predictions based on current match data, making it a useful tool for Dota 2 enthusiasts and analysts.

## Features

- **Real-Time Predictions**: Predict match outcomes using a trained neural network model.
- **Data Integration**:
  - Training data sourced from the **OpenDota API** for historical match results.
  - Current match data fetched using the **Steam API**.
- **User-Friendly Interface**: Interactive Telegram bot interface for easy access to predictions and match information.

## Technologies Used

- **Programming Language**: Python
- **Machine Learning Framework**: TensorFlow
- **APIs**: OpenDota API, Steam API
- **Bot Framework**: python-telegram-bot

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/masterhood13/dota2predictor.git
   cd dota2predictor
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Here’s how your .env file might look:
   ```bash
   OPENDOTA_KEY=your_actual_opendota_api_key
   STEAM_API_KEY=your_actual_steam_api_key
   TELEGRAM_KEY=your_actual_telegram_bot_token
   ```

4. Run the bot:
   ```bash
   python start.py
   ```

## Usage

Once the bot is running, users can interact with it through Telegram to receive predictions for ongoing Dota 2 matches. The bot will provide match details and predicted outcomes based on the trained model.

## High-Level System Schema

Below is a high-level system schema that illustrates how the components interact in the Dota 2 predictor bot:

```mermaid
flowchart TD
    %% User interaction with Telegram bot
    User -->|Sends match query| TelegramBot

    %% TelegramBot passes the request to Backend
    TelegramBot -->|Receives query| Backend

    %% Backend fetches data from external APIs
    Backend -->|Fetch match/player data| OpenDotaAPI
    Backend -->|Fetch match/player data| SteamAPI

    %% Data preprocessing
    Backend -->|Preprocesses data| PreprocessingEngine
    PreprocessingEngine -->|Feeds preprocessed data| XGBoostModel

    %% Model prediction and returning result to user
    XGBoostModel -->|Returns prediction| Backend
    Backend -->|Sends result| TelegramBot
    TelegramBot -->|Displays result| User

    %% Data flow from OpenDota and Steam APIs to Preprocessing Engine
    OpenDotaAPI -->|Returns match/player data| Backend
    SteamAPI -->|Returns player data| Backend
```

## Entity-Relationship Diagram (ERD)

Here’s the ERD showing the key entities and their relationships within the system:

```mermaid
erDiagram
    MATCH {
        int match_id
        date match_date
        int duration
        string outcome
    }
    PLAYER {
        int player_id
        string player_name
        int rank
    }
    TEAM {
        int team_id
        string team_name
    }
    HERO {
        int hero_id
        string hero_name
        string hero_role
    }
    PREDICTION_MODEL {
        int model_id
        string model_name
        string algorithm
        float accuracy
    }

    TEAM ||--o{ PLAYER : has
    PLAYER ||--o{ MATCH : participates_in
    MATCH ||--o{ TEAM : involves
    PLAYER ||--o{ HERO : plays
    PREDICTION_MODEL ||--|{ MATCH : uses_data_from
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [My Medium Article Part 1](https://medium.com/@masterhood13/building-a-dota-2-match-outcome-predictor-my-journey-and-learnings-fd60e1a79a23)
- [OpenDota API](https://docs.opendota.com/)
- [Steam API](https://steamapi.xpaw.me/)
- [PyTorch](https://pytorch.org/)
- [pyTelegramBotAPI](https://pypi.org/project/pyTelegramBotAPI/)

---
