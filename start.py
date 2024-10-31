# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.

import ast
import logging
from io import BytesIO
from telebot import TeleBot
from config import telegram_key
from db.database_operations import get_history_data_as_dataframe
from structure.struct import Markups, CallbackTriggers

logger = logging.getLogger(__name__)
bot = TeleBot(telegram_key)


class CallbackProcessor:
    @staticmethod
    def current_matches(call):
        Markups(bot).gen_dota2_matches_markup(call)

    @staticmethod
    def select_match_list(call):
        bot.edit_message_text(
            "Available matches for prediction",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=Markups(bot).gen_match_markup_by_id(call),
        )

    @staticmethod
    def select_hero_match_list(call):
        bot.edit_message_text(
            "Available matches for prediction",
            call.message.chat.id,
            call.message.message_id,
            reply_markup=Markups(bot).gen_hero_match_markup_by_id(call),
        )

    @staticmethod
    def predict_on_selected_match(call):
        match_id = ast.literal_eval(call.data)[1]
        Markups(bot).make_prediction_for_selected_match(call, match_id)

    @staticmethod
    def predict_on_selected_hero_match(call):
        match_id = ast.literal_eval(call.data)[1]
        Markups(bot).make_hero_pick_prediction_for_selected_match(call, match_id)

    @staticmethod
    def send_history_csv(call):
        try:
            logger.info("Starting to send history CSV...")

            # Create a BytesIO buffer
            csv_buffer = BytesIO()

            # Fetch the DataFrame
            df = get_history_data_as_dataframe()

            # Check if DataFrame is empty
            if df.empty:
                logger.warning("DataFrame is empty!")
                bot.send_message(
                    chat_id=call.message.chat.id, text="No data available to send."
                )
                return

            logger.info("DataFrame retrieved successfully. Writing to CSV...")

            # Write DataFrame to CSV in the buffer
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            csv_buffer.seek(0)  # Go back to the beginning of the buffer

            # Send the document via the bot
            logger.info("Sending CSV document to chat_id=%s", call.message.chat.id)
            bot.send_document(
                chat_id=call.message.chat.id,
                document=csv_buffer,
                visible_file_name="history_data.csv",  # Correct argument name
                caption="Here is the history data you requested.",
            )

            logger.info("CSV sent successfully to chat_id=%s", call.message.chat.id)

        except Exception as e:
            logger.error("Error occurred while sending CSV: %s", e)
            bot.send_message(
                chat_id=call.message.chat.id,
                text="An error occurred while sending the data.",
            )


@bot.callback_query_handler(func=lambda query: True)
def callback_query(call):
    if call.data == CallbackTriggers.dota2_get_current_matches_trigger:
        CallbackProcessor.current_matches(call)
    elif call.data == CallbackTriggers.get_history_of_predictions_trigger:
        CallbackProcessor.send_history_csv(call)
    elif call.data == CallbackTriggers.predict_by_id_trigger:
        CallbackProcessor.select_match_list(call)
    elif call.data == CallbackTriggers.predict_pick_analyser_trigger:
        CallbackProcessor.select_hero_match_list(call)
    elif call.data.startswith(CallbackTriggers.match_trigger):
        CallbackProcessor.predict_on_selected_match(call)
    elif call.data.startswith(CallbackTriggers.hero_match_trigger):
        CallbackProcessor.predict_on_selected_hero_match(call)


@bot.message_handler(func=lambda message: True)
def message_handler(message):
    bot.send_message(
        message.chat.id,
        "Main screen",
        reply_markup=Markups(bot).gen_main_markup(
            message.from_user.id, message.chat.id
        ),
    )


bot.infinity_polling()
