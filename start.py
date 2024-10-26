# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.
import ast

from telebot import TeleBot
from config import telegram_key
from structure.struct import Markups, CallbackTriggers


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
        print(match_id)
        Markups(bot).make_prediction_for_selected_match(call, match_id)

    @staticmethod
    def predict_on_selected_hero_match(call):
        match_id = ast.literal_eval(call.data)[1]
        print(match_id)
        Markups(bot).make_hero_pick_prediction_for_selected_match(call, match_id)


@bot.callback_query_handler(func=lambda query: True)
def callback_query(call):
    if call.data == CallbackTriggers.dota2_get_current_matches_trigger:
        CallbackProcessor.current_matches(call)
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
