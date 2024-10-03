# © 2024 Viktor Hamretskyi <masterhood13@gmail.com>
# All rights reserved.
# This code is licensed under the MIT License. See LICENSE file for details.


from telebot import TeleBot
from config import telegram_key
from structure.struct import Markups, CallbackTriggers

bot = TeleBot(telegram_key)


class CallbackProcessor:
    @staticmethod
    def current_matches(call):
        message = Markups().gen_dota2_matches_markup()
        bot.send_message(chat_id=call.message.chat.id, text=message, parse_mode="HTML")


@bot.callback_query_handler(func=lambda query: True)
def callback_query(call):
    if call.data == CallbackTriggers.dota2_get_current_matches_trigger:
        CallbackProcessor.current_matches(call)


@bot.message_handler(func=lambda message: True)
def message_handler(message):
    bot.send_message(
        message.chat.id,
        "Main screen",
        reply_markup=Markups().gen_main_markup(message.from_user.id, message.chat.id),
    )


bot.infinity_polling()
