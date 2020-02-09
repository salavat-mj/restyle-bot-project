#!/usr/bin/env python
import os
import logging
from tempfile import mkstemp
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InlineQueryResultVoice, InlineQueryResultAudio
from telegram.ext.dispatcher import run_async
from io import BytesIO
from model import StyleTransferModel
from config import token

#model = StyleTransferModel()
#first_image_file = {}


@run_async
def onstart(bot, update):
    logger.info('start command from %s' % update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id, text='Привет, я рестайл-бот. Я перересую ваши изображения.')
    onhelp(bot, update)


@run_async
def onhelp(bot, update):
    logger.info('help command from %s' % update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id, text='''
Чтобы перерисовать изображение, необходимо выполнить 2 действия:
1. Отправить изображение, которое хотите перерисовать.
2. Отправить изображиние-стиль или выбрать стиль из предложенных авторов.

Команды:
/help - помощь и описание
/photo2painting - перересует ваше следующее изображение в картину
'''.strip())

@run_async
def onunknown(bot, update):
    logger.info('unknown command from %s' %
                update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id,
                     text='Простите, я вас не понял. Это должно помочь /help.')

@run_async
def take_photo(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    #bot.send_photo(chat_id, image_info)
    bot.send_message(chat_id, 'Изображние получено')
    if chat_id in first_image_file:
        content = first_image_file[chat_id]
        del first_image_file[chat_id]

        content_image_stream = BytesIO()
        content.download(out=content_image_stream)
        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)

        bot.send_message(chat_id, 'Подождите, это займёт некоторое время')
        output = model.transfer_style(content_image_stream, style_image_stream)

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
    else:
        first_image_file[chat_id] = image_file
        bot.send_message(chat_id, 'Выбирите действие')


def onerror(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))

def main():
    updater = Updater(token=token)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', onstart))
    dp.add_handler(CommandHandler('help', onhelp))
    dp.add_handler(MessageHandler([Filters.command], onunknown))
    dp.add_handler(MessageHandler([Filters.photo], take_photo))
    dp.add_error_handler(onerror)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
