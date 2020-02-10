#!/usr/bin/env python
import os, shutil
import logging
from tempfile import mkstemp
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InlineQueryResultVoice, InlineQueryResultAudio
from telegram.ext.dispatcher import run_async
from io import BytesIO
from model import StyleTransferModel, use_gan
from config import token

model = StyleTransferModel()
first_image_file = {}

import torchvision.transforms as transforms

# Часть бота, отвечающая за рестайлинг
@run_async
def take_photo(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    #bot.send_photo(chat_id, image_info)
    if chat_id in first_image_file:
        #bot.send_message(chat_id, 'Изображние получено')
        content = first_image_file.pop(chat_id)
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
        bot.send_message(chat_id, '''
Отправьте изображиние-стиль или выбирите готовый стиль:
/photo2cezanne - Поль Сезанн
/photo2monet - Клод Моне
/photo2ukiyoe - Укиё-э
/photo2vangogh - Винсент ван Гог
''')

@run_async
def gan_restyling(bot, update):
    chat_id = update.message.chat_id
    if chat_id in first_image_file:
        content = first_image_file.pop(chat_id)
        style = update.message.text[7:]  # отрезаем '/photo2' у команды
        date = update.message.date.isoformat()
        im_dir = './temp/%s_%s/testB/' % (chat_id, date)
        os.makedirs(im_dir)
        content.download(custom_path=im_dir+'real.png')
        bot.send_message(chat_id, 'Идёт обработка изображения в стиле %s' % style)
        use_gan(im_dir, style)
        bot.send_photo(chat_id, photo=open(im_dir+'fake.png', "rb"))
        shutil.rmtree(im_dir[:-6], ignore_errors=True)
        #bot.send_message(chat_id, 'Изображение сохранено')
    else:
        bot.send_message(chat_id, 'Сперва необходимо отправить изображение-контент')

# Стандартная часть
@run_async
def onstart(bot, update):
    logger.info('start command from %s' % update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id, text='Привет, я рестайл-бот. Я перересую ваши изображения.')
    onhelp(bot, update)

@run_async
def onhelp(bot, update):
    logger.info('help command from %s' % update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id, text='''
Для рестайлинга необходимо 2 изображения: контент и стиль.

Вам понадобится выполнить 2 действия:
1. Отправить изображение-контент, которое хотите перерисовать.
2. Отправить изображиние-стиль или выбрать готовый стиль:
/photo2cezanne - Поль Сезанн
/photo2monet - Клод Моне
/photo2ukiyoe - Укиё-э
/photo2vangogh - Винсент ван Гог
'''.strip())

@run_async
def onunknown(bot, update):
    logger.info('unknown command from %s' %
                update.message.from_user.first_name)
    bot.send_message(chat_id=update.message.chat_id,
                     text='Простите, я вас не понял. Это должно помочь /help.')

def onerror(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))

def main():
    updater = Updater(token=token)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', onstart))
    dp.add_handler(CommandHandler('help', onhelp))
    dp.add_handler(CommandHandler(['photo2cezanne', 'photo2monet', 'photo2ukiyoe', 'photo2vangogh'], gan_restyling))
    dp.add_handler(MessageHandler([Filters.command], onunknown))
    dp.add_handler(MessageHandler([Filters.photo], take_photo))
    dp.add_error_handler(onerror)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
