# restyle-bot-project

Телеграм бот для изменения стиля изображений.

## Содержание
- `main.py` - алгоритм бота
- `model.py` - transfer style модель
- `checkpoints` - папка для предобученных сетей из проекта https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- `CycleGAN` - урезанная часть проекта, достаточная для запуска кастомного `test.py`

## Для запуска
- `python main.py` - вручную
- [bot.ipynb](https://colab.research.google.com/drive/1vRvDDxT0t3ISgIXifpa2bDiKiwnwTqoU) - на мощностях Google Colab
