from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy

import subprocess

def use_gan(im_dir, style):
    args = 'python ./CycleGAN/test.py --dataroot %s --name style_%s_pretrained --gpu_ids -1' % (im_dir, style)
    print(args)
    subprocess.call(args.split())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# Transfer

Теперь нужно создать функции, которые будут вычислять расстояния ( $D_C$ и $D_S$). <br />
Они будут выполенены в виде слоев, чтобы брать по ним автоградиент.

$D_S$ - средняя квадратичная ощибка input'а и target'а
"""

class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach() # это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target) # хороший код стаил (чтобы не падало при вызове лосса до форварда)

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

"""Матрица грама позволяет учесть не только сами значения feature map'а, но и кореляцию фич друг с другом. <br /> Это нужно для того, чтобы сделать акцент на встречаемость фич с друг другом, а не на их геометрическом положении. <br />
Полное понимание этого момента можно получить с помощью [этого](https://arxiv.org/pdf/1508.06576.pdf)  и [этого](https://m.habr.com/company/mailru/blog/306916/).

Таким образом:

$D_S$  = $\sum$($G_{ij}$($img_1$) - $G_{ij}$($img_2$)$)^{2}$

Сначала задаем спрособ подсчета матрицы грама: Это просто тензорное тензорное произведение вектора выхода уровня самого на себя.<br /> Однка наш выход - не вектор. В этом случае операция тоже возможна,<br /> но мы получим тензор третьего ранга. Поэтому перед перемножением выход нужно привести к форме вектора.<br />
"""

def gram_matrix(input):
        batch_size , h, w, f_map_num = input.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)

        features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product (matrix multiply)

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)

"""Матрица грама готова, теперь нужно лишь реализовать MSE"""

class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach() # мы не изменяем предобученую нейросеть
            self.loss = F.mse_loss(self.target, self.target) # to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

"""При тренировке VGG каждое изображение на котором она обучалась было нормировано по всем каналам (RGB). Если мы хотим изпользовать ее для нашей модели, то мы должны реализовать нормировку и для наших изображений тоже."""

# хардкод параметры из статьи (нормировка ргб)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            #self.mean = torch.tensor(mean).view(-1, 1, 1)
            #self.std = torch.tensor(std).view(-1, 1, 1)
            self.mean = mean.clone().view(-1, 1, 1)
            self.std = std.clone().view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

"""Теперь соберем это все в одну функцию, которая отдаст на выходе модель и две функции потерь

Определим предобученную модель
"""

cnn = models.vgg19(pretrained=True).features.to(device).eval()

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=['conv_4'],
                                   style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        cnn = copy.deepcopy(cnn)

        # normalization module
        'нейросеть предобучена обрабатывать нормализованные изображения'
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # just in order to have an iterable access to or list of content/syle losses
        'выходы тех уровней, которые считают наши функции потерь'
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                'Пересоздаём relu уровень, т.к. из vgg19 чет не работает'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                'вставляем в vgg наши лоссы'
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                'вставляем в vgg наши лоссы'
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        'выбрасываем все уровни после последенего styel loss или content loss'
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        'обрезаем (в нашем случае на conv_5)'
        model = model[:(i + 1)]

        return model, style_losses, content_losses

def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        # данный оптимизатор показал себя лучше всего из различных протестированных
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

"""Дальше стандартный цикл обучения, но что это за closure?<br /> Это функция, которая вызывается во время каждого прохода, чтобы пересчитать loss. Без нее ничего не получется так как у нас своя функция ошибки"""

# номера -- это эвристика
def conv_layers(*args):
    return ['conv_{}'.format(i) for i in args]

def run_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=500,
                        style_weight=100000, content_weight=1, layers=None):
        """Run the style transfer."""
        # Важные аргументы:
        # num_steps -- количество эпох (не оч важный)
        # style_weight -- альфа
        # content_weight -- бета (можно попробовать 1, 0, -1)
        # практика показывает, что картинки интереснее, когда мы забиваем на контент

        content_layers = conv_layers(*[8])
        style_layers = conv_layers(*range(1, 11))
        if layers:
            content_layers = conv_layers(*layers[0])
            style_layers = conv_layers(*layers[1])

        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img,
            content_layers=content_layers, style_layers=style_layers)
        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                # зануляем градиент
                optimizer.zero_grad()

                # пропускаем картинку через всё нейросеть
                model(input_img)

                # явно описываем наши лоссы
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ошибки (домножение на альфу и бета)
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score

                # обратное распространение ошибки
                # мы не апдейтим сеть
                # мы не апдейтим стиль
                # мы не апдейтим контент
                # мы апдейтим только пиксели исходной картинки (input)
                loss.backward()

                # вывод инфы каждые 100 итераций
                run[0] += 1
                if run[0] % 100 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    # Вывод промежуточных изображений
                    #img = transforms.ToPILImage()(input_img.cpu().clone().squeeze(0))
                    #img.save('{}.png'.format(params[0]), 'png')
                    #plt.imshow(img)
                    #plt.pause(0.001)
                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

# В данном классе мы хотим полностью производить всю обработку картинок, которые поступают к нам из телеграма.
# Это всего лишь заготовка, поэтому не стесняйтесь менять имена функций, добавлять аргументы, свои классы и
# все такое.
class StyleTransferModel:
    def __init__(self):
        # Сюда необходимо перенести всю иницализацию, вроде загрузки свеерточной сети и т.д.

        # Конструктор
        self.to_PIL_img = transforms.ToPILImage() # тензор в картинку
        imsize = 256
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат


    def transfer_style(self, content_img_stream, style_img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.
        # В первой итерации, когда вы переносите уже готовую модель из тетрадки с занятия сюда нужно просто
        # перенести функцию run_style_transfer (не забудьте вынести инициализацию, которая
        # проводится один раз в конструктор.

        # Сейчас этот метод просто возвращает не измененную content картинку
        # Для наглядности мы сначала переводим ее в тензор, а потом обратно
        style_img = self._process_image(style_img_stream)#.cpu()[0]
        content_img = self._process_image(content_img_stream)#.cpu()[0]
        input_img = content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        #input_img = torch.randn(content_img.data.size(), device=device)

        output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img,
                                    num_steps=1000, style_weight=1000000, content_weight=1)
        return self.to_PIL_img(output.cpu()[0])

    # В run_style_transfer используется много внешних функций, их можно добавить как функции класса
    def _process_image(self, img_stream):
        image = Image.open(img_stream)
        image = self.loader(image).unsqueeze(0)
        return image.to(device, torch.float)
