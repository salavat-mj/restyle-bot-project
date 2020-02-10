from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# матрица грамма
def gram_matrix(input):
    """Матрица грама позволяет учесть не только сами значения feature map'а, но и
    кореляцию фич друг с другом. Полное понимание этого момента можно получить с помощью
    https://arxiv.org/pdf/1508.06576.pdf и https://m.habr.com/company/mailru/blog/306916/.

    Сначала задаем спрособ подсчета матрицы грама:
    Это просто тензорное тензорное произведение вектора выхода уровня самого на себя.
    Однка наш выход - не вектор. Хоть операция и возможна, мы получим тензор третьего ранга.
    Поэтому перед перемножением выход нужно привести к форме вектора.
    """
    batch_size , h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)

    features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product (matrix multiply)

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)

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

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach() # мы не изменяем предобученую нейросеть
        self.loss = F.mse_loss(self.target, self.target) # to initialize with something

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleTransferModel:
    def __init__(self,
            imsize=256,
            num_steps=1000,
            style_weight=1000000,
            content_weight=1,
            content_layers=[4],
            style_layers=[1, 2, 3, 4, 5]):
        """При тренировке VGG каждое изображение на котором она обучалась было нормировано по всем каналам (RGB). Если мы хотим изпользовать ее для нашей модели, то мы должны реализовать нормировку и для наших изображений тоже."""
        # нейросеть предобучена обрабатывать нормализованные изображения
        self.device = device
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.normalization = Normalization(normalization_mean, normalization_std).to(self.device)
        # Определим предобученную модель
        self.cnn = copy.deepcopy(cnn)
        # Определим слои для лосса
        self.content_layers = ['conv_{}'.format(i) for i in content_layers]
        self.style_layers = ['conv_{}'.format(i) for i in style_layers]
        # Определим параметры стайл трансфера
        self.imsize = imsize
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight

    def get_style_model_and_losses(self, style_img, content_img):
        # выходы тех уровней, которые считают наши функции потерь
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(self.normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # Пересоздаём relu уровень, т.к. из vgg19 чет не работает
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                # вставляем в vgg наши лоссы
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                # вставляем в vgg наши лоссы
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # выбрасываем все уровни после последенего styel loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        # обрезаем (в нашем случае на conv_5)
        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        """ Добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров.
        Данный оптимизатор показал себя лучше всего из различных протестированных
        """
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, content_img, style_img, input_img):
        """Дальше стандартный цикл обучения, но что это за closure?<br /> Это функция, которая вызывается во время каждого прохода, чтобы пересчитать loss. Без нее ничего не получется так как у нас своя функция ошибки"""
        """Run the style transfer."""

        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)

        optimizer = self.get_input_optimizer(input_img)
        print('Optimizing..')

        step = 100
        scores = [0, 0]

        for steps in range(0, self.num_steps, step):
            for i in range(steps, steps+step):
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
                    style_score *= self.style_weight
                    content_score *= self.content_weight
                    loss = style_score + content_score
                    scores[0] = style_score
                    scores[1] = content_score
                    # обратное распространение ошибки
                    loss.backward()

                    return style_score + content_score

                optimizer.step(closure)
            # вывод инфы каждые 100 итераций
            print("run %s:" % steps)
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(scores[0].item(), scores[1].item()))
            print()
            # Вывод промежуточных изображений
            #img = transforms.ToPILImage()(input_img.cpu().clone().squeeze(0))
            #img.save('{}.png'.format(params[0]), 'png')
            #plt.imshow(img)
            #plt.pause(0.001)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    def process_image(self, img_stream, imsize):
        image = Image.open(img_stream)
        transform = transforms.Compose([
            transforms.Resize(imsize),  # нормируем размер изображения
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # превращаем в удобный формат
        image = transform(image).unsqueeze(0)
        return image.to(device, torch.float)

    def transfer_style(self, content_img_stream, style_img_stream):
        # Этот метод по переданным картинкам в каком-то формате (PIL картинка, BytesIO с картинкой
        # или numpy array на ваш выбор). В телеграм боте мы получаем поток байтов BytesIO,
        # а мы хотим спрятать в этот метод всю работу с картинками, поэтому лучше принимать тут эти самые потоки
        # и потом уже приводить их к PIL, а потом и к тензору, который уже можно отдать модели.

        style_img = self.process_image(style_img_stream, self.imsize)
        content_img = self.process_image(content_img_stream, self.imsize)
        input_img = content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        #input_img = torch.randn(content_img.data.size(), device=device)

        output = self.run_style_transfer(content_img, style_img, input_img)
        return transforms.ToPILImage()(output.cpu()[0]) # тензор в картинку
