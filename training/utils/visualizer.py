import torch
import numpy as np
import random
import copy

from scipy.ndimage.filters import gaussian_filter1d
import torchvision.transforms as T


SQUEEZENET_MEAN = np.array([0.5] * 3)
SQUEEZENET_STD = np.array([0.5] * 3)


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def create_unit_visualization(target, model, dtype, layer=None,
                              layerType=None, singleFilterUnit=False, filterN=None, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.

    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    - dtype: Torch datatype to use for computations

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to gjitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    model.type(dtype)
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 500)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 100)

    # Randomly initialize the image as a PyTorch Tensor, and make it require
    # gradient.
    img = torch.randn(1, 3, 224, 224).mul_(1.0).type(dtype).requires_grad_()

    if layerType == 'FEATLAYER':
        act_model = copy.deepcopy(model)
        act_model = torch.nn.Sequential(
            *list(act_model.module.features.children())[:layer + 1])
        # act_model = torch.nn.Sequential(*list(act_model.features.children())[:layer+1])
    elif layerType == 'CLASSLAYER':
        act_model = copy.deepcopy(model)
        act_model.module.classifier = torch.nn.Sequential(
            *list(act_model.module.classifier.children())[:layer + 1])
        # act_model.classifier = torch.nn.Sequential(*list(act_model.classifier.children())[:layer+1])
    else:
        act_model = model

    # print(act_model)

    for t in range(num_iterations):
        # Randomly jitter the image a bit; this gives slightly nicer results
        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
        img.data.copy_(jitter(img.data, ox, oy))

        if 'iteration' in kwargs:
            if kwargs['iteration'] == 0 and t == 0:
                print('image shape', img.shape)
        #######################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term!                                              #
        # Be very careful about the signs of elements in your code.            #
        #######################################################################
        if layerType == 'FEATLAYER':
            activations = act_model(img)
            if 'iteration' in kwargs:
                if kwargs['iteration'] == 0 and t == 0:
                    print('whole activations.shape', activations.shape)

            activation = activations[:, target]
            if 'iteration' in kwargs:
                if kwargs['iteration'] == 0 and t == 0:
                    print('target activation.shape', activation.shape)

            if singleFilterUnit:
                # activation
                shape = activation.shape
                width = shape[1]
                height = shape[2]
                i = width // 2
                j = height // 2
                activation = activation[:, i, j]
                scores = activation
                if 'iteration' in kwargs:
                    if kwargs['iteration'] == 0 and t == 0:
                        print('unit activation.shape', activation.shape)
            else:
                activation = torch.reshape(
                    input=activation, shape=[
                        activation.shape[0], -1])
                scores = torch.norm(activation)
        elif layerType == 'CLASSLAYER':
            activations = act_model(img)
            # print('activations.shape', activations.shape)
            activation = activations[:, target]
            activation = torch.reshape(
                input=activation, shape=[
                    activation.shape[0], -1])
            scores = torch.norm(activation)
        else:
            scores = act_model(img)
            scores = scores[:, target]
            scores = torch.reshape(input=scores, shape=[scores.shape[0], -1])

        score = scores - (l2_reg * torch.norm(img))
        score.backward()
        img.data += (learning_rate * img.grad.data / torch.norm(img.grad.data))
        img.grad.data.zero_()
        #######################################################################
        #                             END OF YOUR CODE                         #
        #######################################################################

        # Undo the random jitter
        img.data.copy_(jitter(img.data, -ox, -oy))

        # As regularizer, clamp and periodically blur the image
        for c in range(3):
            lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
            hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
            img.data[:, c].clamp_(min=lo, max=hi)
        if t % blur_every == 0:
            blur_image(img.data, sigma=0.5)

        # Periodically show the image
        # if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
        #    plt.imshow(deprocess(img.data.clone().cpu()))
        #    class_name = target #class_names[target_y]
        #    plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
        #    plt.gcf().set_size_inches(4, 4)
        #    plt.axis('off')
        #    plt.show()

    del act_model
    torch.cuda.empty_cache()

    return deprocess(img.data.cpu())
