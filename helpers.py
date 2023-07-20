import os
import re
import numpy as np

## HELPER FUNCTIONS ##

def load_mnist():
    dataset_dir = 'mnist_dataset/'
    data_types = ['train', 't10k']

    result = []
    for dtype in data_types:
        labels_path = f'{dataset_dir}/{dtype}-labels.idx1-ubyte'
        images_path = f'{dataset_dir}/{dtype}-images.idx3-ubyte'

        with open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels = (labels).astype(np.int64)

        with open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            images = (np.expand_dims(images, axis=-1)/255.).astype(np.float32)

        result.extend([images, labels])

    return tuple(result)

def save_model_parameters(epoch, path, **parameters):
    epoch_path = os.path.join(path, f'epoch_{epoch+1}')
    os.makedirs(epoch_path, exist_ok=True)
    for name, values in parameters.items():
        np.save(os.path.join(epoch_path, f'{name}.npy'), values)

def load_model_parameters(path, epoch=None, last_epoch=False, **parameters):
    if last_epoch:
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        epoch_dirs = [d for d in subdirs if re.match(r'.*/epoch_\d+', d)]
        epoch_numbers = [int(re.search(r'epoch_(\d+)', d).group(1)) for d in epoch_dirs]
        epoch = max(epoch_numbers)
    
    epoch_path = os.path.join(path, f'epoch_{epoch}')
    for name in parameters.keys():
        parameters[name] = np.load(os.path.join(epoch_path, f'{name}.npy'))
    
    return parameters

def convolve(X, filters):
    num_filters, filter_h, filter_w = filters.shape
    X_h, X_w = X.shape

    feature_map_h, feature_map_w = X_h - filter_h + 1,  X_w - filter_w + 1
    feature_map = np.zeros((num_filters, feature_map_h, feature_map_w))

    for filter_index in range(num_filters):
        for x in range(feature_map_w):
            for y in range(feature_map_h):
                feature_map[filter_index, y, x] = np.sum(X[y: y + filter_h, x: x + filter_w] * filters[filter_index])
    
    return feature_map

def pool(X, pool=(2, 2), stride=2, type='max'):
    num_maps, X_h, X_w = X.shape
    pool_h, pool_w = pool

    downsampled_h, downsampled_w = int((X_h - pool_h) / stride) + 1, int((X_w - pool_w) / stride) + 1
    downsampled = np.zeros((num_maps, downsampled_h, downsampled_w))
    max_indices = np.zeros((num_maps, downsampled_h, downsampled_w, 2), dtype=int)

    for m in range(num_maps):
        for x in range(0, X_w, stride):
            for y in range(0, X_h, stride):
                if type == 'max':
                    downsampled[m, int(y/stride), int(x/stride)] = np.max(X[m, y: y + pool_h, x: x + pool_w])
                    max_y, max_x = np.unravel_index(np.argmax(X[m, y: y + pool_h, x: x + pool_w]), (pool_h, pool_w))
                    max_indices[m, int(y/stride), int(x/stride)] = [y + max_y, x + max_x]
    return downsampled, max_indices

def ReLU(x):
    return np.maximum(0, x)

def grad_ReLU(z):
    return np.where(z > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

def dropout_mask(shape, dropout_rate=0.1):
    unscaled_mask = np.random.rand(*shape) > dropout_rate
    scaled_mask = unscaled_mask / (1.0 - dropout_rate)
    return unscaled_mask, scaled_mask

def he_initialization_conv(filter_dim):
    filter_size, _, num_filters = filter_dim
    num_inputs_per_filter = filter_size ** 2
    return np.random.randn(num_filters, filter_size, filter_size) * np.sqrt(2.0 / num_inputs_per_filter)

def he_initialization_fc(input_dim, filter_dim, num_outputs):
    input_size, _, _ = input_dim
    filter_size, _, num_filters = filter_dim
    num_inputs_per_filter = int((input_size - filter_size + 1) / 2) ** 2
    return np.random.randn(num_outputs, num_inputs_per_filter * num_filters) * np.sqrt(2.0 / num_inputs_per_filter)

def he_initialization_out(num_neurons, num_outputs):
    return np.random.randn(num_outputs, num_neurons) * np.sqrt(2.0 / num_neurons)

def initialize_parameters(input_dim, filter_dim, num_neurons, num_outputs):
    conv_weights = he_initialization_conv(filter_dim)
    conv_biases = np.zeros((filter_dim[2], 1))
    fc_weights = he_initialization_fc(input_dim, filter_dim, num_neurons)
    fc_biases = np.zeros(num_neurons)
    output_weights = he_initialization_out(num_neurons, num_outputs)
    output_biases = np.zeros(num_outputs)

    return conv_weights, conv_biases, fc_weights, fc_biases, output_weights, output_biases

def one_hot_encode(label, output_size=10):
    one_hot = np.zeros(output_size)
    one_hot[label] = 1
    return one_hot

def learning_rate_optimizer(initial_learning_rate, epoch, decay_rate=0.1):
    return initial_learning_rate * np.exp(-decay_rate * epoch)

def forward_propogation(image, conv_weights, conv_biases, fc_weights, fc_biases, output_weights, output_biases, dropout=True):
    z = convolve(image, conv_weights + conv_biases.reshape(-1, 1, 1))
    conv_out = ReLU(z)
    pool_out, max_indices = pool(conv_out)
    flattened = pool_out.flatten()
    y = np.dot(fc_weights, flattened) + fc_biases
    fc_out = ReLU(y)
    unscaled_mask, scaled_mask = dropout_mask(fc_out.shape)
    if dropout:
        fc_out *= scaled_mask
    output_out = softmax(np.dot(output_weights, fc_out) + output_biases)
    predicted_value = np.argmax(output_out)
    return z, conv_out, pool_out, max_indices, flattened, unscaled_mask, y, fc_out, output_out, predicted_value

def cross_entropy_loss(output_out, label):
    epsilon = 1e-8
    ce_loss = -np.sum(one_hot_encode(label) * np.log(output_out + epsilon))
    return np.mean(ce_loss)

def cross_entropy_loss_differential(output_out, label):
    return output_out - one_hot_encode(label)

def backprop_pool(conv_out, conv_dL_pooled, pooled_dim, pool_dim, max_indices, pool_type='max'):
    pool_h, pool_w, stride = pool_dim
    pooled_h, pooled_w, num_maps = pooled_dim
    conv_dL = np.zeros_like(conv_out)
    for m in range(num_maps):
        for x in range(0, pooled_w, stride):
            for y in range(0, pooled_h, stride):
                if pool_type == 'average':
                    conv_dL[m, y: y + pool_h, x: x + pool_w] = conv_dL_pooled[m, int(y/stride), int(x/stride)] / (pool_h * pool_w)
                if pool_type == 'max':
                    h, w = max_indices[m, y, x]
                    conv_dL[m, h, w] = conv_dL_pooled[m, int(y/stride), int(x/stride)]
    return conv_dL

def backpropogation(image, label, output_weights, fc_weights, pooled_dim, pool_dim, z, conv_out, pool_out, max_indices, flattened, unscaled_mask, y, fc_out, output_out, dropout=True):
    ce_loss = cross_entropy_loss(output_out, label)
    output_dL = cross_entropy_loss_differential(output_out, label)
    output_db = output_dL
    output_dW = np.outer(output_dL, fc_out)

    fc_dL = np.dot(output_weights.T, output_dL) * grad_ReLU(y)
    if dropout:
        fc_dL *= unscaled_mask
    fc_db = fc_dL
    fc_dW = np.outer(fc_dL, flattened)
    
    conv_dL_pooled = np.dot(fc_dL, fc_weights).reshape(pool_out.shape)
    conv_dL = backprop_pool(conv_out, conv_dL_pooled, pooled_dim, pool_dim, max_indices, pool_type='max')    
    conv_dLdz = grad_ReLU(z) * conv_dL
    conv_db = np.sum(conv_dLdz, axis=(1,2)).reshape(-1,1) 
    conv_dW = convolve(np.flip(np.flip(image, 0), 1), conv_dLdz)

    return ce_loss, output_dW, output_db, fc_db, fc_dW, conv_dW, conv_db