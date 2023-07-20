import numpy as np
import random
import matplotlib.pyplot as plt
from helpers import load_mnist, initialize_parameters, learning_rate_optimizer, forward_propogation, backpropogation, save_model_parameters

## INITIALIZATION ##

train_images, train_labels, test_images, test_labels = load_mnist()

num_inputs = len(train_images)
input_dim = (28, 28, 1)
input_size, _, input_depth = input_dim
filter_dim = (3, 3, 32)
filter_size, _, num_filters = filter_dim
pool_dim = (2, 2, 2)
pool_size, _, stride = pool_dim
pooled_dim = (13, 13, 32)
pooled_size, _, num_maps = pooled_dim
num_neurons = 100
num_outputs = 10

conv_weights, conv_biases, fc_weights, fc_biases, output_weights, output_biases = initialize_parameters(input_dim, filter_dim, num_neurons, num_outputs)
initial_learning_rate = 1e-3
num_epochs = 5

plt.ion() 
fig, ax = plt.subplots()


## TRAINING ##

for epoch in range(num_epochs):

    loss = []
    total_correct = 0
    learning_rate = learning_rate_optimizer(initial_learning_rate, epoch, decay_rate=0.1)

    for i in range(num_inputs):
        n = random.randint(0, num_inputs-1)
        image = train_images[n][:].reshape(28, 28)
        label = train_labels[n]
        
        # Forward Propogation #
        z, conv_out, pool_out, max_indices, flattened, unscaled_mask, y, fc_out, output_out, predicted_value = forward_propogation(image, conv_weights, conv_biases, fc_weights, fc_biases, output_weights, output_biases)

        # Backpropogation #
        ce_loss, output_dW, output_db, fc_db, fc_dW, conv_dW, conv_db = backpropogation(image, label, output_weights, fc_weights, pooled_dim, pool_dim, z, conv_out, pool_out, max_indices, flattened, unscaled_mask, y, fc_out, output_out)

        # Parameter Update #
        output_biases = output_biases - learning_rate * output_db
        output_weights = output_weights - learning_rate * output_dW
        fc_biases = fc_biases - learning_rate * fc_db
        fc_weights = fc_weights - learning_rate * fc_dW
        conv_biases = conv_biases - learning_rate * conv_db
        conv_weights = conv_weights - learning_rate * conv_dW


        # Plotting #
        if (predicted_value == label):
            total_correct += 1
            
        loss.append(ce_loss)

        if i % 100 == 0:
            print(f"(Epoch: {epoch+1}, Iteration: {i}) => Training Accuracy: {total_correct/len(train_images)*100:.2f}%, Average Loss: {round(np.average(loss[-100]), 5) if len(loss) > 100 else 0}")

            ax.clear()
            ax.plot(list(range(0, len(loss))), loss)
            plt.title("Gradient Descent Progress")
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            fig.canvas.draw()
            plt.pause(0.01)

    save_model_parameters(epoch, 'model_parameters/', conv_weights=conv_weights, conv_biases=conv_biases, fc_weights=fc_weights, fc_biases=fc_biases, output_weights=output_weights, output_biases=output_biases)

plt.ioff()


## TESTING ##

num_tests = len(test_images)
num_correct_predictions = 0

for i in range(num_tests):
    image = test_images[i][:].reshape(28, 28)
    label = test_labels[i]
    
    *_, predicted_value = forward_propogation(image, conv_weights, conv_biases, fc_weights, fc_biases, output_weights, output_biases, dropout=False)
    if predicted_value == label:
        num_correct_predictions += 1

accuracy = num_correct_predictions / num_tests
print(f"Test Accuracy: {accuracy*100:.2f}%")