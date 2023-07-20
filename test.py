from helpers import load_model_parameters, load_mnist, forward_propogation

## TESTING ##

train_images, train_labels, test_images, test_labels = load_mnist()
params = load_model_parameters('model_parameters/', last_epoch=True, conv_weights=None, conv_biases=None, fc_weights=None, fc_biases=None, output_weights=None, output_biases=None)

conv_weights = params['conv_weights']
conv_biases = params['conv_biases']
fc_weights = params['fc_weights']
fc_biases = params['fc_biases']
output_weights = params['output_weights']
output_biases = params['output_biases']

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
