from Layer import *
import json

m = 1000
bytes_to_read = m * 784
theta_filename = 'parameters/min_cost.json'
# theta_filename = 'parameters/last_epoch.json'
testfile_image = 'data/t10k-images-idx3-ubyte'
testfile_label = 'data/t10k-labels-idx1-ubyte'

# Read thetas
with open(theta_filename, 'r') as f:
    js_obj = json.load(f)

# Read testing set
with open(testfile_image, 'rb') as f:
    meta = f.read(16)
    raw = f.read(bytes_to_read)
x = reshape(array([raw[i] for i in range(bytes_to_read)]), (m,784))
x = insert(x, 0, 1, axis=1)

# Read testing set labels
with open(testfile_label, 'rb') as f:
    meta = f.read(8)
    raw = f.read(m)
y = array([raw[i] for i in range(m)])

network = []
# Input layer
network.append(Layer(a=x,theta=reshape(array(js_obj['theta1']),(25,785))))
# Two hidden layers
network.append(Layer(theta=reshape(array(js_obj['theta2']),(25,26))))
network.append(Layer(theta=reshape(array(js_obj['theta3']),(10,26))))
# Output layer
network.append(Layer())

network[1].activate(network[0],next_to_input=True)
for i in range(2,len(network)):
    network[i].activate(network[i-1])

network_predictions = array([])
for a in network[3].a:
    temp, prediction = 0, 0
    for confidence in range(a.size):
        if confidence > temp:
            temp = a[confidence]
            prediction = confidence
    network_predictions = append(network_predictions, prediction)

correct_predictions = 0
for i in range(m):
    if y[i] == network_predictions[i]:
        correct_predictions += 1

print('Correct network predictions:', str(100*correct_predictions/m)+'%')