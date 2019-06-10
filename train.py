from Layer import *

m = 1000
bytes_to_read = m * 784
lame = 0.05  # Inside joke: this is the regularization term lambda
alpha = 0.3 # Learning rate term

# Parsing the training data and their labels
with open("data/train-images-idx3-ubyte", "rb") as f:
    meta = f.read(16)
    # raw_images = f.read(47040000)
    raw_images = f.read(bytes_to_read)
x = reshape(array([raw_images[i] for i in range(bytes_to_read)]), (m,784))
x = insert(x, 0, 1, axis=1)
with open("data/train-labels-idx1-ubyte", "rb") as f:
    meta = f.read(8)
    raw_labels = f.read(m)
training_labels = reshape(array([raw_labels[i] for i in range(m)]), (m,1))

# Converting the labels read into the output layer format (0-9)
y = zeros((m,10))
for i in range(m):
    y[i] = array([0 for a in range(training_labels[i][0])]+[1]+[0 for b in range(9-training_labels[i][0])])

# Cost function
def J(nn, training_data):
    difference = 0
    for i in range(10):
        temp = y[:,i]*log(nn[3].a[:,i]) + (1-y[:,i])*log(1-nn[3].a[:,i])
        difference += temp.sum()
    regularization = 0
    for i in range(3):
        temp = pow(nn[i].theta[:,1:],2)
        regularization += temp.sum()
    return difference/-m + regularization*lame/(2*m)

# The thetas are randomly initialized to break symmetry
network = []
# Input layer
network.append(Layer(a=x,theta=reshape(random.random(19625),(25,785))))
# Two hidden layers
network.append(Layer(theta=reshape(random.random(650),(25,26))))
network.append(Layer(theta=reshape(random.random(260),(10,26))))
# Output layer
network.append(Layer())

def save(nn, filename):
    import json
    js_obj = {}
    js_obj['cost'] = J(nn,y)    # using "global" instance y
    for i in range(len(nn)-1):
        js_obj['theta'+str(i+1)] = list(reshape(nn[i].theta, (nn[i].theta.size,)))
    with open(filename,'w') as fi:
        json.dump(js_obj, fi)

def train(nn, epoch):
    temp_cost = 50

    # using the instance variable nn
    def forward_propagation():
        nn[1].activate(nn[0],next_to_input=True)
        for i in range(2,len(nn)):
            nn[i].activate(nn[i-1])
    def back_propagation():
        nn[3].delt(y, m, output_layer=True) # "global" instance y
        nn[2].delt(nn[3], m, next_to_output=True)
        for i in range(len(nn)-3,-1,-1):
            nn[i].delt(nn[i+1],m)

        # TODO: Don't penelize theta 0's !!!
        for i in range(len(nn)-2,-1,-1):
            nn[i].theta -= nn[i].gradiant / m
            nn[i].theta[:,1:] -= nn[i].theta[:,1:] * lame / m

    for i in range(epoch):
        forward_propagation()
        cost = J(nn,y)
        # Save the thetas that produce the lowest cost
        if cost < temp_cost:
            save(nn,'parameters/min_cost.json')
            temp_cost = cost
        print('Cost:', cost)
        # print('Network Output:\n', nn[3].a)
        # print('Desired Output:\n', y)
        back_propagation()

train(network, 50)
network[1].activate(network[0],next_to_input=True)
for i in range(2,len(network)):
    network[i].activate(network[i-1])
print(network[3].a[50])
print(y[50])
# Save the thetas from the last descent/epoch
save(network,'parameters/last_epoch.json')