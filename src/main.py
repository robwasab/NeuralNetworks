from optimized_network import OptimizedNetwork as OptNet
from mnist_loader import load_data_wrapper
from network import Network

print 'loading training data'
training_data, validation_data, test_data = load_data_wrapper()

print 'creating network'
net = OptNet([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
