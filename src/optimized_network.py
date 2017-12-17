from network import Network,sigmoid,sigmoid_prime
import numpy as np
import pdb

class OptimizedNetwork(Network):
	def __init__(self, sizes):
		Network.__init__(self,sizes)

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		
		pdb.set_trace()

		x = mini_batch[0][0]
		y = mini_batch[0][1]

		for item in mini_batch[1:]:
			x = np.append(x, item[0], axis=1)
			y = np.append(y, item[1], axis=1)

		x = np.array(x)
		y = np.array(y)

		nabla_b, nabla_w = self.backprop(x,y)

		# nabla_w represents the accumulated changes we must apply to self.weights across all the training samples
		# nabla_b represents the deltas found for each training sample in the column axis. We must sum them into one vector

		nabla_b = [b.sum(axis=1,keepdims=True) for b in nabla_b]

		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases  = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		# x is a matrix of test data, with each test case indexed in the column direction
		# y is a matrix of test data answers, with each answer indexed in the column direction

		# nabla_b represent the derivitives of the biases for each layer
		# nabla_w represent the derivatives of the weights for each layer

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		nabla_b = []
		nabla_w = []

		test_data_len = x.shape[1]

		for b, w in zip(self.biases, self.weights):
			nabla_b.append(np.zeros((b.shape[0], test_data_len)))
			nabla_w.append(np.zeros(w.shape))
		
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer

		for b, w in zip(self.biases, self.weights):
	 		z = np.dot(w, activation)+b
	 		zs.append(z)
	 		activation = sigmoid(z)
	 		activations.append(activation)

		# backward pass
		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.

		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)

