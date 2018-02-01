# Import the needed packages
from __future__ import division
from chainer import Chain, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import numpy as np


""" Classes that define the structure of the CheckerZero network """


""" Definition of a Convblock """
class ConvBlock(Chain):
        
        '''
		Architecture of a convolutional block
		
		Args:
			n_out: Amount of filters the convolutional layer outputs
        '''
	def __init__(self, n_out):

		super(ConvBlock, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(in_channels=None, out_channels=n_out, 
				ksize=3, stride=1, pad=1)
			self.bn = L.BatchNormalization(n_out)

	'''
                Function call which takes network input x and outputs the result of a convolutional block
        '''
	def __call__(self, x):
		y = F.relu(self.bn(self.conv(x)))
		return y


""" Definition of a Resblock """
class ResBlock(Chain):
        
        '''
		Architecture of a residual block
		
		Args:
			n_out: Amount of filters the convolutional layers output
        '''
	def __init__(self, n_out):
		super(ResBlock, self).__init__()
		with self.init_scope():
			self.res_conv1 = L.Convolution2D(in_channels=None, out_channels=n_out, 
				ksize=3, stride=1, pad=1)
			self.res_bn1 = L.BatchNormalization(n_out)
			self.res_conv2 = L.Convolution2D(in_channels=None, out_channels=n_out, 
				ksize=3, stride=1, pad=1)
			self.res_bn2 = L.BatchNormalization(n_out)

	'''
                Function call which takes network input x and outputs the result of a residual block
        '''
	def __call__(self, x):
		h1 = F.relu(self.res_bn1(self.res_conv1(x)))
		h2 = F.relu(self.res_bn2(self.res_conv2(h1)))
		x = F.concat((h2,x))
		y = F.relu(x)
		return y


""" Definition of the CheckerZero network made out of Convblocks, Resblocks and the two tails """
class NetworkZero(Chain):
        
        '''
		Architecture of the network
		
		Args:
			output_size: Amount of filters each convolutional layer outputs
			nr_resblock: Amount of residual blocks the network uses
			nr_policyoutputs: Amount of outputs the policy network should generate
	
        '''
	def __init__(self, output_size, nr_resblocks=9, nr_policyoutputs=4096):

		super(NetworkZero, self).__init__()
		self.nr_resblocks = nr_resblocks
		
		# Add Convolutional block
		self.add_link('convblock', ConvBlock(output_size))
		
		# Add nr_resblocks amount of Residual blocks
		for i in range(1,nr_resblocks+1):
			self.add_link('resblock%d' %i,ResBlock(output_size))
		
		# Policy head
		self.add_link('policy_conv',L.Convolution2D(in_channels=None, out_channels=2,
				ksize=1, stride=1, pad=0))
		self.add_link('policy_bn',L.BatchNormalization(2))
		self.add_link('policy_fc', L.Linear(None, nr_policyoutputs))
		
		# Value head
		self.add_link('value_conv',L.Convolution2D(in_channels=None, out_channels=1,
				ksize=1, stride=1, pad=0))
		self.add_link('value_bn',L.BatchNormalization(1))
		self.add_link('value_fc1', L.Linear(None, output_size))
		self.add_link('value_fc2', L.Linear(None, 1))   
	
	'''
                Function call which takes board input x and outputs the policy vector and value
        '''
	def __call__(self, x):
		# General function calls
		h = self.convblock(x)
		
		for i in range(1,self.nr_resblocks+1):
			h = self['resblock%d' % i](h)
			
		# Policy head function calls
		h1_policy = F.relu(self.policy_bn(self.policy_conv(h)))
		h2_policy = self.policy_fc(h1_policy)
		y_policy = F.softmax(h2_policy)
		
		# Value head function calls
		h1_value = F.relu(self.value_bn(self.value_conv(h)))
		h2_value = F.relu(self.value_fc1(h1_value))
		y_value = F.tanh(self.value_fc2(h2_value))
		
		return (y_policy, y_value)


""" Definition of the classifier used by the CheckerZero network to determine the output of the network and the loss """
class Classifier(Chain):
        '''
		Classifier for NetworkZero which is initialized in this class
        '''
	def __init__(self, output_size, nr_resblocks, nr_policyoutputs):

		super(Classifier, self).__init__()
		with self.init_scope():
			self.predictor = NetworkZero(output_size=output_size, nr_resblocks=nr_resblocks, nr_policyoutputs=nr_policyoutputs)
	
	
	'''
                Returns policy vector and value in numpy when input x is given to the network
        '''
	def get_policy_and_value(self, x):
		# Predict values from input with network and tranform to numpy
		y_policy, y_value = self.predictor(x)
		y_policy = y_policy.data[0]
		y_value = y_value.data[0]
		return (y_policy, y_value)
	
	
	'''
                Function call which takes input x, and the expected values for the policy and the target
        '''
	def __call__(self, x, t_policy, t_value):
		# Get a prediction from the network given data
		y_policy, y_value = self.predictor(x)

		# Calculate total loss (cross-entropy and MSE are weighted equally)
		loss = self.loss_function(y_policy, t_policy, y_value, t_value)
		
		return loss


        '''
                Calculate loss function
        '''
	def loss_function(self, y_policy, t_policy, y_value, t_value):
		y_policy = y_policy.data[0].squeeze()
		loss_policy = F.matmul(t_policy, F.log(y_policy),transa = True)
		loss_value = F.mean_squared_error(y_value, t_value)
		sum_loss = loss_value-loss_policy
		
		return sum_loss
        
