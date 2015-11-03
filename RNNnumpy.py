import numpy as np
from scipy.stats import norm

class RNNnumpy:

	def __init__(self, input_dim, hidden_dim,output_dim,w=None,mu=0,variance=0.1,model_type='Elman'):
		self.input_dim  = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.mu = mu
		self.variance = variance
		self.w_dim = self.hidden_dim*(self.input_dim+2)+self.output_dim*(self.hidden_dim+1)
		modeltypes = ['Elman','Jordan']

		if(model_type in modeltypes):
			self.model_type = model_type
		else:
			print('Error: Model type can only be',modeltypes)
			return

		# Initialize parameters
		if(w==None):
			if(self.model_type == 'Elman'):
				self.W_H = np.random.uniform(-np.sqrt(1./input_dim),np.sqrt(1./input_dim),(hidden_dim,input_dim+1))
				self.W_O = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(output_dim,hidden_dim+1))
				self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
			elif(self.model_type == 'Jordan'):
				self.W_H = np.random.uniform(-np.sqrt(1./input_dim),np.sqrt(1./input_dim),(hidden_dim,input_dim+2))
				self.W_O = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(output_dim,hidden_dim+1))
		elif(w!=None):
			if(self.model_type == 'Elman'):
				self.W_H = np.random.uniform(-np.sqrt(1./input_dim),np.sqrt(1./input_dim),(hidden_dim,input_dim+1))
				self.W_O = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(output_dim,hidden_dim+1))
				self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
			elif(self.model_type == 'Jordan'):
				self.W_H, self.W_O = self.w_vec2mat(w,input_dim,hidden_dim,output_dim) 

	def w_vec2mat(self,w,input_dim,hidden_dim,output_dim):
		w_H = w[0:(hidden_dim*(input_dim+2))]
		id_H = hidden_dim*(input_dim+2)
		w_O = w[id_H:(id_H+1+output_dim*(hidden_dim+1))]
		W_H = np.reshape(w_H,(hidden_dim,input_dim+2))
		W_O = np.reshape(w_O,(output_dim,hidden_dim+1))
		return (W_H,W_O)

	def logi_fun(self,x):
		return 1/(1+np.exp(-x))	
	
	def rect_fun(self,x):
		return np.log(1+np.exp(x)) 

	def forward_prop(self,y):
		T = len(y)
		x = np.ones((T+1,self.input_dim+2))
		x[:,1] = np.hstack((np.array(y),[0]))
		
		if(self.model_type == 'Elman'):
 			# The hidden state
			print('Not done yet')
# 			state     = np.ones((T+1,self.hidden_dim+1))
# 			sigma2    = np.zeros((T+1,self.output_dim))
# 			for t in range(1,T):
# 				state[t]  = self.logi_fun(  self.W_H.dot(y[t]) + self.W.dot(state[t-1]).reshape(self.hidden_dim,1) ).reshape(self.hidden_dim,)
# 				sigma2[t] = np.exp(   self.W_O.dot(state[t]).reshape(self.output_dim,1) )

		elif(self.model_type == 'Jordan'):
			# The hidden state
			state      = np.ones((T+1,self.hidden_dim+1))
			sigma2     = np.ones(T+1)*self.variance    
			for t in range(1,T):
				x[t,2] = sigma2[t-1]
				#print(state[t,1:self.hidden_dim],'\n',state[t,1:self.hidden_dim].shape)
				#print(self.W_H.dot(x[t]))
				state[t,1:] = self.logi_fun(  self.W_H.dot(x[t]) ).reshape(self.hidden_dim,)
				sigma2[t] = np.exp( self.W_O.dot(state[t]) ).reshape(self.output_dim,)
				#print('x[t]',x[t])
				#print('state[t]',state[t])
				#print('sigma2[t]',sigma2[t])
		else:
			print('Error in model type.')
			return []

		#return[state[0:T],sigma2[0:T]]
		return sigma2[0:T]

	def log_likelihood(self,w,y,lam=1):
		self.W_H, self.W_O = self.w_vec2mat(w,self.input_dim,self.hidden_dim,self.output_dim) 
		y = np.array(y)
		T = len(y)
		sigma2 = self.forward_prop(y)
		log_like = 1/2 * T * np.log(2*np.pi) + 1/2*sum(np.log(sigma2)+y**2/sigma2) + lam/2*w.T.dot(w)
		
		return log_like/T

	def VaR(self,y,pct=[0.01,0.025,0.05]):
		est_variance = self.forward_prop(y)
		VaR = {}
		for alpha in pct:
			VaR[str(alpha)] = self.mu + norm.ppf(alpha)*np.sqrt(est_variance)
		return VaR


