from matplotlib import pyplot as plt
import numpy as np
#library for sound
import pyttsx
engine = pyttsx.init()
#length width type(0,1)
data = [[3,1.5,1],
		[2,1,0],
		[4,1.5,1],
		[3,1,0],
		[3.5,0.5,1],
		[2,0.5,0],
		[5.5,1,1],
		[1,1,0]]
mystery_flower = [4.5,1]
#network
#   0    flower type
#  / \   w1 w2 b
# 0   0  length width


#activation function 
def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_p(x):
	return sigmoid(x)*(1-sigmoid(x))
#plot 
T = np.linspace(-6,6,100)#from -5 to 5 with ten steps
Y = sigmoid(T)
Y2 = sigmoid_p(T)
#actual plotting 
"""plt.plot(T,Y,c='r')
plt.plot(T,Y2,c='b')
plt.show()"""

#scatter plot
"""plt.grid()
plt.axis([0,6,0,6])
for i in range (len(data)):
	point = data[i]
	color = "r"
	if point[2]==0:
		color = "b"
	plt.scatter(point[0],point[1],c = color)
plt.show()"""
#training loop
#grab one of the points randomly and see what the network output is
#and use it to get the derivative of the cost
#and bring the derivative back to th parameters
#and update them to minimize the cost and get the
#prediction of the nn closer to correct

#small fraction to to avoid overshooting
learning_rate = 0.2
#keep track of all the costs 
costs=[]

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

#training
for i in range(50000):
	#random index
	ri = np.random.randint(len(data))
	#random point
	point = data[ri]
	#feed the point to the nn
	#var z is the weighted average of the point features and the bias
	
	z = point [0]*w1+point[1]*w2+b
	#print z
	#variable pred 
	#pred is the application of the activation function
	pred = sigmoid(z)
	#print (h)
	target = point[2]
	cost = np.square(pred-target)
	print (point,cost)
	
	#take derivative of the cost with respect to each of our parameters
	#derivative of the cost wrt to prediction
	#power rule
	dcost_pred = 2*(pred-target)
	#derivative of the cost wrt to z
	dpred_dz = sigmoid_p(z)
	#derivative of z wrt to w1
	dz_dw1 = point[0]
	#derivative of z wrt to w2
	dz_dw2 = point[1]
	#derivative of z wrt to b
	dz_db = 1
	
	dcost_dz = dcost_pred* dpred_dz
	#chain the above derivatives together to find the derivative of the cost
	dcost_dw1 = dcost_dz * dz_dw1
	dcost_dw2 = dcost_dz* dz_dw2
	dcost_db = dcost_dz* dz_db
	
	w1 = w1 - learning_rate*dcost_dw1
	w2 = w2 - learning_rate*dcost_dw2	
	b = b - learning_rate*dcost_db	
	
	if i % 100 == 0:
		cost_sum = 0
		for j in range(len(data)):
			point = data[ri]
			z = point[0]*w1+ point[1]*w2 + b
			pred = sigmoid(z)
			
			target = point[2]
			cost_sum += np.square(pred-target)
		costs.append(cost_sum/len(data))
		
def which_flower(l,w):
	z = l*w1 + w*w2+b
	pred = sigmoid(z)
	if pred<.5:
		engine.say("blue flower")
		print("blue flower:",pred)
	else:
		engine.say("red flower")
		print("red flower:",pred)




which_flower(3,1.5)
which_flower(2,1)
which_flower(4,1.5)
which_flower(3,1)
which_flower(3.5,.5)
which_flower(2,.5)
which_flower(5.5,1)
which_flower(1,1)
which_flower(4.5,1)
engine.runAndWait()

plt.title("cost function")
plt.plot(costs)
plt.show()


