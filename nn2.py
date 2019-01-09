def cost(b):
	return (b-4)**2
def slope(b):
	return 2*(b-4)
#apply the slope to minimize 
#the cost and bring b close to the target
b = 6
#loop to simulate the training of a
#neural network
for i in range(10):
	b = b - .1*slope(b)
	print b
	