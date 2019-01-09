import numpy
import pyttsx
engine= pyttsx.init()
phrases = ['seems like its','i guess','i think','possibly']
data = [[3,1.5,1],[2,1,0],[4,1.5,1],[3,1,0],[2,5,0],[5.5,1,1],[1,1,0]]

rand_data = data[numpy.random.randint(len(data))]
m1 = rand_data[0]
m2 = rand_data[1]
print(m1)
print(m2)
def NN(m1,m2,w1,w2,b):
	z = m1*w1 + m2*w2+b
	return(sigmoid(z))
def sigmoid(x):
	return 1/(1+numpy.exp(-x))

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

prediction = NN(m1,m2,w1,w2,b)
prediction_text = ["blue","red"][int(numpy.round(prediction))]
phrase = numpy.random.choice(phrases)+" "+prediction_text
engine.say(phrase)
engine.runAndWait()