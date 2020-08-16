# import the necessary packages
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from sklearn import linear_model
import random
import numpy as np
import cv2
import tensorflow as tf
from sklearn.pipeline import Pipeline


# Loading a needed function
def binaryToDecimal(binary): 
    binary1 = binary 
    decimal, i, n = 0, 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
        return decimal

# Loading and separating data
samples = []
flaw = cv2.cvtColor(cv2.imread("Fill In Photo.jpg"), cv2.COLOR_BGR2GRAY)
cflaw = []
for y in range(0, len(flaw)):
    row = []
    for x in range(0, len(flaw[y])):
        row.append([int(z) for z in list(bin(flaw[y][x])[2:].zfill(8))])
    cflaw.append(row)
cflaw = np.array(cflaw).flatten()
print("Flaw:", cflaw.shape)
cv2.imshow("Flawed Image",  flaw)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(1, 1146):
    image = cv2.cvtColor(cv2.imread("Photo " + str(i) + ".jpg"), cv2.COLOR_BGR2GRAY)
    clone = []
    for y in range(0, len(image)):
        row = []
        for x in range(0, len(image[y])):
            row.append([int(z) for z in list(bin(image[y][x])[2:].zfill(8))])
        clone.append(row)
    samples.append(np.array(clone).flatten())
    print(i)
samples = np.array(samples)
print(samples.shape)
random.shuffle(samples)
samples = np.array(samples)
train, test = samples[:1000], samples[1000:]

# Initializing the RBM and training it
epochs = int(input("How many epochs would you like? "))
rbm = BernoulliRBM(learning_rate = 0.1, n_iter = epochs, n_components = 512)
rbm.fit(train)
print("done")

# Recreating image
recreate = rbm.gibbs(cflaw).reshape(79, 106, 8)
#recreate = rbm.gibbs(train[0]).reshape(79, 106, 8)
accuracy = list(recreate.flatten() - train[0])
print(str(round(100*(accuracy.count(0)/66992), 4 )) + "%")
image = []
for y in range(0, len(recreate)):
    col = []
    for x in range(0, len(recreate[y])):
        row = []
        for i in range(0, len(recreate[y][x])):
            if recreate[y][x][i]:
                row.append('1')
            else:
                row.append('0')
        col.append(int("".join(row), base=2))
    image.append(col)
image = np.array(image)
print(image.shape)
plt.imshow(image)
