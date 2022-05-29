import copy
import numpy as np
import matplotlib
import PIL
from sklearn import svm
import seaborn as sns
from sklearn.metrics import confusion_matrix

f = open("train.txt")
f.readline()
i = 0
model = svm.SVC(C = 5)
train_images = []
train_labels = []
for line in f:
    img_name, img_label = line.split(',')
    train_img = PIL.Image.open("train+validation/{0}".format(img_name))
    train_img = copy.deepcopy(np.asarray(train_img))

    n0, n1, n2 = train_img.shape
    train_img = train_img.reshape((n0, n1 * n2))

    train_images.append(train_img)
    train_labels.append(img_label)

images = np.array(train_images)
labels = np.array(train_labels)

n0, n1, n2 = images.shape
images = images.reshape((n0, n1 * n2))

model.fit(images, labels)

f.close()
f = open("validation.txt")
f.readline()
i = 0

validation_images = []
validation_labels = []
for line in f:
    img_name, img_label = line.split(',')
    validation_img = PIL.Image.open("train+validation/{0}".format(img_name))
    validation_img = copy.deepcopy(np.asarray(validation_img))

    n0, n1, n2 = validation_img.shape
    validation_img = validation_img.reshape((n0, n1 * n2))


    validation_images.append(validation_img)
    validation_labels.append(img_label)


validation_images = np.array(validation_images)
n0, n1, n2 = validation_images.shape
validation_images = validation_images.reshape((n0, n1 * n2))

rez = model.predict(validation_images)
print(np.mean(rez == validation_labels))

cf_matrix = confusion_matrix(validation_labels, rez)
sns.heatmap(cf_matrix, annot=True)
print(cf_matrix)

f.close()
f = open("test.txt")
g = open("prediction.txt", "w")
f.readline()
i = 0

g.write("id,label\n")
predictions = []

for img_name in f:
    test_img = PIL.Image.open("test/{0}".format(img_name[:-1]))
    test_img = copy.deepcopy(np.asarray(test_img))

    n0, n1, n2 = test_img.shape
    test_img = test_img.reshape((n0, n1 * n2))


    predictions.append(test_img)


predictions = np.array(predictions)
n0, n1, n2 = predictions.shape
predictions = predictions.reshape((n0, n1 * n2))

rez = model.predict(predictions)
f.close()
f = open("test.txt")
f.readline()

i = 0
for img_name in f:
    g.write(img_name[:-1] + "," + rez[i])
    i = i + 1