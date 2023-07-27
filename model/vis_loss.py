import numpy as np 

from tqdm import tqdm
import matplotlib.pyplot as plt

epoch = []
train_loss = []
val_loss = []

with open("./weights_bn6/training.log", "r") as fr: # or "./weights_bn6_test/training.log

    for l in tqdm(fr.readlines()[1:]):
        arr = l.strip().split(",")
        epoch.append(int(arr[0]))
        train_loss.append(float(arr[1]))
        val_loss.append(float(arr[3]))

# from IPython import embed;embed()
# plt.plot(epoch, train_loss, 'o')



epoch_512 = []
train_loss_512 = []
val_loss_512 = []

with open("./weights_bn7/training.log", "r") as fr:

    for l in tqdm(fr.readlines()[1:]):
        arr = l.strip().split(",")
        epoch_512.append(int(arr[0]))
        train_loss_512.append(float(arr[1]))
        val_loss_512.append(float(arr[3]))




epoch_img_only = []
train_loss_img_only = []
val_loss_img_only = []

with open("./weights_img_only/training.log", "r") as fr:

    for l in tqdm(fr.readlines()[1:]):
        arr = l.strip().split(",")
        epoch_img_only.append(int(arr[0]))
        train_loss_img_only.append(float(arr[1]))
        val_loss_img_only.append(float(arr[3]))


epoch_base = []
train_loss_base = []
val_loss_base = []

with open("./weights_bn_test/training.log", "r") as fr:

    for l in tqdm(fr.readlines()[1:]):
        arr = l.strip().split(",")
        epoch_base.append(int(arr[0]))
        train_loss_base.append(float(arr[1]))
        val_loss_base.append(float(arr[3]))

plt.plot(epoch_img_only, val_loss_img_only)
plt.plot(epoch_base, val_loss_base)
plt.plot(epoch_512, val_loss_512)
plt.plot(epoch, val_loss)
plt.axis([0, 50, 0.35, 0.8])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(['img_only', 'img+velocity baseline', 'ours_512', 'ours_128'])
plt.show()

