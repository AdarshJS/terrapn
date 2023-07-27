import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from model_bn import get_model
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model


fd_name = "weights_bn6"

def get_data(data_lst, v_len=50):
    img_data = [] 
    vel_data = []
    labels = []

    with open(data_lst, "r") as r:
        print("loading data")
        for l in tqdm(r.readlines()):
            l = l.strip()
            pre =  "/".join(l.split("/")[:-1])
            n = l.split("/")[-1].split("_")[0]
            vel_dir = pre + "/" + "input_velocity_" + n + ".npy"
            label_dir = pre + "/" + "label_" + n + ".npy"
            i = Image.open(l)
            i = np.asarray(i)
            img_data.append(i)
            v = np.load(vel_dir)[-1 * v_len:]
            vel_data.append(v.reshape(-1))
            ll = np.load(label_dir)
            labels.append(ll)
    return np.array(img_data), np.array(vel_data), np.array(labels)



v_len = 50
patch_len = 100
seed = 0
img_data, vel_data, labels = get_data("/home/rayguan/Desktop/self_sup/dataset/patch100_lst.txt", v_len)
train = True
model_lst = "/home/gamma-robot/Downloads/self_sup/Code_Results/weights/Weights-190--0.21869.hdf5"

np.random.seed(seed)  
rand = np.arange(12740)
np.random.shuffle(rand)

img_train = img_data[rand[1274:]]
vel_train = vel_data[rand[1274:]]
labels_train = labels[rand[1274:]]
img_test = img_data[rand[:1274]]
vel_test = vel_data[rand[:1274]]
labels_test = labels[rand[:1274]]

# model = get_model(v_len, patch_len)
#checkpoint_name = './weights/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]


# model.fit([img_train, vel_train], labels_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

if train:
    model = get_model(v_len, patch_len)
    checkpoint_name = './' + fd_name + '/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    csv_logger = CSVLogger('./' + fd_name + '/training.log')
    callbacks_list = [checkpoint, csv_logger]
    model.fit([img_train, vel_train], labels_train, epochs=50, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
else:
    model = load_model(model_lst)

out = model.predict([img_test, vel_test])
np.save("./predict.npy", out)
np.save("./gt.npy", labels_test)
