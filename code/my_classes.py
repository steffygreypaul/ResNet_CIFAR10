import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow import keras
import tensorflow.keras.backend as K


class SaveModel(keras.callbacks.Callback):
    
    def __init__(self, outputPath,every=5,start_at=0):
        self.outputPath=outputPath     
        self.every = every
        self.num_Epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) % self.every == 0:
            print("\n saving model weights")
            p = os.path.sep.join([self.outputPath,"epoch_{}.hdf5".format(self.num_Epoch + 1)])
            self.model.save(p, overwrite=True)
        self.num_Epoch+=1


#############################################################################################
class MyPlot(keras.callbacks.Callback):
    def __init__(self,figPath,dataPath=None, start_at=0):
        self.figPath = figPath
        self.dataPath=dataPath
        self.start_at = start_at 
    def on_train_begin(self,logs={}):
        self.data={}
        if self.dataPath is not None:
            if os.path.exists(self.dataPath):
                with open(self.dataPath, "rb") as myFile:
                    self.data = pickle.load(myFile)
                    myFile.close() 
                if self.start_at > 0:
		            # loop over the entries in the history log and
		            # trim any entries that are past the starting epoch
                    for k in self.data.keys():
                        self.data[k] = self.data[k][:self.start_at]
    def on_epoch_end(self, epoch, logs={}):
        for key,value in logs.items():
          if np.logical_and(epoch == 0, self.start_at==0):
            self.data[key]=[value]
          else:
            self.data[key].append(value)
        if self.dataPath is not None:
            with open(self.dataPath, "wb") as myFile:
                pickle.dump(self.data, myFile)
                myFile.close()
        if len(self.data["loss"]) > 1:
          N = np.arange(0, len(self.data["loss"]))
          plt.style.use("ggplot")
          plt.figure()
          plt.plot(N, self.data["loss"], label="train_loss")
          plt.plot(N, self.data["val_loss"], label="val_loss")
          plt.title("Training Loss [Epoch {}]".format(len(self.data["loss"])))
          plt.xlabel("Epoch #")
          plt.ylabel("Loss")
          plt.legend()
          plt.savefig(self.figPath+"_loss.png")
          plt.close()
          
          plt.figure()
          plt.plot(N, self.data["accuracy"], label="train_acc")
          plt.plot(N, self.data["val_accuracy"], label="val_acc")
          plt.title("Training Accuracy [Epoch {}]".format(len(self.data["loss"])))
          plt.xlabel("Epoch #")
          plt.ylabel("Accuracy")
          plt.legend()
          plt.savefig(self.figPath+"_accuracy.png")
          plt.close()
#############################################################################################

class LearningRateDecay(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        if  (epoch%35==0) and (epoch !=0):
            print("[INFO] old learning rate value: {}".format(K.get_value(self.model.optimizer.lr)))
            K.set_value(self.model.optimizer.lr, (K.get_value(self.model.optimizer.lr))/2 )
            print("[INFO] new learning rate value: {}".format(K.get_value(self.model.optimizer.lr)))


            
