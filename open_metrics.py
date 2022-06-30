import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

data_path = "/home/pedro/Desktop/new_AL_models/efficientNet_b1_lr5_NCI/history/"

val_loss_path = f"{data_path}efficientNet_b1_lr5_val_losses_0.5_low_entropy.npy"
val_metrics_path = f"{data_path}efficientNet_b1_lr5_val_metrics_0.5_low_entropy.npy"
train_loss_path = f"{data_path}efficientNet_b1_lr5_train_losses_0.5_low_entropy.npy"
train_metrics_path = f"{data_path}efficientNet_b1_lr5_train_metrics_0.5_low_entropy.npy"

val_losses = np.load(val_loss_path)
val_metrics = np.load(val_metrics_path)
train_losses = np.load(train_loss_path)
train_metrics = np.load(train_metrics_path)

trained_model_name = "NCI_effNet_lr5"
percentage = "0.5"
train_description = "20AL_100AUTO_low_1E7"

print("Accuracy: ", np.max(val_metrics[:,0]))
print("F1-Score: ", np.max(val_metrics[:,1]))
print("Recall: ", np.max(val_metrics[:,2]))
print("Precision: ", np.max(val_metrics[:,3]))
print("Min-Loss: ", np.min(val_losses))

#plt.figure(figsize=(10,5))
#plt.title(f"{train_description} Accuracy and Loss ({trained_model_name}_{percentage}%)")
#plt.plot(val_losses,label="val-loss", linestyle='--', color="green")
#plt.plot(train_losses,label="train-loss", color="green")
#plt.plot(val_metrics[:,0], label = "val-acc", linestyle='--',color="red")
#plt.plot(train_metrics[:,0], label="train-acc",color="red")
#plt.xlabel("Iterations")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
##plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_{train_description}_metrics_{percentage}p.png"))
#plt.show()


#plt.figure(figsize=(10,5))
#plt.title(f"{train_description} Recall, Precision,F1 ({trained_model_name}_{percentage}%)")
#plt.plot(val_metrics[:,1],label="val-recall", linestyle='--', color="green")
#plt.plot(train_metrics[:,1],label="train-recall", color="green")
#plt.plot(val_metrics[:,2], label = "val-precision", linestyle='--',color="red")
#plt.plot(train_metrics[:,2], label="train-precision",color="red")
#plt.plot(val_metrics[:,3], label = "val-f1", linestyle='--',color="blue")
#plt.plot(train_metrics[:,3], label="train-f1",color="blue")
#plt.xlabel("Iterations")
#plt.ylabel("Metrics")
#plt.legend()
##plt.savefig(os.path.join(trained_models_dir,f"{trained_model_name}_{train_description}_metrics2_{percentage}p.png"))
#plt.show()

