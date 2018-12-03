import numpy as np
import matplotlib.pyplot as plt

train_acc_path = "plots/test2/acc_history.txt"
val_acc_path = "plots/test2/acc_val_history.txt"
train_loss_path = "plots/test2/loss_val_history.txt"
val_loss_path = "plots/test2/loss_history.txt"

val_acc = np.loadtxt(val_acc_path, delimiter=',')
acc = np.loadtxt(train_acc_path, delimiter=',')
val_loss = np.loadtxt(val_loss_path, delimiter=',')
loss = np.loadtxt(train_loss_path, delimiter=',')

plt.plot(acc, 'b', label='Training acc')
plt.plot(val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(loss, 'b', label='Training loss')
plt.plot(val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()