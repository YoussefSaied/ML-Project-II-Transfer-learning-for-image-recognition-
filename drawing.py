YoussefPicklingPath = '/home/youssef/EPFL/MA1/Machine learning/MLProject2/ML2/' # Path for pickling 
from sklearn import metrics
PicklingPath =YoussefPicklingPath
file_name= PicklingPath+"PredictionsAndLabelsModel1"
import pickle
with open(file_name, 'rb') as pickle_file:
    [predictions,labels] = pickle.load(pickle_file)
    pickle_file.close()

fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

# importing the required module 
import matplotlib.pyplot as plt 
# x axis and y axis values 
x ,y = fpr, tpr
plt.rcParams.update({'font.size': 15})

# plotting the points  
plt.plot(x, y,marker='x', label = "ROC") 
plt.plot(x, x,marker='x', label = "y=x" )

#plt.yscale('log')
plt.xscale('log')

# naming the x axis 
plt.xlabel('False Positive Rate') 
# naming the y axis 
plt.ylabel('True Positive Rate') 

# giving a title to my graph 
plt.title('Reciever operating characteristic curve') 
plt.legend()
# function to show the plot 
plt.show()

auc = metrics.roc_auc_score(labels, predictions)
print("AUROC: %5f"%auc)
