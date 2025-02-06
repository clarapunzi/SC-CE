"""
Super cool fancy plots for confusion matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(y_true, y_pred, class_labels=None, cmap='RdPu'):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to DataFrame for easier handling
    df_cm = pd.DataFrame(cm/cm.sum(1), index=class_labels, columns=class_labels)
    
    # Calculate percentages
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm, dtype=object)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c}\n{p:.2f}%'

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the heatmap
    sns.heatmap(df_cm, annot=annot, fmt='', cmap=cmap, cbar=False, ax=ax,annot_kws={"size": 18})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0,fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=18)
    # Customize the plot
    ax.set_ylabel('True Class',rotation=90,fontsize=20)
    ax.set_xlabel('Predicted',rotation=0,fontsize=20)
    plt.title('Confusion Matrix')

    # Add sum rows and columns
    sum_col = np.sum(cm, axis=0)
    sum_lin = np.sum(cm, axis=1)

    # Add sum row
    ax.add_patch(plt.Rectangle((0, cm.shape[0]), cm.shape[1], 1, fill=False, edgecolor='gray', lw=2))
    offset = 0.2
    for j in range(cm.shape[1]):
        plt.text(j+0.5, cm.shape[0]+offset*2, f"{sum_col[j]}\n{sum_col[j]/np.sum(sum_col)*100:.2f}%",
                 ha="center", va="center",fontsize=18)

    # Add sum column
    ax.add_patch(plt.Rectangle((cm.shape[1], 0), 1, cm.shape[0], fill=False, edgecolor='gray', lw=2))
    for i in range(cm.shape[0]):
        plt.text(cm.shape[1]+offset, i+0.5, f"{sum_lin[i]}\n{sum_lin[i]/np.sum(sum_lin)*100:.2f}%",
                 ha="center", va="center",fontsize=18)

    # Add total sum
    ax.add_patch(plt.Rectangle((cm.shape[1], cm.shape[0]), 1, 1, fill=False, edgecolor='gray', lw=2))
    plt.text(cm.shape[1]+offset, cm.shape[0]+offset, f"{np.sum(sum_lin)}\n100%",
                ha="center", va="center",fontsize=18)

    plt.tight_layout()
    plt.show()
def test_cmap (colorm):
    """
    Test the colormap
    """
    # create a colormap for the correlation
    # test the colormap
    plt.clf()
    plt.figure(figsize=(6,5))
    cmap = colorm
    random_data = np.random.rand(10000)*2-1
    colors = cmap(random_data)
    plt.scatter(np.arange(len(random_data)),random_data,c=random_data,cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar()
    plt.show()

# buid a colormap for the correlation that is whiteish for absolute values smaller than 0.7, and then goes fastly (not linearly) to the extremes
def build_cmap(t=0.7,c1="blue",c2="white",c3="red"):
    
    # Define colors: blue -> white -> red
    colors = [c1, c2,c2, c3]
    threshold = t
    # Define positions for colors (notice double white for sharp transition)
    positions = (np.array([-1, -threshold, threshold, 1])+1)/2
    
    return LinearSegmentedColormap.from_list('custom_corr', list(zip(positions, colors)))

gold_map  = build_cmap(0.7,'#3c6b5e', '#fff1d9','#deb062')
gold_map_r  = build_cmap(0.7,'#deb062', '#fff1d9','#3c6b5e')
#test_cmap(gold_map)

def plot_decision_space(my_model,model_name='',
                        draw_confidence=True,
                        X=None,y=None):
    """
        This funciton is used to plot the decision space of a model
    """
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))

    if draw_confidence:
        Z = np.max(my_model.predict_proba(np.c_[xx.ravel(), yy.ravel()]),axis=1)
    else:
        Z = my_model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.YlOrBr.reversed()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap,vmax=1.0,vmin=0.5,levels=np.linspace(0.5,1,11))
    plt.colorbar()
    # for the legend use the shaded color and the class labels: "predicted class i"
    #for i in range(0, len(set(y))):
    #    plt.scatter([], [], s=100, label="Class "+str(i),color=plt.cm.RdYlBu(i/len(set(y))),vmin=0,vmax=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black',vmin=0,vmax=1,alpha=0.2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    if len(model_name)>0:
        model_name = ": "+model_name+'(Best)'
    plt.title('Decision Space of model'+str(model_name))
    plt.show()