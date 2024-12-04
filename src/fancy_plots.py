"""
Super cool fancy plots for confusion matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap
    sns.heatmap(df_cm, annot=annot, fmt='', cmap=cmap, cbar=False, ax=ax,annot_kws={"size": 18})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0,fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=18)
    # Customize the plot
    ax.set_ylabel('Predicted',rotation=90,fontsize=20)
    ax.set_xlabel('Actual',rotation=0,fontsize=20)
    plt.title('Confusion Matrix')

    # Add sum rows and columns
    sum_col = np.sum(cm, axis=0)
    sum_lin = np.sum(cm, axis=1)

    # Add sum row
    ax.add_patch(plt.Rectangle((0, cm.shape[0]), cm.shape[1], 1, fill=False, edgecolor='gray', lw=2))
    for j in range(cm.shape[1]):
        plt.text(j+0.5, cm.shape[0]+1, f"{sum_col[j]}\n{sum_col[j]/np.sum(sum_col)*100:.2f}%",
                 ha="center", va="center",fontsize=18)

    # Add sum column
    ax.add_patch(plt.Rectangle((cm.shape[1], 0), 1, cm.shape[0], fill=False, edgecolor='gray', lw=2))
    for i in range(cm.shape[0]):
        plt.text(cm.shape[1]+0.5, i+0.5, f"{sum_lin[i]}\n{sum_lin[i]/np.sum(sum_lin)*100:.2f}%",
                 ha="center", va="center",fontsize=18)

    # Add total sum
    ax.add_patch(plt.Rectangle((cm.shape[1], cm.shape[0]), 1, 1, fill=False, edgecolor='gray', lw=2))
    plt.text(cm.shape[1]+0.5, cm.shape[0]+0.5, f"{np.sum(sum_lin)}\n100%",
                ha="center", va="center",fontsize=18)

    plt.tight_layout()
    plt.show()
