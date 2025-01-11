#!/usr/bin/env python3
"""
tfhelpers.py - AI/ML utility functions, especially to be used with 
               Tensorflow


Author:
    https://github.com/vidurp/generic/blob/main/tfhelpers.py

Date:
"""


import matplotlib.pyplot as plt
import numpy as np
import zipfile
import itertools
from sklearn.metrics import confusion_matrix


def TFPlotLossCurves( TrainingHistory, Epochs = 0, FigSize=(8,4) ):
    """
    Plots Training Curves returned from Tensorflow Fit() function
    
    Args:
        TrainingHistory: Training History Object returned from fit()

        Epochs: Training Epochs ( if zero, uses epoch list from TrainingHistory )

        FigSize: figure size , default (8,4)
        
    Returns:
        none
    """
      
    plt.figure(figsize=FigSize)
    if 'val_loss' in TrainingHistory.history:
        ValDataPresent = True
    else:
        ValDataPresent = False
    
    if Epochs == 0:
        epochs = np.array(TrainingHistory.epoch)
    else:
        epochs = np.linspace(1,Epochs,Epochs)
    
    plt.subplot(1,2,1)
    if ValDataPresent:

        plt.plot( epochs, TrainingHistory.history['loss'], label = 'loss', color = 'red' )
        plt.plot( epochs, TrainingHistory.history['val_loss'], label = 'val_loss', color = 'blue' )
    else:
        plt.plot( epochs, TrainingHistory.history['loss'], label = 'loss', color = 'red' )
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1,2,2)
    if ValDataPresent:

        plt.plot( epochs, TrainingHistory.history['accuracy'], label = 'accuracy', color = 'red' )
        plt.plot( epochs, TrainingHistory.history['val_accuracy'], label = 'val_accuracy', color = 'blue' )
    else:
        plt.plot( epochs, TrainingHistory.history['accuracy'], label = 'accuracy', color = 'red' )
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.legend( )
    plt.show()


def UnZipFiles( ZipFilePath, OutputPath = '.' ):
    """
    Unzip a file to target directory
    
    Args:
        ZipFilePath: ZIP File to decompress
        
        OutputPath: target directory, default to cwd
        
    Returns:
        none
    """
    with zipfile.ZipFile(ZipFilePath,'r') as File:
        File.extractall( OutputPath )
    
    
# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# Our function needs a different name to sklearn's plot_confusion_matrix
def PlotConfusionMatrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    PlotConfusionMatrix( y_true=test_labels, # ground truth test labels
                         y_pred=y_preds, # predicted labels
                        classes=class_names, # array of class label names
                        figsize=(15, 15),
                        text_size=10 )
  """  
  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  ax.tick_params(axis='x',rotation=90)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")