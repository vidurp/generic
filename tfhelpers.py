import matplotlib.pyplot as plt
import numpy as np
import zipfile

def TFPlotLossCurves( TrainingHistory, Epochs, FigSize=(8,4) ):
    """
    Plots Training Curves returned from Tensorflow Fit() function
    
    Args:
        TrainingHistory: Training History Object returned from fit()

        Epochs: Training Epochs

        FigSize: figure size , default (8,4)
        
    Returns:
        none
    """
    plt.figure(figsize=FigSize)
    epochs = np.linspace( 1, Epochs, Epochs )
    if 'val_loss' in TrainingHistory.history:
        ValDataPresent = True
    else:
        ValDataPresent = False
    
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
    