import matplotlib.pyplot as plt
import numpy as np
import zipfile

def TFPlotLossCurves( TrainingHistory, Epochs ):
    """
    Plots Training Curves returned from Tensorflow Fit() function
    
    Args:
        TrainingHistory: Training History Object returned from fit()
        
        Epochs: Training Epochs
        
    Returns:
        none
    """

    epochs = np.linspace( 1, Epochs, 1 )
    if 'val_loss' in TrainingHistory.history:
        ValDataPresent = True
    else:
        ValDataPresent = False
    
    plt.subplot(1,2,1)
    if ValDataPresent:

        plt.plot( epochs, TrainingHistory.history['loss'], label = 'loss', color = 'red' )
        plt.plot( epochs, TrainingHistory.history['val_loss'], label = 'val_loss', color = 'blue' )
        plt.axis( False )
        plt.legend( )
        plt.show( )
    else:
        plt.plot( epochs, TrainingHistory.history['loss'], label = 'loss', color = 'red' )
        plt.axis( False )
        plt.legend( )
        plt.show( )

    plt.subplot(1,2,2)
    if ValDataPresent:

        plt.plot( epochs, TrainingHistory.history['accuracy'], label = 'accuracy', color = 'red' )
        plt.plot( epochs, TrainingHistory.history['val_accuracy'], label = 'val_accuracy', color = 'blue' )
        plt.axis( False )
        plt.legend( )
        plt.show( )
    else:
        plt.plot( epochs, TrainingHistory.history['accuracy'], label = 'accuracy', color = 'red' )
        plt.axis( False )
        plt.legend( )
        plt.show( )


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
    