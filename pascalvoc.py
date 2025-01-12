#!/usr/bin/env python3
"""
pascalvoc.py - Utilities to Manage Oxford PASCAL-VOC datasets


Author:
    https://github.com/vidurp/generic/blob/main/pascalvoc.py


"""

import re
import json
import os

ExamplePascalAnnotationString = '''# PASCAL Annotation Version 1.00

Image filename : "VOC2005_1/PNGImages/ETHZ_motorbike-testset/motorbikes005.png"
Image size (X x Y x C) : 640 x 480 x 3
Database : "The VOC2005 Dataset 1 Database (ETHZ)"
Objects with ground truth : 1 { "PASmotorbikeSide" }

# Note that there might be other objects in the image
# for which ground truth data has not been provided.

# Top left pixel co-ordinates : (1, 1)

# Details for object 1 ("PASmotorbikeSide")
Original label for object 1 "PASmotorbikeSide" : "motorbikeSide"
Bounding box for object 1 "PASmotorbikeSide" (Xmin, Ymin) - (Xmax, Ymax) : (206, 242) - (427, 365)
'''

def ParsePascalString( str ):
    Lines = str.split('\n')
    for line in Lines:
        if('filename' in line):
            line = line.split(':')
            FileName = re.findall(r'(?<=["\']).*?(?=["\'])', line[1])
        if('Original' in line):
            line = line.split(':')
            Label = re.findall(r'(?<=["\']).*?(?=["\'])', line[1])
        if('Bounding' in line):
            line = line.split(':')
            bbox = re.findall('\d+',line[1])

    Dict = {
        'FilePath' : FileName[0],
        'class' : Label[0],
        'xmin': bbox[0],
        'ymin': bbox[1],
        'xmax': bbox[2],
        'ymax': bbox[3]
    }
    return ( Dict )
  
def CreateJSONFromPascalDataSet( RootFilePath, JSONFileName ):
    """
    Creates a JSON File from a Tree Structure of PASCAL VOC image data
    PASCAL - Pattern Analysis, Statistical Modeling & Computational Learning
             built by University of Oxford. The root directory is expected in the
             following format
              VOC
              +->Annotations
                    +->Class1
                    +->Class2
              +->GTMasks
                    +->Class1
                    +->Class2
              +->PNGImages
                    +->Class1
                    +->Class2

    Args:
      RootFilePath - Dataset Root directory
      JSONFileName - JSON File to save
      
    Returns:
       void
    """
    # Write the data to a JSON file
    JsonData = {}
    Idx = 0
    with open(JSONFileName, "w") as outfile:
        for Root, Dirs, Files in os.walk( RootFilePath ):
            for File in Files:
                with open(Root + '/' + File, 'r') as TextFile:
                    Text = TextFile.read()
                    TextFile.close()
                    tokens = ParsePascalString( Text )
                    JsonData[ Idx ] = tokens
                    Idx = Idx + 1
        
        # convert dict to json
        json_string = json.dumps(JsonData)
        # write json  to disc
        outfile.write(json_string)
        outfile.close()

