#!/usr/bin/env python3
"""
pascalvoc.py - Utilities to Manage Oxford PASCAL-VOC datasets


Author:
    https://github.com/vidurp/generic/blob/main/pascalvoc.py


"""

import re
import json
import os

#Example Annotation Text File 
# This is a sample string with 2 objects, although not present in PASCAL-VOC 2005,
# we can support it now

ExamplePascalAnnotationString = '''# PASCAL Annotation Version 1.00

Image filename : "VOC2005_1/PNGImages/ETHZ_motorbike-testset/motorbikes005.png"
Image size (X x Y x C) : 640 x 480 x 3
Database : "The VOC2005 Dataset 1 Database (ETHZ)"
Objects with ground truth : 2 { "A", "B" }

# Note that there might be other objects in the image
# for which ground truth data has not been provided.

# Top left pixel co-ordinates : (1, 1)

# Details for object 1 ("A")
Original label for object 1 "A" : "Class_A"
Bounding box for object 1 "A" (Xmin, Ymin) - (Xmax, Ymax) : (206, 242) - (427, 365)

# Details for object 2 ("B")
Original label for object 2 "B" : "Class_B"
Bounding box for object 2 "B" (Xmin, Ymin) - (Xmax, Ymax) : (112, 112) - (333, 333)
'''

def ParsePascalString( str ):
    Lines = str.split('\n')
    ClassList = []
    BndBoxList = []
    for line in Lines:
        if 'filename' in line:
            line = line.split(':')
            FileName = re.findall(r'(?<=["\']).*?(?=["\'])', line[1])
        if 'Original' in line:
            line = line.split(':')
            ClassList.append(re.findall(r'(?<=["\']).*?(?=["\'])', line[1]))
        if 'Image size' in line:
            line = line.split(':')
            ImageSize  = re.findall('\d+',line[1])
        if 'Bounding' in line:
            line = line.split(':')
            BndBoxList.append(re.findall('\d+',line[1]))
        if 'Objects with ground truth' in line:
            line = line.split(':')
            NumObjects = re.findall(r'\d+', line[1])

    Dict = {
        'FilePath' : FileName[0],
        'numobjects' : NumObjects[0],
        'imagesize' : {
            'width': ImageSize[0],
            'height': ImageSize[1],
            'channels': ImageSize[2]
        },
        'object' : []
    }

    for Idx in range(int(NumObjects[0])):
        Obj = {
            'label' : ClassList[Idx][0],
            'bndbox' : {
                'xmin': BndBoxList[Idx][0],
                'ymin': BndBoxList[Idx][1],
                'xmax': BndBoxList[Idx][2],
                'ymax': BndBoxList[Idx][3]    
                }  
        }
        
        Dict['object'].append(Obj)
           
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
    JsonData = { 'Files' : [] }
    Idx = 0
    with open(JSONFileName, "w") as outfile:
        for Root, Dirs, Files in os.walk( RootFilePath ):
            for File in Files:
                with open(Root + '/' + File, 'r') as TextFile:
                    Text = TextFile.read()
                    TextFile.close()
                    tokens = ParsePascalString( Text )
                    JsonData[ 'Files' ].append( tokens )
                    Idx = Idx + 1
        
        # Add attribute number of images in set
        JsonData['NumImages'] = Idx
        
        # convert dict to json
        json_string = json.dumps(JsonData)
        # write json  to disc
        outfile.write(json_string)
        outfile.close()

