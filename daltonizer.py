"""
Minecraft Texture Pack Daltonization
Adam Sykes
Jan 14, 2021
Version 1.2
This program is designed to convert a Minecraft Texture Pack into colourblind-friendly colours.
The matrix transformations are based off of Martin Krzywinski's research found at this link: https://mkweb.bcgsc.ca/colorblind/resources.mhtml,
as well as Joe Dietrich's https://github.com/joergdietrich/daltonize/blob/master/daltonize/daltonize.py
and Jim's blog at https://ixora.io/projects/colorblindness/color-blindness-simulation-research/.
This program now supports Protanopia, Deuteranopia, and Tritanopia.

To run this program, call 'daltonizer.py' from the command line. The program will prompt the user for the colour vision deficiency to correct
for, as well as the file path to the texture pack it needs to work on and the correction strength. 
"""
from types import SimpleNamespace
import threading
import os
import numpy
import sys
from PIL import Image
from PIL import ImageEnhance

def main():
    #main work thread function
    #Input: type of vision deficiency, the path to the pictures, and the correction strength
    #Output: None, pictures are changed in place
    cvdType = input("Protanopia, Deuteranopia, or Tritanopia? ")
    picPath = input("Path to pictures: ")
    strength = input("Strength of colour compensation(0-100, 100 = Full Strength): ")
    pictureList = getPictures(picPath)

    picCounter = SimpleNamespace()
    picCounter.n = 0

    #dynamically start a number of threads, each given a maximum of 20 pictures to work with
    #if less than 20 pictures, only start one thread
    threads = list()
    if(len(pictureList)//20 > 0):
        dividedList = numpy.array_split(pictureList,len(pictureList)//20)
        for i in range(len(pictureList)//20):
            x = threading.Thread(target=imageProcess,args=(cvdType,dividedList[i],picCounter,strength))
            threads.append(x)
            x.start()
    else:
        x = threading.Thread(target=imageProcess,args=(cvdType,pictureList,picCounter,strength))
        threads.append(x)
        x.start()
    prog = threading.Thread(target=progress,args=(picCounter,len(pictureList)),daemon=True)
    prog.start()

    for thread in threads:
        thread.join()
    prog.join()
    
    return None

#wrapper function for the thread to handle the progress bar
def progress(counter,length):
    while (counter.n != length):
        progBar(counter.n,length)

#Process the images given by picList according to the colourblindness given by blindType
def imageProcess(blindType,picList,picCounter,sigStrength):
    
    #Arrays for the various transformations needed. The exact values are calibrated from sigStrength using Interpolation
    protanTransform = numpy.array([(calcCorrect(1,100-int(sigStrength)),calcCorrect(1.05118294,int(sigStrength)),calcCorrect(-0.05116099,int(sigStrength))),(0,1,0),(0,0,1)])
    deuteranTransform = numpy.array([(1,0,0),(calcCorrect(0.9513092,int(sigStrength)),calcCorrect(1,100-int(sigStrength)),calcCorrect(0.04866992,int(sigStrength))),(0,0,1)])
    tritanTransform = numpy.array([(1,0,0),(0,1,0),(calcCorrect(-0.86744736,int(sigStrength)),calcCorrect(1.86727089,int(sigStrength)),calcCorrect(1,100-int(sigStrength)))])
    
    #array to manipulate color data based on the difference between normal and deficient vision
    compensatorArray = numpy.array([[calcCorrect(1,100-int(sigStrength)), 0, 0], [calcCorrect(0.7,sigStrength), 1, 0], [calcCorrect(0.7,sigStrength), 0, 1]])
    
    #arrays to covert between LMS and RGB color spaces
    LMSTransform = numpy.array([[0.0841456, 0.708538, 0.148692], [-0.0767272, 0.983854, 0.0817696], [-0.0192357, 0.152575, 0.876454]])
    RGBTransform = numpy.linalg.inv(LMSTransform)
    
    #Go through each pixel and apply the correct series of transformations to it.
    for image in picList:
        picCounter.n += 1
        if(image.endswith('.png',len(image)-4,len(image))):
            im = Image.open(image).convert('RGBA')
            width, height = im.size
            for x in range(width):
                for y in range(height):
                    pixel = im.getpixel((x,y))
                    linPixel = (linearizeV(pixel[0]),linearizeV(pixel[1]),linearizeV(pixel[2]))
                    npArray = numpy.asarray(linPixel)
                    lmsArray = numpy.matmul(npArray,LMSTransform)
                    if(blindType[0] == 'p' or blindType[0] == 'P'):
                        correctedArray = numpy.matmul(lmsArray,protanTransform)
                    if(blindType[0] == 'd' or blindType[0] == 'D'):
                        correctedArray = numpy.matmul(lmsArray,deuteranTransform)
                    if(blindType[0] == 't' or blindType[0] == 'T'):
                        correctedArray = numpy.matmul(lmsArray,tritanTransform)
                    rgbArray = numpy.matmul(correctedArray,RGBTransform)
                    difference = npArray - rgbArray
                    comp = numpy.matmul(difference,compensatorArray)
                    #compute = rgbArray
                    compute = npArray + comp
                    rgbCorrected = (delinearizeV(compute[0]),delinearizeV(compute[1]),delinearizeV(compute[2]))
                    pix = tuple(rgbCorrected)
                    im.putpixel((x,y),pix)
            im = im.save(image)

def getPictures(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getPictures(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# Linearize RGB values from 0-255 into 0-1, with gamma correction
def linearizeV(v):
    value = v/255
    if(value <= 0.04045):
        value /= 12.92
    else:
        value += 0.055
        value /= 1.055
        value = value**2.4
    return value

#Delinearize RGB values from 0-1 into 0-255, with gamma correction
def delinearizeV(v):
    value = v
    if(value <= 0.0031308):
        value *= 12.92
        value *= 255
    else:
        value = value ** (1/2.4)
        value *=1.055
        value -= 0.055
        value *= 255
    value = int(value)
    return value

#Display a progress bar on the command line
def progBar(current_val, end_val, bar_length=20):
    percent = float(current_val) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}% | {2}/{3} |".format(hashes + spaces, int(round(percent * 100)),current_val,end_val))
    sys.stdout.flush()

#wrapper for linear interpolation to change the processing strength based on user input value
def calcCorrect(number, strength):
    interp = numpy.interp(strength,[0,100],[0,number])
    return interp

if __name__ == "__main__":
    main()