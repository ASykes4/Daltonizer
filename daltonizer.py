"""
Image Daltonization
Adam Sykes
September 22, 2026
Version 1.3
This program is designed to convert images into colourblind-friendly colours.
The original matrix transformations are based on: 
    https://mkweb.bcgsc.ca/colorblind/resources.mhtml 
    https://github.com/joergdietrich/daltonize 
    https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

This program supports Protanopia, Deuteranopia, and Tritanopia.

To run this program, call 'daltonizer.py' from the command line. The program will prompt the user for the colour vision deficiency to correct
for, as well as the file path to the folder of images it needs to work on and the correction strength. 
"""
from types import SimpleNamespace
import threading
import os
import numpy as np
import sys
from PIL import Image

def main(): 
    """ 
    Main command-line entry point. 
    """ 
    cvdType = input("Protanopia, Deuteranopia, or Tritanopia? ") 
    picPath = input("Path to pictures: ") 
    strength = input("Strength of colour compensation (0-100, 100 = Full Strength): ")

    try: 
        strength = float(strength) 
    except ValueError: 
        print("Strength must be a number from 0 to 100.") 
        return 
    
    strength = max(0.0, min(100.0, strength)) 

    pictureList = getPictures(picPath) 
    if not pictureList: 
        print("No PNG files were found.") 
        return 
    
    picCounter = SimpleNamespace() 
    picCounter.n = 0 

    # Distribute image files across several worker threads. 
    threadCount = max(1, (len(pictureList) + 19) // 20) 
    threadCount = min(threadCount, len(pictureList)) 
    dividedList = np.array_split( pictureList, threadCount)

    threads = [] 
    for fileList in dividedList: 
        thread = threading.Thread(target=imageProcess, args=(cvdType, fileList.tolist(), picCounter, strength)) 
        threads.append(thread) 
        thread.start() 

    prog = threading.Thread(target=progress, args=(picCounter, len(pictureList)), daemon=True) 
    prog.start() 

    for thread in threads: 
        thread.join() 
        prog.join()

# Process the images given by picList according to the colourblindness given by blindType
# Input: blindType - String, first letter matching one of p, d, or t
# Input: picList - list of Strings, where the strings are directory paths to a png image
# Input: picCounter - SimpleNamespace, to keep track of overall progress
# Input: sigStrength - int, the % of correction to apply
# Output: None, images are edited in-place
def imageProcess(blindType, picList, picCounter, sigStrength): 
    """ 
    Process a list of PNG images. 
    Each image is processed as a complete NumPy array. 
    """ 

    for imagePath in picList: 
        try: 
            with Image.open(imagePath) as source: 
                im = source.convert("RGBA") 
                corrected = daltonize_image(im, blindType, sigStrength) 

                corrected.save(imagePath) 

        except Exception as exc: 
            print( f"\nError processing '{imagePath}': {exc}", file=sys.stderr) 

        finally: picCounter.n += 1

# Recursively search a given directory for PNG files
# Input: dirname - String representation of the folder path that contains the files
# Output: list of Strings, each one a full directory path to a PNG image
def getPictures(dirName): 
    """ 
    Recursively find PNG files in dirName. 
    """ 

    allFiles = [] 
    for root, _, files in os.walk(dirName): 
        for filename in files: 
            if filename.lower().endswith(".png"): 
                allFiles.append(os.path.join(root, filename))

    return allFiles

# Create the CVD simulation, compensation, and colour space transformation matrices
# Input: blindType - String, first letter should match either p, d, or t
# Input: sigStrength - int, % indicating how much colour correction to apply
# Output: tuple containing 4 numpy ndarrays, in the order of LMS transform, cvd simulation, RGB transform, compensator transform
def get_transformation_matrices(blindType, sigStrength): 
    """ 
    Construct the CVD simulation and compensation matrices.  
    """ 

    strength = float(sigStrength) 

    # Matrices for simulating the various colour vision deficiencies. 
    # These are applied in LMS colour space. 
    protanTransform = np.array([(calcCorrect(1.0, 100.0 - strength), calcCorrect(1.05118294, strength), calcCorrect(-0.05116099, strength)), 
                                (0.0, 1.0, 0.0), 
                                (0.0, 0.0, 1.0) ], 
                                dtype=np.float32) 
    
    deuteranTransform = np.array([(1.0, 0.0, 0.0), 
                                  (calcCorrect(0.9513092, strength), calcCorrect(1.0, 100.0 - strength), calcCorrect(0.04866992, strength)), 
                                  (0.0, 0.0, 1.0) ], 
                                  dtype=np.float32) 
    
    tritanTransform = np.array([(1.0, 0.0, 0.0), 
                                (0.0, 1.0, 0.0), 
                                (calcCorrect(-0.86744736, strength), calcCorrect(1.86727089, strength), calcCorrect(1.0, 100.0 - strength))], 
                                dtype=np.float32) 

    # Matrix used to redistribute the lost colour information. 
    compensatorArray = np.array([[calcCorrect(1.0, 100.0 - strength), 0.0, 0.0 ],
                                 [calcCorrect(0.7, strength), 1.0, 0.0 ], 
                                 [calcCorrect(0.7, strength), 0.0, 1.0 ]], 
                                 dtype=np.float32) 
    
    # RGB -> LMS transformation. 
    LMSTransform = np.array([[0.0841456, 0.708538, 0.148692], 
                             [-0.0767272, 0.983854, 0.0817696], 
                             [-0.0192357, 0.152575, 0.876454]], 
                             dtype=np.float32) 
    
    # LMS -> RGB transformation. 
    RGBTransform = np.linalg.inv(LMSTransform).astype(np.float32) 

    blind = blindType.strip().lower() 
    if blind.startswith("p"): 
        simulation = protanTransform 
    elif blind.startswith("d"): 
        simulation = deuteranTransform 
    elif blind.startswith("t"): 
        simulation = tritanTransform 

    else: raise ValueError( "Colour vision deficiency must be " "Protanopia, Deuteranopia, or Tritanopia." ) 

    return (LMSTransform, simulation, RGBTransform, compensatorArray)

def daltonize_image(image, blindType, strength): 
    """ 
    Apply the original Daltonizer algorithm to an entire image at once. 
    Parameters 
    ---------- 
    image: 
        A PIL Image in RGB or RGBA format. 
    blindType: 
        Protanopia, Deuteranopia, or Tritanopia. 
    strength: 
        Correction strength from 0 to 100. 
        
    Returns 
    ------- 
    PIL.Image Corrected image with the original alpha channel preserved. 
    """ 
    if image.mode != "RGBA": 
        image = image.convert("RGBA") 

    # Convert PIL image to NumPy. 
    # 
    # Shape: 
    #   height x width x 4 
    # 
    # The first three channels are RGB and the fourth is alpha. 
    pixels = np.asarray(image) 
    rgb = pixels[:, :, :3] 
    alpha = pixels[:, :, 3] 

    # Construct matrices once for the entire image. 
    (LMSTransform, 
     simulationTransform, 
     RGBTransform, 
     compensatorArray) = get_transformation_matrices(blindType, strength)

    # Convert sRGB -> linear RGB 
    linearRGB = linearize(rgb) 

    # Convert RGB -> LMS 
    # Vectorized; every pixel is multiplied simultaneously.
    lms = linearRGB @ LMSTransform 

    # Simulate the selected colour vision deficiency
    simulatedLMS = lms @ simulationTransform 

    # Convert simulated LMS -> RGB 
    simulatedRGB = simulatedLMS @ RGBTransform 

    # Determine the colour information lost by the simulation 
    difference = linearRGB - simulatedRGB 

    # Redistribute the lost colour information 
    compensation = difference @ compensatorArray 

    # Apply the compensation 
    correctedRGB = linearRGB + compensation 

    # Convert linear RGB -> sRGB 
    # This also clips values outside [0, 1]
    correctedRGB = delinearize(correctedRGB) 

    # Reattach the original alpha channel
    output = np.empty_like(pixels) 
    output[:, :, :3] = correctedRGB 
    output[:, :, 3] = alpha 

    return Image.fromarray(output, mode="RGBA")

# Convert an sRGB image from uint8 [0, 255] to linear RGB [0, 1]
# Input: image - numpy array
# Output: numpy array of same shape as input, with values linearized between 0 - 1
def linearize(image): 
    """ 
    Convert an sRGB image from uint8 [0, 255] to linear RGB [0, 1]. 
    Works on an entire NumPy image array at once. 
    """ 

    value = image.astype(np.float32) / 255.0 
    return np.where(value <= 0.04045, 
                    value / 12.92, 
                    ((value + 0.055) / 1.055) ** 2.4)

# Delinearize RGB values from 0-1 into 0-255, with gamma correction
# Input: image - numpy array 
# Output: numpy array of same shape as input, with values delinearized to between 0 - 255
def delinearize(image):
    """ 
    Convert linear RGB [0, 1] to sRGB uint8 [0, 255]. 
    Values outside the valid RGB range are clipped before conversion. 
    """ 

    value = np.clip(image, 0.0, 1.0) 
    value = np.where(value <= 0.0031308, 
                     value * 12.92, 
                     1.055 * (value ** (1.0 / 2.4)) - 0.055)
    
    return np.clip(np.round(value * 255.0), 0, 255 ).astype(np.uint8)

# Wrapper function for the thread to handle the progress bar
# Input: counter - SimpleNamespace, counting how many units of work done as n
# Input length - integer, the total amount of work to be done
# Output: None
def progress(counter, length): 
    """ 
    Wrapper for the progress display thread. 
    """ 

    while counter.n < length: 
        progBar(counter.n, length) 

    progBar(length, length) 
    print()

# Display a progress bar on the command line composed of # and spaces
# Input: current_val - integer, amount of work done
# Input: end_val - integer, total amount of work to do
# Input; bar_length - integer, how long the progress bar should be, in characters
# Output: None
def progBar(current_val, end_val, bar_length=20): 
    """ 
    Display a progress bar on the command line. 
    """ 
    if end_val == 0: 
        percent = 1.0 
    else: 
        percent = float(current_val) / end_val 

    hashes = "#" * int(round(percent * bar_length)) 
    spaces = " " * (bar_length - len(hashes)) 

    sys.stdout.write("\rPercent: [{0}] {1}% | {2}/{3} |".format( hashes + spaces, int(round(percent * 100)), current_val, end_val)) 
    sys.stdout.flush()

# Wrapper for linear interpolation to change the processing strength based on user input value
# Input: number - an arbitrary number that can cast to float, acts as '100%'
# Input: strength - a percentage that can cast to float, acts as % of number
# Output: float that is strength% of number
def calcCorrect(number, strength):
    """ 
    Interpolate a value from 0 at 0% strength to `number` at 100%. 
    """

    return np.interp(strength, [0, 100], [0, number])

if __name__ == "__main__":
    main()