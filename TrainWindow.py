import itertools
import tkinter as tk
import cv2 as cv
import numpy as np
import pandas as pd
import os
import time
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from datetime import datetime
import tensorflow as tf
#from tensorflow.keras.models import load_model

#window
root = Tk()
root.title("Object Localization")

#localize window
localize = tk.Frame(root)

#label and train images for localization window
trainW = tk.Frame(root)

#show train window
trainW.pack(fill='both',expand=1)

#TRAIN WINDOW

global imageFolder
imageFolder = ""
global imagePaths
imagePaths = []
global images
images = []
global albumCv
albumCv = []
global albumImg
albumImg= []
global indexAlbum
indexAlbum = 0


left =tk.Frame(trainW)
left.grid(row=0,column=0)
right = tk.Frame(trainW)
right.grid(row=0,column=1)

#LEFT FRAME
# top panel - choose img folders and label folders
topPanel = tk.Label(left)
imagePath = tk.Entry(topPanel,font=('Helvetica', 12),width=30)
loadFolder = tk.Button(topPanel,text="Load Img Folder",font=('Helvetica', 12),command=lambda:loadFolder())

labelPath = tk.Entry(topPanel,font=('Helvetica', 12),width=30)
labelFolder = tk.Button(topPanel,text="Choose Label Folder",font=('Helvetica', 12),command=lambda:labelFolder())

topPanel.grid(row=0,column=0, padx=10, pady=10)
imagePath.grid(row=0,column=0, padx=10, pady=10)
loadFolder.grid(row=0,column=1, padx=10, pady=10)
labelPath.grid(row=0,column=2, padx=10, pady=10)
labelFolder.grid(row=0,column=3, padx=10, pady=10)

#bottom panel - show images
imgPanel = tk.Canvas(left, width=1125, height=750)
imgPanel.create_rectangle(0,0,1125,750,fill="white",width=1)
imgPanel.grid(row=1,column=0, padx=10, pady=10)

#RIGHT FRAME
sidePanel = tk.Frame(right)
label = tk.Button(sidePanel,text="Label",font=('Helvetica', 12),command=lambda:saveLabel())
nextPrevPanel = tk.Label(sidePanel)
nextButton = tk.Button(nextPrevPanel,text="Next",font=('Helvetica', 12),command=lambda:nextImg())
prevButton = tk.Button(nextPrevPanel,text="Previous",font=('Helvetica', 12),command=lambda:prevImg())
labelClass = tk.Label(sidePanel)
classEntry = tk.Entry(labelClass,font=('Helvetica', 12),width=10)
classWord = tk.Label(labelClass,text="Class")
classInfo = tk.Label(sidePanel)
numClasses = tk.Label(classInfo)
numClassEntry = tk.Entry(numClasses,font=('Helvetica', 12),width=10)
numClassWord = tk.Label(numClasses,text="Number of Classes")
typesClasse = tk.Label(classInfo)
typesClassEntry = tk.Entry(typesClasse,font=('Helvetica', 12),width=10)
typesClassWord = tk.Label(typesClasse,text="Classes")
train = tk.Button(sidePanel,text="Train",font=('Helvetica', 12),command=lambda:trainModel())
predict = tk.Button(sidePanel,text="Predict",font=('Helvetica', 12),command=lambda:switchLocalize())

label.grid(row=0,column=0, padx=10, pady=10)
nextButton.grid(row=0,column=0, padx=10, pady=10)
prevButton.grid(row=0,column=1, padx=10, pady=10)
nextPrevPanel.grid(row=1,column=0, padx=10, pady=10)

numClassEntry.grid(row=0,column=0, padx=10, pady=10)
numClassWord.grid(row=0,column=1, padx=10, pady=10)
typesClassEntry.grid(row=0,column=0, padx=10, pady=10)
typesClassWord.grid(row=0,column=1, padx=10, pady=10)
numClasses.grid(row=1,column=0, padx=10, pady=10)
typesClasse.grid(row=2,column=0, padx=10, pady=10)

classEntry.grid(row=0,column=0, padx=10, pady=10)
classWord.grid(row=0,column=1, padx=10, pady=10)
labelClass.grid(row=2,column=0, padx=10, pady=10)
classInfo.grid(row=3,column=0, padx=10, pady=10)
train.grid(row=4,column=0, padx=10, pady=10)
predict.grid(row=5,column=0, padx=10, pady=10)
sidePanel.grid(row=0,column=0, padx=10, pady=10)


def loadFolder():
    global imageFolder
    imageFolder = filedialog.askdirectory()
    print(imageFolder)
    loadImages(imageFolder)

def loadImages(imageFolder):
    global images
    images = []
    global justNames
    justNames = []
    for filename in os.listdir(imageFolder):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            justNames.append(filename)
            images.append(f"{imageFolder}/{filename}")
    if(len(images)>0):
        firstImg = Image.open(images[0])
        width,height = firstImg.size
        widthRatio = width/1125
        heightRatio = height/750
        imagePaths.clear()
        albumCv.clear()
        albumImg.clear()
        imagePaths.extend(images)
        imagePath.delete(0,END)
        imagePath.insert(0,imagePaths[0])
        for i in range(len(imagePaths)):
            imageAll = cv.imread(imagePaths[i])
            imageAll = cv.cvtColor(imageAll, cv.COLOR_BGR2RGB)
            imageCv = imageAll
            albumCv.append(imageCv)
            imageAll = arrToImg(imageAll,750,1125)
            albumImg.append(imageAll)
        imgPanel.create_image(0,0,anchor=tk.NW,image=albumImg[0])
        imgPanel.image = albumImg[0]
def nextImg():
     print("next")
     global albumImg, indexAlbum, images
     if(len(imagePaths)>0):
          indexAlbum += 1
          if(indexAlbum<len(images)):
                imgPanel.create_image(0,0,anchor=tk.NW,image=albumImg[indexAlbum])
                imgPanel.image = albumImg[indexAlbum]
                imagePath.delete(0,END)
                imagePath.insert(0,imagePaths[indexAlbum])
          else:
               indexAlbum -= 1
def prevImg():
        global albumImg, indexAlbum, images
        if(len(imagePaths)>0):
            indexAlbum -= 1
            if(indexAlbum>=0):
                imgPanel.create_image(0,0,anchor=tk.NW,image=albumImg[indexAlbum])
                imgPanel.image = albumImg[indexAlbum]
                imagePath.delete(0,END)
                imagePath.insert(0,imagePaths[indexAlbum])
            else:
                indexAlbum += 1
def startXY(event):
    global x1,y1
    x1 = event.x
    y1 = event.y
def drawRect(event):
     global x1,y1, xInterval , yInterval, readyLabel
     imgPanel.delete("rect")
     imgPanel.create_rectangle(x1,y1,event.x,event.y,outline="red",tag="rect")
     xInterval = abs(x1-event.x)
     yInterval = abs(y1-event.y)
     readyLabel = True
def saveLabel():
    global x1,y1, xInterval, yInterval, imagePaths, indexAlbum, labelFolder, widthRatio, heightRatio, readyLabel
    if(readyLabel):
        img = albumCv[indexAlbum]
        width,height,_ = img.shape
        print(width,height)
        widthRatio = width/1125
        heightRatio = height/750
        resizeWidthratio = 1024/width
        resizeHeightratio = 1024/height
        x1 = int(round(x1*widthRatio))
        x1 = int(round(x1*resizeWidthratio))
        y1 = int(round(y1*heightRatio))
        y1 = int(round(y1*resizeHeightratio))
        xInterval = int(round(xInterval*widthRatio))
        xInterval = int(round(xInterval*resizeWidthratio))
        yInterval = int(round(yInterval*heightRatio))
        yInterval = int(round(yInterval*resizeHeightratio))
        centerX  = x1 + xInterval/2
        centerY = y1 + yInterval/2
        normalizedX = centerX/1024
        normalizedY = centerY/1024
        normalizedWidth = xInterval/1024
        normalizedHeight = yInterval/1024
        labelFile = labelFolder+"/"+justNames[indexAlbum].split(".")[0]+".txt"
        mode = "a" if os.path.exists(labelFile) else "w"
        classLabel = classEntry.get()
        with open(labelFile,mode) as label:
            #label.write(str(classLabel)+" "+str(normalizedX)+" "+str(normalizedY)+" "+str(normalizedWidth)+" "+str(normalizedHeight)+"\n")
             label.write(str(classLabel)+" "+str(centerX)+" "+str(centerY)+" "+str(xInterval)+" "+str(yInterval)+"\n")
        readyLabel = False
    

def labelFolder():
    global labelFolder
    labelFolder = filedialog.askdirectory()
    labelPath.delete(0,END)
    labelPath.insert(0,labelFolder)
def arrToImg(arr, newH, newW): #helper function converts array of pixels to image
	global width,height
	arr = Image.fromarray(arr)
	if(newH!=0 and newW!=0):
		arr=arr.resize((newW,newH),Image.LANCZOS)
	arr = ImageTk.PhotoImage(arr)
	return arr
def switchLocalize():
    #train.forget
    print("Switching to localize")
    localize.pack(fill='both',expand=1)
    trainW.forget()
def trainModel():
    global imageFolder, labelFolder, numClassEntry, typesClassEntry
    numClasses = numClassEntry.get()
    typesClasses = typesClassEntry.get()
    print(numClasses,typesClasses)
    #Resize images to 1024x1024 to new folder
    resizedFolder = f"{imageFolder}/resized"
    os.mkdir(resizedFolder)
    for i in range(len(imagePaths)):
        img = cv.imread(imagePaths[i])
        img = cv.resize(img,(1024,1024))
        cv.imwrite(f"{imageFolder}/resized/{justNames[i]}",img)
    print(resizedFolder)
    print(labelFolder)
    os.system(f"python3 train.py --dataset {resizedFolder} --annot {labelFolder} --classes {numClasses} --labels {typesClasses}")
    print("Training model")

imgPanel.bind("<Button-1>",startXY)
imgPanel.bind("<B1-Motion>",drawRect)
# Train window end

# LOCALIZE WINDOW

global imageFolderL
imageFolderL = ""
global imagesL
imagesL = []
global justNamesL
justNamesL = []
global predictions
predictions = []
global albumCvL
albumCvL = []
global albumImgL
albumImgL= []
global indexAlbumL
indexAlbumL = 0



leftL =tk.Frame(localize)
leftL.grid(row=0,column=0)
rightL = tk.Frame(localize)
rightL.grid(row=0,column=1)

#LEFT FRAME
# top panel - choose img folders and label folders
topPanelL = tk.Label(leftL)
imagePathL = tk.Entry(topPanelL,font=('Helvetica', 12),width=30)
loadFolderL = tk.Button(topPanelL,text="Load Img Folder",font=('Helvetica', 12),command=lambda:loadPFolder())


topPanelL.grid(row=0,column=0, padx=10, pady=10)
imagePathL.grid(row=0,column=0, padx=10, pady=10)
loadFolderL.grid(row=0,column=1, padx=10, pady=10)

#bottom panel - show images
imgPanelL = tk.Canvas(leftL, width=1125, height=750)
imgPanelL.create_rectangle(0,0,1125,750,fill="white",width=1)
imgPanelL.grid(row=1,column=0, padx=10, pady=10)
#RIGHT FRAME
sidePanelL = tk.Label(rightL)
nextPrevPanelL = tk.Label(sidePanelL)
nextButtonL = tk.Button(nextPrevPanelL,text="Next",font=('Helvetica', 12),command=lambda:nextImgL())
prevButtonL = tk.Button(nextPrevPanelL,text="Previous",font=('Helvetica', 12),command=lambda:prevImgL())
nextButtonL.grid(row=0,column=0, padx=10, pady=10)
prevButtonL.grid(row=0,column=1, padx=10, pady=10)
nextPrevPanelL.grid(row=1,column=0, padx=10, pady=10)
sidePanelL.grid(row=0,column=0, padx=10, pady=10)
#model = load_model('trained_model.h5')

def custom_loss(y_true, y_pred):
    # Split the predictions into class probabilities and bounding box coordinates
    y_true_class = y_true[:, :5]
    y_true_bbox = y_true[:, 5:]

    y_pred_class = y_pred[:, :5]
    y_pred_bbox = y_pred[:, 5:]

    # Classification loss
    class_loss = categorical_crossentropy(y_true_class, y_pred_class)

    # Localization loss (smooth L1 loss)
    diff = tf.abs(y_true_bbox - y_pred_bbox)
    loc_loss = tf.where(diff < 1, 0.5 * diff ** 2, diff - 0.5)
    loc_loss = tf.reduce_sum(loc_loss, axis=1)
    # Weighted sum of classification and localization losses
    total_loss = class_loss + loc_loss

    return total_loss

with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
    model = tf.keras.models.load_model('trained_model.h5')

def loadPFolder():
    global imageFolderl
    imageFolderL = filedialog.askdirectory()
    print(imageFolderL)
    global imagesL
    imagesL = []
    global justNamesL
    justNamesL = []
    global predictions
    predictions = []
    global albumCvL
    albumCvL = []
    global albumImgL
    albumImgL= []
    global indexAlbumL
    indexAlbumL = 0
    for filename in os.listdir(imageFolderL):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            justNamesL.append(filename)
            imagesL.append(f"{imageFolderL}/{filename}")
    if(len(imagesL)>0):
        firstImg = Image.open(imagesL[0])
        width,height = firstImg.size
        widthRatio = width/1125
        heightRatio = height/750
        imagePaths.clear()
        albumCvL.clear()
        albumImgL.clear()
        imagePaths.extend(imagesL)
        imagePathL.delete(0,END)
        imagePathL.insert(0,imagePaths[0])
        for i in range(len(imagesL)):
            prediction = predictImg(imagesL[i])
            predictions.append(prediction)
            img = cv.imread(imagesL[i])
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = arrToImg(img,750,1125)
            albumImgL.append(img)
            albumCvL.append(cv.imread(imagesL[i]))
        imgPanelL.create_image(0,0,anchor=tk.NW,image=albumImgL[0])
        imgPanelL.image = albumImgL[0]
        imgPanelL.create_rectangle(predictions[0][0],predictions[0][1],predictions[0][2],predictions[0][3],outline="red",width=2)
def nextImgL():
        print("next")
        global albumImgL, indexAlbumL, imagesL
        if(len(imagePaths)>0):
            indexAlbumL += 1
            if(indexAlbumL<len(imagesL)):
                    imgPanelL.create_image(0,0,anchor=tk.NW,image=albumImgL[indexAlbumL])
                    imgPanelL.image = albumImgL[indexAlbumL]
                    imagePathL.delete(0,END)
                    imagePathL.insert(0,imagePaths[indexAlbumL])
                    imgPanelL.create_rectangle(predictions[indexAlbumL][0],predictions[indexAlbumL][1],predictions[indexAlbumL][2],predictions[indexAlbumL][3],outline="red",width=2)
            else:
                indexAlbumL -= 1
def prevImgL():
        global albumImgL, indexAlbumL, imagesL
        if(len(imagePaths)>0):
            indexAlbumL -= 1
            if(indexAlbumL>=0):
                imgPanelL.create_image(0,0,anchor=tk.NW,image=albumImgL[indexAlbumL])
                imgPanelL.image = albumImgL[indexAlbumL]
                imagePathL.delete(0,END)
                imagePathL.insert(0,imagePaths[indexAlbumL])
                imgPanelL.create_rectangle(predictions[indexAlbumL][0],predictions[indexAlbumL][1],predictions[indexAlbumL][2],predictions[indexAlbumL][3],outline="red",width=2)
            else:
                indexAlbumL += 1
def predictImg(imagePath):
    img = cv.imread(imagePath)
    width,height,_ = img.shape
    preprocessedImg = preprocess_image(imagePath)
    prediction = model.predict(preprocessedImg)
    widthRatio = 1125/1024
    heightRatio = 750/1024
    print(prediction)
    centerX = prediction[0][1]*widthRatio
    centerY = prediction[0][2]*heightRatio
    xInterval = prediction[0][3]*widthRatio
    yInterval = prediction[0][4]*heightRatio
    x1 = int(round(centerX - xInterval/2))
    y1 = int(round(centerY - yInterval/2))
    x2 = int(round(centerX + xInterval/2))
    y2 = int(round(centerY + yInterval/2))
    print(x1,y1,x2,y2)
    return [x1,y1,x2,y2]
def preprocess_image(imagePath):
    image = cv.imread(imagePath)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (1024, 1024))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
root.mainloop()