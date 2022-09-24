#import os.path as pth
import cv2 
import os.path
#from os.path import join as pjoin
import shutil, os
import matplotlib.pyplot as plt
#import glob,os,re
import time 
#import argparse
#import imutils
import msvcrt
#import subprocess
#import sys
#import test2.py





PIC_FOLDER = 'CNN'
IMG_SUFFIX = 'test.png'

def menu():
    while True:
        print("que voulez vous faire ")
        print("1 - faire une capture d'écran")
        #print("2- comparer les images")
        print("2 - Quitter le menu")
        choix = int(input())
        
        if choix == 1:
            main()
        if choix == 2:
            print("vous avez quitté le menu")
            break
        
        
    

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)


def kbfunc():
    x = msvcrt.kbhit()
    if x == chr(120):
         capture() 
         make_1080p()
         #change_res(1280, 720)
         #temp(3000)
        #return msvcrt.getch().decode()
        
    

def main():
    cap=cv2.VideoCapture(1)
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #cap.set(cv2.CAP_PROP_FOCUS, focus_distance)
    
    
    cap.set(3,1280)
    cap.set(4,720)
    
    cap.set(cv2.CAP_PROP_AUTOFOCUS ,20)
    
    d=0
     
    while True:
        cap.isOpened()
        #cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
        
        #cap.set(100, 1280) # set the Horizontal resolution
        #cap.set(100, 720) # Set the Vertical resolution
        
        #repeat()
        ret,frame = cap.read();
        filename="CNN/pictures/capture/test_%d.jpg"%d
        cv2.imwrite(filename, frame)
       
        print(ret)
        print(frame)
        d+=1
        time.sleep(5)
    else:
        ret=False;
        
       
        
    #img1=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img1 = frame
    
    plt.imshow(img1)
    plt.title('color image RGB')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    cv2.imwrite('test.png', img1)
    files = ['test.png']
    for f in files:
                shutil.copy(f,'CNN/pictures/vaguelettes')
   # fileName = "test.png"
    #path_to_file = pjoin ('C:','Users','srabh','Desktop','dossier-test', fileName)
    #file = open (path_to_file, 'w')
    #os.path.isfile('C:\Users\srabh\Desktop\test.png')
    if os.path.isfile('test.png'):
        def saveImage():
            currentImages = test("*.png")
            numList = [0]
            for img in currentImages:
                i = os.path.splitext(img)[0]
                try:
                    num = re.findall('[0-9]+$', i)[0]
                    numList.append(int(num))
                except IndexError:
                    pass
            numList = sorted(numList)
            newNum= max(numlist)+1
            saveName='test.%04d.png'% newNum
            print("Saving %s") %saveName
            spyder.image.save(screen, savename)
        #filename = "test.png"
        #filenum = 0
        #while (pth.exists(filename+str(0)+".png"):
         #   filenum+=1
        #if len(os.listdir(PIC_FOLDER)) > 0:
               # pic_index = np.max([int(n.split('.')[0][len(IMG_SUFFIX):]) for n in os.listdir(PIC_FOLDER)]) + 1
               # pic_index += 1
                       
        print ('Bonjour')
    else:
        print ('Get lost')
    
    cap.release()
    d+=1
    
#def verification():
    #saveImage()
    #import train2.py
    #import test2.py
    
    
        
if __name__ == '__main__':
    #main()
    menu()
    #subprocess.Popen(train2.py)
    
    #pic_index+=1

