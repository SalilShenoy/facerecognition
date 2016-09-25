'''
TESTING MODULE            

DESCRIPTION: The module is responsible for testing phase of the model built
             in training phase pf the face recognition. 
    Input: 
        - is a single image from the test set.
    
    Output:
        - give the correct image from the database after computing the features
          calculated for the input image.
          (if 5 or more of the 7 distances extracted are matched then we return
           the image)
        - in case of failure add the new input image to the database
    Return:
        - None
'''

import cv2
import FeatureCalculation as fc

import sys

from shutil import copyfile

class TestingModule:
    testPath = ''
    
    def  __init__(self, path):
        self.InitCascadeFiles()
        TestingModule.testPath = path
    
    def InitCascadeFiles(self):
        self.headShoulderCascade = cv2.CascadeClassifier("HS.xml")
        self.mouthCascade        = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.frontalFaceCascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyeCascade          = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.noseCascade         = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    def Testing(self):
        imageName = raw_input("Enter the name of the image to test : ")
        imageName = TestingModule.testPath + "\\" + imageName + ".jpg"
        
        feature_dict   = {}
        left_eye_list  = []
        right_eye_list = []
        nose_list      = []
        mouth_list     = []
            
        image   = cv2.imread(imageName)

        resized = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_AREA) #resize the image
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        face = self.frontalFaceCascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (30,30))

        for (x,y,w,h) in face:
            cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0),1)
                
            roi_gray = gray[y:y+h, x:x+w]
            roi_color =gray[y:y+h, x:x+w]
                
            eyes  = self.eyeCascade.detectMultiScale(roi_gray)
            nose  = self.noseCascade.detectMultiScale(roi_gray)
            mouth = self.mouthCascade.detectMultiScale(gray, 1.3, 25)
                
            if ((len(eyes) == 2)  and
                (len(nose)  == 1) and
                (len(mouth) == 1)):
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    
                    for (nx,ny,nw,nh) in nose:
                        #print "nose :", nx,ny,nw,nh
                        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,0),2)
                    
                    for (mx, my, mw, mh) in mouth:
                        my = int(my - 0.15 * mh)
                        mw = int(mw + 0.5 * mw)
                        cv2.rectangle(gray, (mx,my),(mx+mw,my+mh), (0, 255, 0), 3)

                    left_eye_list.append(eyes[0][0])
                    left_eye_list.append(eyes[0][1])
                    right_eye_list.append(eyes[1][0])
                    right_eye_list.append(eyes[1][1])
        
                    nose_list.append(nx)
                    nose_list.append(ny)

                    mouth_list.append(mx)
                    mouth_list.append(my)

                    feature_dict["left_eye" ] = left_eye_list
                    feature_dict["right_eye"] = right_eye_list
                    feature_dict["nose"     ] = nose_list
                    feature_dict["mouth"    ] = mouth_list
                    
                    imageNamePart = imageName.split('\\')
                    imageName = imageNamePart[len(imageNamePart)-1]
                    
                    ImageDictionary = {}
                    ImageDistances = {}
                    ImageDictionary[imageName] = feature_dict
                    
                    featureCal = fc.Feature_Calculation()
                    featureCal.calculate_features(ImageDictionary, ImageDistances)

                    self.compareFeatures(ImageDistances,image)
                    
    def compareFeatures(self, ImageDistances, image):
        for key, value in ImageDistances.items():
            ImageDistances_List = [key, value]

        featurethresholdCount = 0
        with open('featuresFile.txt') as f:
            present = False
            
            file_line_list = f.readline()
            
            while file_line_list != "":
                file_line_list = file_line_list.split("\t")
    
                if ImageDistances_List[1]['eyes_dist'] == (float(file_line_list[1])):
                    featurethresholdCount = featurethresholdCount + 1
                    
                if ImageDistances_List[1]['left_eye_nose'] == (float(file_line_list[2])):
                    featurethresholdCount = featurethresholdCount + 1
        
                if ImageDistances_List[1]['right_eye_nose'] == (float(file_line_list[3])):
                    featurethresholdCount = featurethresholdCount + 1
                    
                if ImageDistances_List[1]['mouth_nose'] == (float(file_line_list[4])):
                    featurethresholdCount = featurethresholdCount + 1
                    
                if ImageDistances_List[1]['left_eye_mouth'] == (float(file_line_list[5])):
                    featurethresholdCount = featurethresholdCount + 1
                
                if ImageDistances_List[1]['right_eye_mouth'] == (float(file_line_list[6])):      
                    featurethresholdCount = featurethresholdCount + 1
                    
                if featurethresholdCount >= 5:
                    print "Match"
                    imageMatched = "training\\" + file_line_list[0]
                    imageMatched = cv2.imread(imageMatched)
                    cv2.imshow('Matched Image',imageMatched)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    present = True
                    break
                
                file_line_list = f.readline()
            
            if present == False:
                print "Not Present in Database"
                src = "testing\\" + ImageDistances_List[0]
                copyfile(src, "training")
                print "The database has been updated with the new image. Exiting the system  \
                        and retrain the model"
                sys.exit()
                