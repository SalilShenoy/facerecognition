'''
TRAINING MODULE            

DESCRIPTION: The module is responsible for training the model for
face recognition. 
    Input: 
        - the path of the source folder which contains the images
          on which the model is to be trained.
        - an empty dictionary which will contain the coordinates 
          of the features like eyes, nose, mouth which are 
          extracted from the face.
    Output:
        - the dictionary which contains the coordinates of the 
          features eyes, nose, mouth.
    Return:
        - None
'''

import cv2
import os

import FeatureCalculation as fc
reload(fc)

class FaceRecognition:
    # Class variable which will contain the path of source folder for training
    # images.
    path = ""
    
    '''
    #Constructor
    parameters : 
        - path : the folder which contains training images
    '''
    def __init__(self, path):
        self.InitCascadeFiles()
        FaceRecognition.path = path
            
    '''
    #Function : InitCascadeFiles
    Description :
        - the function initializes the member variables with the path of the 
          haarcascade files
    Parameters : 
        - none
    '''
    def InitCascadeFiles(self):
        self.headShoulderCascade = cv2.CascadeClassifier("HS.xml")
        self.mouthCascade        = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
        self.frontalFaceCascade  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eyeCascade          = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.noseCascade         = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

    '''
    #Function : FaceDetection
    Description: the function contains the main business logic to calculate the 
                 coordinates of features extracted from the face.
    
    Parameters:
        - All_Features_Dict - This is passed as an empty dictionary which is 
          filled with the coordinates of the features extracted from the face.
    '''
    def FaceDetection(self, All_Features_Dict):
            images = [os.path.join(FaceRecognition.path,f) for f in os.listdir(FaceRecognition.path)]
            for imageName in images:
                feature_dict   = {}
                left_eye_list  = []
                right_eye_list = []
                nose_list      = []
                mouth_list     = []
                
                image = cv2.imread(imageName) 
                resized = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_AREA) #resize the image
                gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
                face = self.frontalFaceCascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (30,30))
                        
                for (x,y,w,h) in face:
                    
                    cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0),1)
                    
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color =gray[y:y+h, x:x+w]
                    
                    eyes = self.eyeCascade.detectMultiScale(roi_gray)
                    nose = self.noseCascade.detectMultiScale(roi_gray)
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
                        All_Features_Dict[imageName] = feature_dict
    
                        #cv2.imshow('Faces Found', gray)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()