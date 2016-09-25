
""" 
   Module Name: Save features
   Description: This module writes the distances and features to a text file to 
                store it on the disk.
                This file is then used in comparing and testing
   Input:       Dictionary which contains all the distances
   Output:      File
"""

#Store the data retrieved from the features.
class SaveFeatures:  
    Feature_Stats = {}
    
    def WriteToFile(self, Feature_Stats):        
        #Error Handle
        with open("featuresFile.txt","w") as fToWrite:
            for imageName in Feature_Stats:
                line = imageName + \
                       "\t" + str(Feature_Stats[imageName]['eyes_dist']) + \
                       "\t" + str(Feature_Stats[imageName]['left_eye_nose']) + \
                       "\t" + str(Feature_Stats[imageName]['right_eye_nose']) + \
                       "\t" + str(Feature_Stats[imageName]['mouth_nose']) + \
                       "\t" + str(Feature_Stats[imageName]['left_eye_mouth']) + \
                       "\t" + str(Feature_Stats[imageName]['right_eye_mouth']) + "\n"
                fToWrite.write(line)