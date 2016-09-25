#Main Module 
'''
The main module which creates objects and is central point from where we start 
training, save features, and test the model 
'''
import TrainingModule as training
reload (training)

import TestingModule as testing
reload (testing)

import FeatureCalculation as fc
reload (fc)

import SaveFeatures as sf
reload (sf)

import collections

def main():
    All_Features_Dict = {}
    Feature_Stats = {}
    
    facereco = training.FaceRecognition("training")
    facereco.FaceDetection(All_Features_Dict)
   
    featureCalc = fc.Feature_Calculation()
    featureCalc.calculate_features(All_Features_Dict, Feature_Stats)
    
    saveFeatures = sf.SaveFeatures()
    collections.OrderedDict(sorted(Feature_Stats.items()))
    saveFeatures.WriteToFile(Feature_Stats)

    testingObject = testing.TestingModule("testing")
    testingObject.Testing()
    
if __name__ == "__main__":
    main()
