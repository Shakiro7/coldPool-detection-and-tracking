#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:24:13 2022

@author: jannik
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from skimage.measure import label
from tracking import track
from utils import unique_nonzero, findObjIndex




class RainPatch:
    
    def __init__(self,identificationNumber,startTimestep,area,rint_mean,age=1):
        
        self.__id = identificationNumber
        self.__startTimestep = startTimestep
        self.__area = area
        self.__rint_mean = rint_mean
        #self.__rain_sum = rain_sum
        self.__age = age
        self.__parents = []
        self.__mainParent = None
        self.__patrons = []

    def __del__(self):
        """
        Deletes the RainPatch object
        """

    def getId(self):
        
        return self.__id
    
    def getStart(self):
        
        return self.__startTimestep
    
    def getArea(self):
        
        return self.__area

    def getRintMean(self):
        
        return self.__rint_mean
    
    # def getTotalRainSum(self):
        
    #     return self.__rain_sum
    
    def getAge(self):
        
        return self.__age

    def getParents(self):
        
        return self.__parents

    def getMainParent(self):
        
        return self.__mainParent
    
    def getPatrons(self):
        
        return self.__patrons
    
    # def setTotalRainSum(self, additionalRainSum):
    #     """
    #     Adds a new rain sum to the previous total rain sum of that RainPatch
    #     """        
    #     self.__rain_sum += additionalRainSum

    def setArea(self,newArea):
        """
        Allows to modify the area of an existing RainPatch
        """          
        self.__area = newArea

    def setAge(self):
        """
        Increases the age of an existing RainPatch by one
        """          
        self.__age += 1

    def setParents(self,parent):
        """
        Adds the ID of a parent ColdPool to the corresponding child RainPatch
        """          
        self.__parents.append(parent)
        
    def setMainParent(self,mainParent):
        """
        Adds the ID of the parent ColdPool with the largest contribution (overlap) to the corresponding child RainPatch
        """          
        self.__mainParent = mainParent

    def setPatrons(self,patron):
        """
        Adds the ID of a patron ColdPool (a CP that promotes the RainPatch after its birth by interactions) to the corresponding RainPatch
        """          
        self.__patrons.append(patron)


class RainField:
    
    # Create empty list to store all RainPatches
    # The rainpatch_list is a class variable (shared by all RainFields) and stores all RainPatches for the selected time steps 
    rainpatch_list = []
    
    def __init__(self,timestep,rainIntensity,rainMarkersOld=None,oldCps=None,rainIntensityThreshold=1,
                 rainPatchMinSize=0,periodicDomain=True):
        
        self.__tstep = timestep
        rint = rainIntensity
        rainMarkersOld = rainMarkersOld
        oldCps = oldCps
        rintThresh = rainIntensityThreshold
        minSize = rainPatchMinSize
        periodicBc = periodicDomain
        
        self.__rainMarkers = label(filters.gaussian(rint, sigma=2.0) >= rintThresh)
        # Check if rain patches are smaller than bounding box in x and y and remove them if yes
        # for patch in  unique_nonzero(self.__rainMarkers, return_counts=False):
        #     slice_x, slice_y = ndi.find_objects(self.__rainMarkers==patch)[0]
        #     bounding_box = self.__rainMarkers[slice_x, slice_y]
        #     if (bounding_box.shape[0]<minSize) or (bounding_box.shape[1]<minSize):
        #         self.__rainMarkers = np.where(self.__rainMarkers==patch,0,self.__rainMarkers)        
        # Check if rain patches are smaller (have less pixel) than rainPatchMinSize
        patches, patches_count = unique_nonzero(self.__rainMarkers, return_counts=True)
        l = 0
        for patch in patches:
            if patches_count[l] < minSize:
                self.__rainMarkers = np.where(self.__rainMarkers==patch,0,self.__rainMarkers)
            l += 1
        
        # Change labels to start after the max of the last timestep (rainMarkers & old cps) if existing
        if (rainMarkersOld is not None) and (len(RainField.rainpatch_list) > 0):
            label_list = np.unique(self.__rainMarkers)
            if label_list[0] == 0:                    
                label_list = label_list[1:]        
            j = len(label_list)
            lastMax = max([obj.getId() for obj in RainField.rainpatch_list])
            if np.max(oldCps) > lastMax:
                lastMax = np.max(oldCps)
            for labl in reversed(label_list):
                self.__rainMarkers = np.where(self.__rainMarkers==labl,lastMax+j,self.__rainMarkers)
                j -= 1   

        # Take care of periodic BC for rain marker patches
        if periodicBc:
            for k in range(self.__rainMarkers.shape[0]):
                if self.__rainMarkers[k, 0] > 0 and self.__rainMarkers[k, -1] > 0:
                    self.__rainMarkers[self.__rainMarkers == self.__rainMarkers[k, -1]] = self.__rainMarkers[k, 0]
            for k in range(self.__rainMarkers.shape[1]):
                if self.__rainMarkers[0, k] > 0 and self.__rainMarkers[-1, k] > 0:
                    self.__rainMarkers[self.__rainMarkers == self.__rainMarkers[-1, k]] = self.__rainMarkers[0, k]

        # Track rain patches from previous RainField if existing
        if rainMarkersOld is not None:
            self.__rainMarkers = track(newLabeledField=self.__rainMarkers,oldLabeledField=rainMarkersOld,
                                       trackingFactor=0.01)

        # Track rain patches with X% overlap with old cold pools
        if oldCps is not None:
            self.__rainMarkers = track(newLabeledField=self.__rainMarkers,oldLabeledField=oldCps,
                                       trackingFactor=1.0)

        # Count the number of different rain patches
        self.__numberOfRainPatches = np.count_nonzero(np.unique(self.__rainMarkers,return_counts=False))

        # Create RainPatches for all unique labels in rainMarkers and store them in the list
        rain_labels, rain_counts = unique_nonzero(self.__rainMarkers, return_counts=True)
        n = 0                   
        if rainMarkersOld is not None:
            for rain in rain_labels:
                if any(obj.getId() == rain for obj in RainField.rainpatch_list):
                    # Only modify existing RainPatch
                    index = findObjIndex(RainField.rainpatch_list,rain)
                    RainField.rainpatch_list[index].setAge()
                    if rain_counts[n] > RainField.rainpatch_list[index].getArea():
                        RainField.rainpatch_list[index].setArea(rain_counts[n])                    

                else:
                    # Create new RainPatch and append it to rainpatch_list
                    rain_region = self.__rainMarkers == rain
                    rainpatch = RainPatch(identificationNumber=rain,startTimestep=self.__tstep,area=rain_counts[n],
                                          rint_mean=np.mean(rint[rain_region]))
                    RainField.rainpatch_list.append(rainpatch)                     

                n += 1
                                        
        else:
            # Make sure rainpatch_list is empty at first
            RainField.rainpatch_list = []
            for rain in rain_labels:
                # Create new RainPatches for all rain patches and append them to rainpatch_list
                rain_region = self.__rainMarkers == rain
                rainpatch = RainPatch(identificationNumber=rain,startTimestep=self.__tstep,area=rain_counts[n], 
                                      rint_mean=np.mean(rint[rain_region]))
                RainField.rainpatch_list.append(rainpatch)
                n += 1
    
    def __del__(self):
        """
        Deletes the RainField object
        """
        
    def getTimestep(self):
        
        return self.__tstep


    def getRainMarkers(self):
        
        return self.__rainMarkers
    
    def getNumberRainPatches(self):
        
        return self.__numberOfRainPatches
    
