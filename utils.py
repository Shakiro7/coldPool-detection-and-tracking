#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:35:39 2022

@author: jannik
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


# Function to compute virtual temperature
def computeTv(temperature, moisture):
    t = temperature
    q = moisture
    return (np.multiply(t, (1 + q / 0.622) / (1 + q)))


# Modified numpy unique function that drops the value 0
def unique_nonzero(array, return_counts=False):    
    if return_counts:
        labels, counts = np.unique(array,return_counts=True)
        if labels[0] == 0:                    
            labels = labels[1:]
            counts = counts[1:]
        return labels, counts
    else:
        labels = np.unique(array,return_counts=False)
        if labels[0] == 0:                    
            labels = labels[1:]
        return labels


# Function to scale an array so that it is in the range between 0 and 1
def scale01(array):
    array01 = (array - np.min(array))/np.ptp(array)
    return array01


# Function to invert an array that is in the range from 0 to 1
def invert01(array01):
    array01_inverted = (array01 - np.ones_like(array01)) * (-1)
    return array01_inverted


# Function to get the index of the minimum within a blob with respect to a selectable field
def searchBlobMin(pixelBlob, field):
    overlap = pixelBlob * field
    overlap = np.ma.masked_where(overlap == 0, overlap)
    index_minimum = np.unravel_index(overlap.argmin(), overlap.shape)    
    return index_minimum


# Function to check if a blob is in contact with another blob
def checkBlobContact(pixelBlob,labeledBlobs):
    boundary_blob_bool = find_boundaries(pixelBlob, connectivity=1, mode='outer', background=0)
    boundary_blob = boundary_blob_bool * labeledBlobs
    if len(np.unique(boundary_blob)) == 1:
        return False
    else:
        return True


# Function to check and correct predator-prey relations that occur in both directions
def correctPredPreyRepitions(predatorList,preyList):
    predator = predatorList
    prey = preyList
    repitition_test = [ [] for _ in range(len(predator)) ]
    for i in range(len(predator)):
        repitition_test[i] = sorted([predator[i], prey[i]])           
    repitition_count=[repitition_test.count(x) for x in repitition_test]
    
    if any(np.array(repitition_count) > 1):
        i = 0
        while i < len(predator):
            if repitition_count[i] > 1:
                pred_count = predator.count(predator[i])
                prey_count = predator.count(prey[i])
                if prey_count >= pred_count:
                    predator_temp = predator[i]
                    predator[i] = prey[i]
                    prey[i] = predator_temp
                    if prey.count(prey[i]) > 1:
                        del prey[i]
                        del predator[i]
                        i -= 1
            i += 1
    return predator, prey
    

# Function to create lists for master predators and their preys.
# If a prey is also predator, the master predator gets those preys as well
def createUniquePredatorPreyLists(predatorList,preyList):
    predator = predatorList
    prey = preyList
  
    # Check if there are equal elements in the lists and if yes, delete those entries
    if any(np.array(predator) == np.array(prey)):
        itemindex = np.where(np.array(predator) == np.array(prey))
        predator = np.delete(predator, itemindex)
        prey = np.delete(prey, itemindex)    
    
    # Check and correct for repeating relationships
    predator,prey = correctPredPreyRepitions(predator, prey)
    
    predatorSet = set(predator)
    intersectionSet = predatorSet.intersection(prey)
    intersection = list(intersectionSet)      
    
    count = 0
    while len(intersection) > 0:
        if count == 100:
            print("Predator list: ")
            print(predatorList)
            print("Prey list: ")
            print(preyList)
            raise RuntimeError('Could not solve predator-prey relations')
        else:
            for predprey in intersection:
                masterpred = predator[prey.index(predprey)]
                predprey_indices = [i for i, val in enumerate(predator) if val == predprey]
                for index in predprey_indices:
                    predator[index] = masterpred
                            
            predatorSet = set(predator)
            intersectionSet = predatorSet.intersection(prey)
            intersection = list(intersectionSet)
            count += 1        
            
    predator_new = list(np.unique(predator))
    prey_new = [ [] for _ in range(len(predator_new)) ]
    
    k = 0
    for pred in predator_new:
        pred_indices = [i for i, val in enumerate(predator) if val == pred]
        for index in pred_indices:
            prey_new[k].append(prey[index])
        k += 1

    return predator_new, prey_new, predator_new+prey



# Function to combine the markers from new rain events and exisiting cold pools
def combineMarkers(rainfield_list,rainPatchList,oldCps,coldPoolList,segmentation,dissipationThresh=3):
    
    dissipationThresh = dissipationThresh
    
    # Get current rain field with labeled rain patches
    rainMarkers = rainfield_list[-1].getRainMarkers()
    rain_labels, rain_counts = unique_nonzero(rainMarkers,return_counts=True)
    
    # Create empty array to collect the combined markers in
    markers = np.zeros_like(rainMarkers)
    
    # Loop over old cold pools and add markers (either active rain or last active rain patch)
    for oldCpLabel in unique_nonzero(oldCps,return_counts=False):
        # Check if the old cold pool still has active rain
        if oldCpLabel in rain_labels:            
            # If yes, add this rain patch to the markers array,add the old region to the segmentation 
            # and store patrons (overlapping old cps) if any
            pixel_rain = rainMarkers == oldCpLabel
            markers[pixel_rain] = oldCpLabel
            segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
            rain_overlap = pixel_rain * oldCps
            unique = unique_nonzero(rain_overlap, return_counts=False)
            if oldCpLabel in unique:
                itemindex = np.where(unique==oldCpLabel)
                unique = np.delete(unique, itemindex)
            if len(unique) > 0:
                # Find the index of the rain patch
                for i, obj in enumerate(rainPatchList):
                    if obj.getId() == oldCpLabel:
                        index = i
                        break                
                for patron in unique:
                    if patron not in rainPatchList[index].getPatrons():
                        rainPatchList[index].setPatrons(patron)
        else:
            # If no, get the last rain patch of that cold pool and check if the segmentation still allows it
            # First find the old cold pool and check if it merged
            for i, obj in enumerate(coldPoolList):
                if obj.getId() == oldCpLabel:
                    index_oldCp = i
                    break
            if len(coldPoolList[index_oldCp].getMerged()) > 0:
                # Check if the merged CP already had own rain. If not, take the last rain of the contributors
                if oldCpLabel in [obj.getId() for obj in rainPatchList]:
                    for i, obj in enumerate(rainPatchList):
                        if obj.getId() == oldCpLabel:
                            index = i
                            break
                    lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                    for i, obj in enumerate(rainfield_list):
                        if obj.getTimestep() == lastTimestep:
                            index = i
                            break             
                    oldRainMarkers = rainfield_list[index].getRainMarkers()
                    pixel_rain = oldRainMarkers == oldCpLabel
                else:
                    pixel_rain = np.zeros_like(rainMarkers,dtype=bool)
                    for merged_cp in coldPoolList[index_oldCp].getMerged():
                        for i, obj in enumerate(rainPatchList):
                            if obj.getId() == merged_cp:
                                index = i
                                break
                        lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                        for i, obj in enumerate(rainfield_list):
                            if obj.getTimestep() == lastTimestep:
                                index = i
                                break
                        oldRainMarkers = rainfield_list[index].getRainMarkers()
                        pixel_rain = np.where(oldRainMarkers == merged_cp,True,pixel_rain)                                            
            else:
                for i, obj in enumerate(rainPatchList):
                    if obj.getId() == oldCpLabel:
                        index = i
                        break
                lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                for i, obj in enumerate(rainfield_list):
                    if obj.getTimestep() == lastTimestep:
                        index = i
                        break             
                oldRainMarkers = rainfield_list[index].getRainMarkers()
                pixel_rain = oldRainMarkers == oldCpLabel
            pixel_count_rain = np.count_nonzero(pixel_rain)
            rain_overlap = pixel_rain * segmentation
            # If the segmentation still allows 100% of that rain patch (is 1 everywhere): add that rain patch to markers
            # where it doesn't overlap current rain markers and add the old cold pool region to the segmentation
            if np.count_nonzero(rain_overlap) == pixel_count_rain:
                markers[pixel_rain] = np.where(rainMarkers[pixel_rain]==0,oldCpLabel,markers[pixel_rain])
                segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
            # If the segmentation still allows at least one pixel of that rain patch: add that rain patch to markers
            # where it doesn't overlap current rain markers,add the old cold pool region to the segmentation and set dissipating            
            elif np.count_nonzero(rain_overlap) != 0:
                markers[pixel_rain] = np.where(rainMarkers[pixel_rain]==0,oldCpLabel,markers[pixel_rain])
                segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
                for i, obj in enumerate(coldPoolList):
                    if obj.getId() == oldCpLabel:
                        index = i
                        break                  
                coldPoolList[index].setState()
                # print("CP " + str(oldCpLabel) + " partly dissipated. Increased state from " + 
                #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))
            # If the segmentation does not allow it in the whole rain patch: set dissipating and add the rain patch
            # and its segmentation only if the dissipation threshold is not reached yet
            else:
                for i, obj in enumerate(coldPoolList):
                    if obj.getId() == oldCpLabel:
                        index = i
                        break                  
                if coldPoolList[index].getState() < dissipationThresh:
                    markers[pixel_rain] = np.where(rainMarkers[pixel_rain]==0,oldCpLabel,markers[pixel_rain])
                    segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)                    
                    coldPoolList[index].setState()
                    # print("CP " + str(oldCpLabel) + " dissipated, but below threshold. Increased state from " + 
                    #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))
                else:
                    coldPoolList[index].setState()
                    # print("CP " + str(oldCpLabel) + " dissipated and above threshold. Increased state from " + 
                    #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))                    
    
    # Loop over remaining rainMarkers (= new rain patches) and add them to markers
    new_rain_labels = [x for x in rain_labels if x not in unique_nonzero(oldCps)]
    for new_rain in new_rain_labels:
        pixel_new_rain = rainMarkers == new_rain
        markers[pixel_new_rain] = new_rain
        # Check if the new rain overlaps with old cold pools. If yes, add them as parents
        new_rain_overlap = pixel_new_rain * oldCps
        unique = unique_nonzero(new_rain_overlap, return_counts=False)        
        if len(unique) > 0:
            # Find the index of the rain patch
            for i, obj in enumerate(rainPatchList):
                if obj.getId() == new_rain:
                    index = i
                    break                
            for parent in unique:
                if parent not in rainPatchList[index].getParents():
                    rainPatchList[index].setParents(parent)              
        

    return markers, segmentation





