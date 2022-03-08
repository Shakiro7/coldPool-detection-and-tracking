#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:35:39 2022

@author: jannik
"""

import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries, watershed
from skimage.measure import label
from skimage import filters
from scipy.io import savemat
from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass
from dataloader import DataLoader




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


# Function to export a matlab array
def exportMatlab(filename,array,arrayName="array"):
    savemat(filename + ".mat", mdict={arrayName: array})


# Function to get the index of the minimum within a blob with respect to a selectable field
def searchBlobMin(pixelBlob, field):
    overlap = pixelBlob * field
    overlap = np.ma.masked_where(overlap == 0, overlap)
    index_minimum = np.unravel_index(overlap.argmin(), overlap.shape)    
    return index_minimum


# Function to return the index of a class object on a sorted list based on its ID
def findObjIndex(object_list,object_id):
    ready = False
    length = len(object_list)
    end = length-1
    start = 0
    counter = 0
    while not ready:
        counter += 1
        if counter == length:
            warnings.warn("Element " + str(object_id) + " not in list. No index returned. Study results carefully.")
            return
        index = start+math.ceil((end-start)/2)
        if object_list[index].getId() < object_id:
            start += math.ceil((end-start)/2)
        elif object_list[index].getId() > object_id:
            end -= math.ceil((end-start)/2)
        else:
            ready = True
            return index


# Function to get the center of masses within blob(s) with respect to a selectable field
def searchCenterOfMass(pixelBlob, field,periodicDomain=True):
    boundary_00 = pixelBlob[0,:]
    boundary_01 = pixelBlob[-1,:]
    boundary_10 = pixelBlob[:,0]
    boundary_11 = pixelBlob[:,-1]
    
    if not periodicDomain or (not (boundary_00.any()==True and boundary_01.any()==True) and not (boundary_10.any()==True and boundary_11.any()==True)):
        labeledBlobs = label(pixelBlob)
        coordinate_arr = np.zeros_like(pixelBlob,dtype=bool)
        for blob in unique_nonzero(labeledBlobs):
            pixel = labeledBlobs == blob
            overlap = pixel * field
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True
        
    elif (boundary_00.any()==True and boundary_01.any()==True) and not (boundary_10.any()==True and boundary_11.any()==True):
        pad_width = ((pixelBlob.shape[0], pixelBlob.shape[0]),(0,0))
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')   
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True
        coordinate_arr = coordinate_arr[pixelBlob.shape[0]:pixelBlob.shape[0]*2,:] 
        
    elif not (boundary_00.any()==True and boundary_01.any()==True) and (boundary_10.any()==True and boundary_11.any()==True):
        pad_width = ((0,0),(pixelBlob.shape[1], pixelBlob.shape[1]))
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')   
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True
        coordinate_arr = coordinate_arr[:,pixelBlob.shape[1]:pixelBlob.shape[1]*2]  
    else:
        pad_width = (pixelBlob.shape[0], pixelBlob.shape[1])
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')   
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True
        coordinate_arr = coordinate_arr[pixelBlob.shape[0]:pixelBlob.shape[0]*2, 
                                        pixelBlob.shape[1]:pixelBlob.shape[1]*2]        
        
    return coordinate_arr      


# Function to get the coordinates of the center of mass within a blob with respect to a selectable field
def searchOrigin(pixelBlob, field,periodicDomain=True):
    boundary_00 = pixelBlob[0,:]
    boundary_01 = pixelBlob[-1,:]
    boundary_10 = pixelBlob[:,0]
    boundary_11 = pixelBlob[:,-1]
    
    if not periodicDomain or (not (boundary_00.any()==True and boundary_01.any()==True) and not (boundary_10.any()==True and boundary_11.any()==True)):
        labeledBlobs = label(pixelBlob)
        coordinate_arr = np.zeros_like(labeledBlobs,dtype=bool)  
        for blob in unique_nonzero(labeledBlobs):
            pixel = labeledBlobs == blob
            overlap = pixel * field
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True               
        coordinates = np.where(coordinate_arr == True)
        if len(coordinates[0]) > 1:
            # If origin is not unique, find the candidate with the lowest field value and discard the others
            coordinate_arrNew = np.zeros_like(coordinate_arr,dtype=bool)
            maximum = np.ma.MaskedArray.max(np.ma.masked_where(coordinate_arr*field==False,coordinate_arr*field))
            coordinate_arrNew = np.where(coordinate_arr*field==maximum,True,coordinate_arrNew)
            coordinates = np.where(coordinate_arrNew == True)
            # If still not unique raise error
            if len(coordinates[0]) > 1:          
                raise ValueError('No unique origin found: ' + str(coordinates))
            # If origin is unique now, display a warning
            else:                
                coordinate_center = (int(coordinates[0]),int(coordinates[1]))
                warnings.warn("No unique origin found. Selected " + str(coordinate_center) + " based on highest field value.")
        else:
            coordinate_center = (int(coordinates[0]),int(coordinates[1]))        
        
    elif (boundary_00.any()==True and boundary_01.any()==True) and not (boundary_10.any()==True and boundary_11.any()==True):        
        pad_width = ((pixelBlob.shape[0], pixelBlob.shape[0]),(0,0))
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)  
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True        
        coordinate_arr = coordinate_arr[pixelBlob.shape[0]:pixelBlob.shape[0]*2,:]         
        coordinates = np.where(coordinate_arr == True)
        if len(coordinates[0]) > 1:
            # If origin is not unique, find the candidate with the lowest field value and discard the others
            coordinate_arrNew = np.zeros_like(coordinate_arr,dtype=bool)
            maximum = np.ma.MaskedArray.max(np.ma.masked_where(coordinate_arr*field==False,coordinate_arr*field))
            coordinate_arrNew = np.where(coordinate_arr*field==maximum,True,coordinate_arrNew)
            coordinates = np.where(coordinate_arrNew == True)
            # If still not unique raise error
            if len(coordinates[0]) > 1:          
                raise ValueError('No unique origin found: ' + str(coordinates))
            # If origin is unique now, display a warning
            else:                
                coordinate_center = (int(coordinates[0]),int(coordinates[1]))
                warnings.warn("No unique origin found. Selected " + str(coordinate_center) + " based on highest field value.")
        else:
            coordinate_center = (int(coordinates[0]),int(coordinates[1]))        
        
    elif not (boundary_00.any()==True and boundary_01.any()==True) and (boundary_10.any()==True and boundary_11.any()==True):        
        pad_width = ((0,0),(pixelBlob.shape[1], pixelBlob.shape[1]))
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)  
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True        
        coordinate_arr = coordinate_arr[:,pixelBlob.shape[1]:pixelBlob.shape[1]*2]          
        coordinates = np.where(coordinate_arr == True)
        if len(coordinates[0]) > 1:
            # If origin is not unique, find the candidate with the lowest field value and discard the others
            coordinate_arrNew = np.zeros_like(coordinate_arr,dtype=bool)
            maximum = np.ma.MaskedArray.max(np.ma.masked_where(coordinate_arr*field==False,coordinate_arr*field))
            coordinate_arrNew = np.where(coordinate_arr*field==maximum,True,coordinate_arrNew)
            coordinates = np.where(coordinate_arrNew == True)
            # If still not unique raise error
            if len(coordinates[0]) > 1:          
                raise ValueError('No unique origin found: ' + str(coordinates))
            # If origin is unique now, display a warning
            else:                
                coordinate_center = (int(coordinates[0]),int(coordinates[1]))
                warnings.warn("No unique origin found. Selected " + str(coordinate_center) + " based on highest field value.")
        else:
            coordinate_center = (int(coordinates[0]),int(coordinates[1]))        
        
    else:        
        pad_width = (pixelBlob.shape[0], pixelBlob.shape[1])
        labeledBlobs_pad = label(np.pad(pixelBlob,pad_width,mode='wrap'))
        field_pad = np.pad(field,pad_width,mode='wrap')
        coordinate_arr = np.zeros_like(labeledBlobs_pad,dtype=bool)  
        for blob in unique_nonzero(labeledBlobs_pad):
            pixel = labeledBlobs_pad == blob
            overlap = pixel * field_pad
            index_centerOfMassFloat = center_of_mass(overlap)
            index_centerOfMass = tuple([round(x) if isinstance(x, float) else x for x in index_centerOfMassFloat])
            coordinate_arr[index_centerOfMass] = True        
        coordinate_arr = coordinate_arr[pixelBlob.shape[0]:pixelBlob.shape[0]*2, 
                                        pixelBlob.shape[1]:pixelBlob.shape[1]*2]         
        coordinates = np.where(coordinate_arr == True)
        if len(coordinates[0]) > 1:
            # If origin is not unique, find the candidate with the lowest field value and discard the others
            coordinate_arrNew = np.zeros_like(coordinate_arr,dtype=bool)
            maximum = np.ma.MaskedArray.max(np.ma.masked_where(coordinate_arr*field==False,coordinate_arr*field))
            coordinate_arrNew = np.where(coordinate_arr*field==maximum,True,coordinate_arrNew)
            coordinates = np.where(coordinate_arrNew == True)
            # If still not unique raise error
            if len(coordinates[0]) > 1:          
                raise ValueError('No unique origin found: ' + str(coordinates))
            # If origin is unique now, display a warning
            else:                
                coordinate_center = (int(coordinates[0]),int(coordinates[1]))
                warnings.warn("No unique origin found. Selected " + str(coordinate_center) + " based on highest field value.")
        else:
            coordinate_center = (int(coordinates[0]),int(coordinates[1]))

    return coordinate_center  


# Function to check if a blob is in contact with another blob
def checkBlobContact(pixelBlob,labeledBlobs):
    boundary_blob_bool = find_boundaries(pixelBlob, connectivity=1, mode='outer', background=0)
    boundary_blob = boundary_blob_bool * labeledBlobs
    if len(np.unique(boundary_blob)) == 1:
        return False
    else:
        return True


# Function to create a list that contains the labels of the root ends of a merged cold pool 
def findRootEnds(coldPoolList,contributorList):    
    mergedCp_list = contributorList.copy()
    rootCp_list = []        
    for cp in mergedCp_list:
        for i, obj in enumerate(coldPoolList):
            if obj.getId() == cp:
                index_cp = i
                break 
        if len(coldPoolList[index_cp].getMerged()) == 0:
            rootCp_list.append(cp)
        else:
            for prev_cp in coldPoolList[index_cp].getMerged():
                mergedCp_list.append(prev_cp)    
    
    return rootCp_list
    

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
                    del prey[i]
                    del predator[i]
                    del repitition_count[i]
                    i -= 1
            i += 1
    return predator, prey
    

# Function to create lists for master predators and their preys.
# If a prey is also predator, the master predator gets those preys as well
def createUniquePredatorPreyLists(predatorList,preyList):
    predator = predatorList.copy()
    prey = preyList.copy()
  
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
        if count == 1000:
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


# Function to flood a masked elevation map from labeled markers and fill potential holes
def createLabeledCps(markers,elevationMap,mask,periodicDomain=True,fillOnlyBackgroundHoles=False):
    boundary_00 = mask[0,:]
    boundary_01 = mask[-1,:]
    boundary_10 = mask[:,0]
    boundary_11 = mask[:,-1]
    
    if not periodicDomain or (not (boundary_00.any() and boundary_01.any()) and not (boundary_10.any() and boundary_11.any())):
        labeledCps = watershed(elevationMap, markers, mask=mask)
        if fillOnlyBackgroundHoles:
            for cp in unique_nonzero(labeledCps):
                # Fill possible holes
                filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                if not np.array_equal(filled_cp, (labeledCps == cp)):
                    filled_cp_zeros = (filled_cp == True) & (labeledCps == 0)
                    fill_blobs = filled_cp_zeros & (labeledCps != cp)
                    labeled_fill_blobs = label(fill_blobs)
                    for blob in unique_nonzero(labeled_fill_blobs):
                        boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                        boundary_blob = boundary_blob_bool * labeledCps
                        if all(unique_nonzero(boundary_blob)==cp):
                            labeledCps[labeled_fill_blobs==blob] = cp
        else:
            for cp in unique_nonzero(labeledCps):
                # Fill possible holes in the cp
                filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                labeledCps = np.where(filled_cp, cp, labeledCps)

    elif (boundary_00.any() and boundary_01.any()) and not (boundary_10.any() and boundary_11.any()):
        pad_width = ((mask.shape[0], mask.shape[0]),(0,0))
        labeledCps = watershed(np.pad(elevationMap,pad_width,mode='wrap'), np.pad(markers,pad_width,mode='wrap'), 
                               mask=np.pad(mask,pad_width,mode='wrap'))
        # If cold pools emerged at boundary 0, fill holes with periodic boundary 0, else without  
        if labeledCps[mask.shape[0],:].any() and labeledCps[mask.shape[0]*2-1,:].any():
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = labeledCps[mask.shape[0]:mask.shape[0]*2,:] 
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    if cp not in boundary_00 and cp not in boundary_01:
                        filled_cp = ndi.binary_fill_holes(labeledCpsCenter == cp)
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        filled_cp = filled_cp[mask.shape[0]:mask.shape[0]*2,:] 
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    if cp not in boundary_00 and cp not in boundary_01:
                        filled_cp = ndi.binary_fill_holes(labeledCps[mask.shape[0]:mask.shape[0]*2,:] == cp)
                        labeledCps[mask.shape[0]:mask.shape[0]*2,:] = np.where(filled_cp, cp, labeledCps[mask.shape[0]:mask.shape[0]*2,:])                 
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        labeledCps = np.where(filled_cp, cp, labeledCps)
                labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2,:]
        else:
            labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2,:]
            if fillOnlyBackgroundHoles:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    if not np.array_equal(filled_cp, (labeledCps == cp)):
                        filled_cp_zeros = (filled_cp == True) & (labeledCps == 0)
                        fill_blobs = filled_cp_zeros & (labeledCps != cp)
                        labeled_fill_blobs = label(fill_blobs)
                        for blob in unique_nonzero(labeled_fill_blobs):
                            boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                            boundary_blob = boundary_blob_bool * labeledCps
                            if all(unique_nonzero(boundary_blob)==cp):
                                labeledCps[labeled_fill_blobs==blob] = cp
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    labeledCps = np.where(filled_cp, cp, labeledCps)            

    elif not (boundary_00.any() and boundary_01.any()) and (boundary_10.any() and boundary_11.any()):
        pad_width = ((0,0),(mask.shape[1], mask.shape[1]))
        labeledCps = watershed(np.pad(elevationMap,pad_width,mode='wrap'), np.pad(markers,pad_width,mode='wrap'), 
                               mask=np.pad(mask,pad_width,mode='wrap'))
        # If cold pools emerged at boundary 1, fill holes with periodic boundary 1, else without  
        if labeledCps[:,mask.shape[1]].any() and labeledCps[:,mask.shape[1]*2-1].any():        
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = labeledCps[:,mask.shape[1]:mask.shape[1]*2]  
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    if cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCpsCenter  == cp)
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        filled_cp = filled_cp[:,mask.shape[1]:mask.shape[1]*2] 
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    if cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCps[:,mask.shape[1]:mask.shape[1]*2] == cp)
                        labeledCps[:,mask.shape[1]:mask.shape[1]*2] = np.where(filled_cp, cp, labeledCps[:,mask.shape[1]:mask.shape[1]*2])                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        labeledCps = np.where(filled_cp, cp, labeledCps)
                labeledCps = labeledCps[:,mask.shape[1]:mask.shape[1]*2] 
        else:
            labeledCps = labeledCps[:,mask.shape[1]:mask.shape[1]*2] 
            if fillOnlyBackgroundHoles:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    if not np.array_equal(filled_cp, (labeledCps == cp)):
                        filled_cp_zeros = (filled_cp == True) & (labeledCps == 0)
                        fill_blobs = filled_cp_zeros & (labeledCps != cp)
                        labeled_fill_blobs = label(fill_blobs)
                        for blob in unique_nonzero(labeled_fill_blobs):
                            boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                            boundary_blob = boundary_blob_bool * labeledCps
                            if all(unique_nonzero(boundary_blob)==cp):
                                labeledCps[labeled_fill_blobs==blob] = cp
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    labeledCps = np.where(filled_cp, cp, labeledCps)             
                        
    else:
        pad_width = (mask.shape[0], mask.shape[1])
        labeledCps = watershed(np.pad(elevationMap,pad_width,mode='wrap'), np.pad(markers,pad_width,mode='wrap'), 
                               mask=np.pad(mask,pad_width,mode='wrap'))
        centerBoundary_00 = labeledCps[mask.shape[0],:]
        centerBoundary_01 = labeledCps[mask.shape[0]*2-1,:]
        centerBoundary_10 = labeledCps[:,mask.shape[1]]
        centerBoundary_11 = labeledCps[:,mask.shape[1]*2-1]
        # If cold pools emerged at boundary 0 and boundary 1, fill also holes with fully periodic boundaries 
        if centerBoundary_00.any() and centerBoundary_01.any() and centerBoundary_10.any() and centerBoundary_11.any():          
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = labeledCps[mask.shape[0]:mask.shape[0]*2, 
                                              mask.shape[1]:mask.shape[1]*2]
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    if cp not in boundary_01 and cp not in boundary_10 and cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCpsCenter == cp)
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        filled_cp = filled_cp[mask.shape[0]:mask.shape[0]*2, 
                                              mask.shape[1]:mask.shape[1]*2]
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    if cp not in boundary_01 and cp not in boundary_10 and cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCps[mask.shape[0]:mask.shape[0]*2,mask.shape[1]:mask.shape[1]*2] == cp)
                        labeledCps[mask.shape[0]:mask.shape[0]*2,mask.shape[1]:mask.shape[1]*2] = np.where(filled_cp, cp, labeledCps[mask.shape[0]:mask.shape[0]*2,mask.shape[1]:mask.shape[1]*2])                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        labeledCps = np.where(filled_cp, cp, labeledCps)                        
                labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2, 
                                        mask.shape[1]:mask.shape[1]*2]                   
        # If cold pools emerged at boundary 0, but not boundary 1, fill holes with periodic boundary 0 only 
        elif (centerBoundary_00.any() and centerBoundary_01.any()) and not (centerBoundary_10.any() and centerBoundary_11.any()): 
            labeledCps = labeledCps[:,mask.shape[1]:mask.shape[1]*2] 
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = labeledCps[mask.shape[0]:mask.shape[0]*2,:] 
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    if cp not in boundary_00 and cp not in boundary_01:
                        filled_cp = ndi.binary_fill_holes(labeledCpsCenter  == cp)
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        filled_cp = filled_cp[mask.shape[0]:mask.shape[0]*2,:] 
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    if cp not in boundary_00 and cp not in boundary_01:
                        filled_cp = ndi.binary_fill_holes(labeledCps[mask.shape[0]:mask.shape[0]*2,:] == cp)
                        labeledCps[mask.shape[0]:mask.shape[0]*2,:] = np.where(filled_cp, cp, labeledCps[mask.shape[0]:mask.shape[0]*2,:])                 
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        labeledCps = np.where(filled_cp, cp, labeledCps)
                labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2,:]
        # If cold pools emerged at boundary 1, but not boundary 0, fill holes with periodic boundary 1 only 
        elif not (centerBoundary_00.any() and centerBoundary_01.any()) and (centerBoundary_10.any() and centerBoundary_11.any()): 
            labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2,:]
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = labeledCps[:,mask.shape[1]:mask.shape[1]*2]  
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    if cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCpsCenter == cp)
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        filled_cp = filled_cp[:,mask.shape[1]:mask.shape[1]*2] 
                        if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                            filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                            fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                            labeled_fill_blobs = label(fill_blobs)
                            for blob in unique_nonzero(labeled_fill_blobs):
                                boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                                boundary_blob = boundary_blob_bool * labeledCpsCenter
                                if all(unique_nonzero(boundary_blob)==cp):
                                    labeledCpsCenter[labeled_fill_blobs==blob] = cp
                labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    if cp not in boundary_10 and cp not in boundary_11:
                        filled_cp = ndi.binary_fill_holes(labeledCps[:,mask.shape[1]:mask.shape[1]*2] == cp)
                        labeledCps[:,mask.shape[1]:mask.shape[1]*2] = np.where(filled_cp, cp, labeledCps[:,mask.shape[1]:mask.shape[1]*2])                        
                    else:
                        filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                        labeledCps = np.where(filled_cp, cp, labeledCps)
                labeledCps = labeledCps[:,mask.shape[1]:mask.shape[1]*2] 
        # If no cold pool emerged at both boundaries, fill holes without periodic boundaries
        else:
            labeledCps = labeledCps[mask.shape[0]:mask.shape[0]*2, 
                                    mask.shape[1]:mask.shape[1]*2]
            if fillOnlyBackgroundHoles:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    if not np.array_equal(filled_cp, (labeledCps == cp)):
                        filled_cp_zeros = (filled_cp == True) & (labeledCps == 0)
                        fill_blobs = filled_cp_zeros & (labeledCps != cp)
                        labeled_fill_blobs = label(fill_blobs)
                        for blob in unique_nonzero(labeled_fill_blobs):
                            boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                            boundary_blob = boundary_blob_bool * labeledCps
                            if all(unique_nonzero(boundary_blob)==cp):
                                labeledCps[labeled_fill_blobs==blob] = cp
            else:
                for cp in unique_nonzero(labeledCps):
                    # Fill possible holes in the cp
                    filled_cp = ndi.binary_fill_holes(labeledCps == cp)
                    labeledCps = np.where(filled_cp, cp, labeledCps)            
            
    return labeledCps 


# Function to create markers from new rain events and exisiting cold pools
def createMarkers(rainfield_list,rainPatchList,segmentation,dataset,
                  coldPoolList=None,oldCps=None,dissipationThresh=3,periodicDomain=True):
    
    # Select field for center of mass evaluation: True -> rint, False -> t + w
    rintFieldCenter = False
    
    dissipationThresh = dissipationThresh
    dataloader = DataLoader(dataset=dataset, timestep=rainfield_list[-1].getTimestep())    
    if rintFieldCenter:
        field = filters.gaussian(dataloader.getRint(), sigma=2.0)
    else:
        field = invert01(scale01(scale01(filters.gaussian(dataloader.getT(), sigma=1.0))+
                                 scale01(filters.gaussian(dataloader.getW(), sigma=2.0))))

        
    # Get current rain field with labeled rain patches
    rainMarkers = rainfield_list[-1].getRainMarkers()
    rain_labels, rain_counts = unique_nonzero(rainMarkers,return_counts=True)
    
    # Create empty array to collect the combined markers in
    markers = np.zeros_like(rainMarkers)
    
    if oldCps is not None:
        # Loop over old cold pools and add markers (either active rain or last active rain patch)
        for oldCpLabel in unique_nonzero(oldCps,return_counts=False):
            # Check if the old cold pool still has active rain
            if oldCpLabel in rain_labels:
                # If yes, add the current marker and the origin one to the markers array,add the old region to the segmentation 
                # and store patrons (overlapping old cps) if any
                index_oldCp = findObjIndex(coldPoolList,oldCpLabel)            
                pixel_rain = rainMarkers == oldCpLabel
                pixel_origin = coldPoolList[index_oldCp].getOrigin()
                markers[searchCenterOfMass(pixel_rain, field,periodicDomain=periodicDomain)] = oldCpLabel
                if oldCps[pixel_origin]==oldCpLabel:
                    markers[pixel_origin] = np.where(rainMarkers[pixel_origin]==0,oldCpLabel,markers[pixel_origin])
                segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
                rain_overlap = pixel_rain * oldCps
                unique = unique_nonzero(rain_overlap, return_counts=False)
                if oldCpLabel in unique:
                    itemindex = np.where(unique==oldCpLabel)
                    unique = np.delete(unique, itemindex)
                if len(unique) > 0:
                    # Find the index of the rain patch
                    index = findObjIndex(rainPatchList,oldCpLabel)               
                    for patron in unique:
                        if patron not in rainPatchList[index].getPatrons():
                            rainPatchList[index].setPatrons(patron)
            else:
                # If no, get the last rain patch of that cold pool and check if the segmentation still allows it
                # First find the old cold pool and check if it merged
                index_oldCp = findObjIndex(coldPoolList,oldCpLabel)
                if len(coldPoolList[index_oldCp].getMerged()) > 0:
                    # Check if the merged CP already had own rain. If not, take the last rain of the contributors
                    if oldCpLabel in [obj.getId() for obj in rainPatchList]:
                        print(str(oldCpLabel) + " has own rain")
                        index = findObjIndex(rainPatchList,oldCpLabel) 
                        lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                        for i, obj in enumerate(rainfield_list):
                            if obj.getTimestep() == lastTimestep:
                                index = i
                                break             
                        oldRainMarkers = rainfield_list[index].getRainMarkers()
                        dataloaderOld = DataLoader(dataset=dataset, timestep=rainfield_list[index].getTimestep())    
                        if rintFieldCenter:
                            oldField = filters.gaussian(dataloaderOld.getRint(), sigma=2.0)
                        else:
                            oldField = invert01(scale01(scale01(filters.gaussian(dataloaderOld.getT(), sigma=1.0))+
                                                        scale01(filters.gaussian(dataloaderOld.getW(), sigma=2.0)))) 
                        pixel_rain = oldRainMarkers == oldCpLabel
                        pixel_rainMarker = searchCenterOfMass(pixel_rain, oldField,periodicDomain=periodicDomain)
                    else:
                        print(str(oldCpLabel) + " has no own rain")
                        pixel_rain = np.zeros_like(rainMarkers,dtype=bool)
                        pixel_rainMarker = np.zeros_like(rainMarkers,dtype=bool)
                        root_list = findRootEnds(coldPoolList,coldPoolList[index_oldCp].getMerged())
                        print(coldPoolList[index_oldCp].getMerged())
                        print(root_list)
                        root_list = list(set(root_list))
                        for merged_cp in root_list:
                            index = findObjIndex(rainPatchList,merged_cp) 
                            lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                            for i, obj in enumerate(rainfield_list):
                                if obj.getTimestep() == lastTimestep:
                                    index = i
                                    break
                            oldRainMarkers = rainfield_list[index].getRainMarkers()
                            dataloaderOld = DataLoader(dataset=dataset, timestep=rainfield_list[index].getTimestep())    
                            if rintFieldCenter:
                                oldField = filters.gaussian(dataloaderOld.getRint(), sigma=2.0)
                            else:
                                oldField = invert01(scale01(scale01(filters.gaussian(dataloaderOld.getT(), sigma=1.0))+
                                                            scale01(filters.gaussian(dataloaderOld.getW(), sigma=2.0)))) 
                            pixel_rain = np.where(oldRainMarkers == merged_cp,True,pixel_rain)
                            pixel_rainMarker = np.where(searchCenterOfMass(oldRainMarkers == merged_cp, oldField,periodicDomain=periodicDomain),True,pixel_rainMarker)                                                
                else:
                    index = findObjIndex(rainPatchList,oldCpLabel) 
                    lastTimestep = rainPatchList[index].getStart() + rainPatchList[index].getAge() - 1
                    for i, obj in enumerate(rainfield_list):
                        if obj.getTimestep() == lastTimestep:
                            index = i
                            break             
                    oldRainMarkers = rainfield_list[index].getRainMarkers()
                    dataloaderOld = DataLoader(dataset=dataset, timestep=rainfield_list[index].getTimestep())    
                    if rintFieldCenter:
                        oldField = filters.gaussian(dataloaderOld.getRint(), sigma=2.0)
                    else:
                        oldField = invert01(scale01(scale01(filters.gaussian(dataloaderOld.getT(), sigma=1.0))+
                                                    scale01(filters.gaussian(dataloaderOld.getW(), sigma=2.0))))                    
                    pixel_rain = oldRainMarkers == oldCpLabel
                    pixel_rainMarker = searchCenterOfMass(pixel_rain, oldField,periodicDomain=periodicDomain)
                pixel_count_rain = np.count_nonzero(pixel_rain)
                pixel_origin = coldPoolList[index_oldCp].getOrigin()
                rain_overlap = pixel_rain * segmentation
                # If the segmentation still allows 100% of that rain patch (is 1 everywhere): add a rain marker to markers
                # if it doesn't overlap current rain markers and add the old cold pool region to the segmentation
                if np.count_nonzero(rain_overlap) == pixel_count_rain:
                    markers[pixel_rainMarker] = np.where((rainMarkers[pixel_rainMarker]==0)&(oldCps[pixel_rainMarker]==oldCpLabel),
                                                         oldCpLabel,markers[pixel_rainMarker])
                    if oldCps[pixel_origin]==oldCpLabel:
                        markers[pixel_origin] = np.where(rainMarkers[pixel_origin]==0,oldCpLabel,markers[pixel_origin])
                    if oldCpLabel in markers[pixel_rainMarker] or markers[pixel_origin] == oldCpLabel:
                        segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
                # If the segmentation still allows at least one pixel of that rain patch: add that rain patch to markers
                # where it doesn't overlap current rain markers,add the old cold pool region to the segmentation and set dissipating            
                elif np.count_nonzero(rain_overlap) != 0:
                    markers[pixel_rainMarker] = np.where((rainMarkers[pixel_rainMarker]==0)&(oldCps[pixel_rainMarker]==oldCpLabel),
                                                         oldCpLabel,markers[pixel_rainMarker])
                    if oldCps[pixel_origin]==oldCpLabel:
                        markers[pixel_origin] = np.where(rainMarkers[pixel_origin]==0,oldCpLabel,markers[pixel_origin])                    
                    if oldCpLabel in markers[pixel_rainMarker] or markers[pixel_origin] == oldCpLabel:
                        segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)
                    index = findObjIndex(coldPoolList,oldCpLabel)                
                    coldPoolList[index].setState()
                    # print("CP " + str(oldCpLabel) + " partly dissipated. Increased state from " + 
                    #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))
                # If the segmentation does not allow it in the whole rain patch: set dissipating and add the rain patch
                # and its segmentation only if the dissipation threshold is not reached yet
                else:
                    index = findObjIndex(coldPoolList,oldCpLabel)                  
                    if coldPoolList[index].getState() < dissipationThresh:
                        markers[pixel_rainMarker] = np.where((rainMarkers[pixel_rainMarker]==0)&(oldCps[pixel_rainMarker]==oldCpLabel),
                                                             oldCpLabel,markers[pixel_rainMarker])
                        if oldCps[pixel_origin]==oldCpLabel:
                            markers[pixel_origin] = np.where(rainMarkers[pixel_origin]==0,oldCpLabel,markers[pixel_origin])                        
                        if oldCpLabel in markers[pixel_rainMarker] or markers[pixel_origin] == oldCpLabel:
                            segmentation = np.where(oldCps == oldCpLabel, 1, segmentation)                    
                        coldPoolList[index].setState()
                        # print("CP " + str(oldCpLabel) + " dissipated, but below threshold. Increased state from " + 
                        #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))
                    else:
                        coldPoolList[index].setState()
                        # print("CP " + str(oldCpLabel) + " dissipated and above threshold. Increased state from " + 
                        #       str(coldPoolList[index].getState()-1) + " to " + str(coldPoolList[index].getState()))
        
        # Loop over remaining rainMarkers (= new rain patches) and add teir center of mass to markers
        new_rain_labels = [x for x in rain_labels if x not in unique_nonzero(oldCps)]
        for new_rain in new_rain_labels:
            pixel_new_rain = rainMarkers == new_rain
            markers[searchCenterOfMass(pixel_new_rain, field,periodicDomain=periodicDomain)] = new_rain
            # Check if the new rain overlaps with old cold pools. If yes, add them as parents
            new_rain_overlap = pixel_new_rain * oldCps
            unique = unique_nonzero(new_rain_overlap, return_counts=False)        
            if len(unique) > 0:
                # Find the index of the rain patch
                index = findObjIndex(rainPatchList,new_rain)               
                for parent in unique:
                    if parent not in rainPatchList[index].getParents():
                        rainPatchList[index].setParents(parent)
    else:
        # Loop over rainMarkers (= new rain patches) and add their center of mass to markers
        for new_rain in rain_labels:
            pixel_new_rain = rainMarkers == new_rain
            markers[searchCenterOfMass(pixel_new_rain, field,periodicDomain=periodicDomain)] = new_rain      
        

    return markers, segmentation





