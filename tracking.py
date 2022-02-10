#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:25:50 2022

@author: jannik
"""

import numpy as np
from utils import unique_nonzero, createUniquePredatorPreyLists





def track(newLabeledField,oldLabeledField,trackingFactor=0.5):
    """
    Function to compare two labeled fields and track overlapping patches based on a tracking factor.

    Parameters
    ----------
    newLabeledField : array_like
        2D array with labeled patches from current time step.
    oldLabeledField : array_like
        2D array with labeled patches from previous time step.
    trackingFactor : float, optional
        Required overlap (proportion of the new patch) between two patches to be tracked. 
        The default is 0.5.

    Returns a labeled field with patches relabeled based on the tracking.
    -------
    blobs_new : array_like
        2D array with relabeled patches based on the tracking.
    """
    
    blobs_new = newLabeledField
    blobs_old = oldLabeledField
    tf = trackingFactor
    
    blob_labels_new, blob_counts_new = unique_nonzero(blobs_new,return_counts=True)
        
    l = 0             
    k = 0
    updated_index = []
    updated_label = []
    
    for blob in blob_labels_new:
        blob_region = blobs_new == blob
        overlap = blob_region * blobs_old
        unique, number = unique_nonzero(overlap, return_counts=True)                  
        itemindex = np.where(unique==blob)
        unique = np.delete(unique, itemindex)
        number = np.delete(number, itemindex)
        if len(unique) > 0:
            if np.max(number) > (blob_counts_new[k]*tf):
                #print("Overlap detetcted: Patch " + str(blob_labels_new[k]) + " is patch " + str(unique[np.argmax(number)]))
                updated_index.append(k)
                updated_label.append(unique[np.argmax(number)])
        k += 1
    
    for index in updated_index:
        blobs_new = np.where(blobs_new==blob_labels_new[index],updated_label[l],blobs_new)                     
        l += 1
        
    return blobs_new





def merge(newLabeledField,oldLabeledField,rainMarkers,trackingFactor=0.5):
    """
    Function to compare two labeled fields and merge overlapping patches based on a tracking factor (if rain-free).

    Parameters
    ----------
    newLabeledField : array_like
        2D array with labeled patches from current time step.
    oldLabeledField : array_like
        2D array with labeled patches from previous time step.
    trackingFactor : float, optional
        Required overlap ("overruled" proportion of the old patch) between two patches to be merged. 
        The default is 0.5.

    Returns a labeled field with patches relabeled based on the merging.
    -------
    blobs_new : array_like
        2D array with relabeled patches based on the merging.
    """
    
    blobs_new = newLabeledField
    blobs_old = oldLabeledField
    rainMarkers = rainMarkers
    tf = trackingFactor
    
    blob_labels_old, blob_counts_old = unique_nonzero(blobs_old,return_counts=True)
    
    # Delete old blob labels that have active rain from the list (they cannot be overruled)
    for rain in unique_nonzero(rainMarkers):
        oldLabelindex = np.where(blob_labels_old==rain)
        blob_labels_old = np.delete(blob_labels_old, oldLabelindex)
        blob_counts_old = np.delete(blob_counts_old, oldLabelindex)        
        
    l = 0             
    k = 0
    updated_index = []
    updated_label = []
    
    for blob in blob_labels_old:
        blob_region = blobs_old == blob
        overlap = blob_region * blobs_new
        unique, number = unique_nonzero(overlap, return_counts=True)                  
        itemindex = np.where(unique==blob)
        unique = np.delete(unique, itemindex)
        number = np.delete(number, itemindex)
        if len(unique) > 0:
            if np.max(number) > (blob_counts_old[k]*tf):
                #print("Overlap detetcted: Patch " + str(blob_labels_old[k]) + " is patch " + str(unique[np.argmax(number)]))
                updated_index.append(k)
                updated_label.append(unique[np.argmax(number)])
        k += 1
    
    for index in updated_index:
        blobs_new = np.where(blobs_new==blob_labels_old[index],updated_label[l],blobs_new)                     
        l += 1
        
    return blobs_new






def mergeNew(newLabeledField,oldLabeledField,rainMarkers,rainPatchList,coldPoolList,trackingFactor=0.5):
    """
    Function to compare two labeled fields and merge dissipating patches that overlap
    based on a tracking factor. The merged patch gets a new label.

    Parameters
    ----------
    newLabeledField : array_like
        2D array with labeled patches from current time step.
    oldLabeledField : array_like
        2D array with labeled patches from previous time step.
    trackingFactor : float, optional
        Required overlap ("overruled" proportion of the old patch) between two patches to be merged. 
        The default is 0.5.

    Returns a labeled field with patches relabeled based on the merging.
    -------
    blobs_new : array_like
        2D array with relabeled patches based on the merging.
    merged_blobsNew_list : list
        List that stores the new labels of merged blobs.
    merged_blobsOld_list : list
        List that stores the old labels of the merging blobs.        
    """
    
    coldPoolList = coldPoolList
    rainPatchList = rainPatchList
    blobs_new = newLabeledField
    blobs_old = oldLabeledField
    rainMarkers = rainMarkers
    tf = trackingFactor
    
    blob_labels_old, blob_counts_old = unique_nonzero(blobs_old,return_counts=True)
    blob_labels_new = unique_nonzero(blobs_new)
    
    # Keep only labels that are in both fields
    for label in blob_labels_old:
        if label not in blob_labels_new:
            labelindex = np.where(blob_labels_old==label)
            blob_labels_old = np.delete(blob_labels_old, labelindex)
            blob_counts_old = np.delete(blob_counts_old, labelindex)
    for label in blob_labels_new:
        if label not in blob_labels_old:
            blobs_new = np.where(blobs_new==label,0,blobs_new)             
    
    # Delete labels that have active rain from the list (they cannot be overruled)
    for rain in unique_nonzero(rainMarkers):
        oldLabelindex = np.where(blob_labels_old==rain)
        blob_labels_old = np.delete(blob_labels_old, oldLabelindex)
        blob_counts_old = np.delete(blob_counts_old, oldLabelindex)  
        blobs_new = np.where(blobs_new==rain,0,blobs_new)  


    # Delete old blob labels that belong to CPs that are not dissipating (they cannot be overruled)
    for label in blob_labels_old:
        for i, obj in enumerate(coldPoolList):
            if obj.getId() == label:
                index = i
                break                  
        state = coldPoolList[index].getState()        
        if state == 0:
            oldLabelindex = np.where(blob_labels_old==label)
            blob_labels_old = np.delete(blob_labels_old, oldLabelindex)
            blob_counts_old = np.delete(blob_counts_old, oldLabelindex)
            blobs_new = np.where(blobs_new==label,0,blobs_new)

    # Initialize merge dict
    mergeDict = {
        "newLabels": [],
        "predator": [],
        "prey": []}

    # Check if new cold pools overruled old cold pools according to the tracking_factor
    if len(blob_labels_old) > 1:                   
        k = 0
        prey = []
        predator = []        
        for blob in blob_labels_old:
            blob_region = blobs_old == blob
            overlap = blob_region * blobs_new
            unique, number = unique_nonzero(overlap, return_counts=True)                  
            itemindex = np.where(unique==blob)
            unique = np.delete(unique, itemindex)
            number = np.delete(number, itemindex)
            if len(unique) > 0:
                if np.max(number) > (blob_counts_old[k]*tf):
                    #print("Overlap detetcted: Patch " + str(blob_labels_old[k]) + " is patch " + str(unique[np.argmax(number)]))
                    prey.append(blob_labels_old[k])
                    predator.append(unique[np.argmax(number)])
            k += 1        
        predator, prey, predPreyList = createUniquePredatorPreyLists(predatorList=predator, preyList=prey)

        # If any merges are necessary find the last max label, store everything in mergeDict and update the blob field
        if len(predator) > 0:
            lastMax = max([obj.getId() for obj in rainPatchList])
            if np.max(blobs_old) > lastMax:
                lastMax = np.max(blobs_old)
            newLabels = []
            for i in range(len(predator)):
                newLabels.append(lastMax+1+i)
            mergeDict = {
                "newLabels": newLabels,
                "predator": predator,
                "prey": prey}
            # Update the field
            for index in range(len(newLabels)):
                blobs_new = np.where(blobs_new==mergeDict["predator"][index],mergeDict["newLabels"][index],blobs_new)
                for label in mergeDict["prey"][index]:
                    blobs_new = np.where(blobs_new==label,mergeDict["newLabels"][index],blobs_new)                                 
            blobs_new = np.where(blobs_new!=0,blobs_new,newLabeledField)
            return blobs_new, mergeDict, predPreyList
        
        # If not just return an empty dict and the original field
        else:
            return newLabeledField, mergeDict, predPreyList            

    else:
        predPreyList = []
        return newLabeledField, mergeDict, predPreyList

