#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:25:50 2022

@author: jannik
"""

import numpy as np
from utils import unique_nonzero





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
    
    # Delete old blob labels that are rain-free from the list
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

