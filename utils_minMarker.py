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


# Function to combine the markers from new rain events and exisiting cold pools
def combineMarkers(rainMarkers,rainPatchList,oldCps,coldPoolList,segmentation,dissipationThresh=3):
    # Create array with labeled minima of active old cold pools
    minima = np.zeros_like(rainMarkers)
    for oldCpLabel in unique_nonzero(oldCps,return_counts=False):
        index = -1
        # Find the old cold pool and the index of its minimum
        for i, obj in enumerate(coldPoolList):
            if obj.getId() == oldCpLabel:
                index = i
                break
        index_minimum = coldPoolList[index].getOrigin()
        # If the segmentation allows it (=is 1), add the label of the old cold pool to minima
        if segmentation[index_minimum] != 0:
            minima[index_minimum] = oldCpLabel
        # If the segmentation doesn't allow it (=is 0), but the parent rain patch is still active, add the label to minima 
        elif (segmentation[index_minimum] == 0) & (oldCpLabel in rainMarkers):
            minima[index_minimum] = oldCpLabel
        # If the segmentation doesn't allow it (=is 0), but the cold pool is is still below the dissipation threshold,
        # add the label to minima, but increase the dissipation time steps by one (setStatus)
        elif (segmentation[index_minimum] == 0) & (coldPoolList[index].getState() < dissipationThresh-1):
            minima[index_minimum] = oldCpLabel
            coldPoolList[index].setState()
        # Else don't add a marker, but only increase the dissipation time steps
        else:
            coldPoolList[index].setState()
         
    # Create the final marker array and add the rain markers
    markers = np.zeros_like(rainMarkers)
    rain_labels, rain_counts = unique_nonzero(rainMarkers,return_counts=True)
    k = 0
    for rain in rain_labels:
        pixel_rain = rainMarkers == rain
        rain_overlap = pixel_rain * oldCps
        unique = np.unique(rain_overlap, return_counts=False)
        # If the rain overlaps no old cold pool, add the rainMarker
        if (len(unique) == 1) & (unique[0] == 0):                    
            markers[pixel_rain] = rain
        # If the rain overlaps 100% or partly with one or more old cold pools, add the rainMarker and save 
        # all old cold pool as its parents if not already done or the old cold pool label equals the rain label
        else:
            markers[pixel_rain] = rain
            # Find the rain patch index and add parent or patron (if the rain patch is not new)
            index = -1
            for i, obj in enumerate(rainPatchList):
                if obj.getId() == rain:
                    index = i
                    break
            if unique[0] == 0:                    
                unique = unique[1:]
            for oldCpLabel in unique:
                if (oldCpLabel != rain) & (oldCpLabel not in rainPatchList[index].getParents()):
                    if (rainPatchList[index].getAge()==1):
                        rainPatchList[index].setParents(oldCpLabel)
                    else:
                        rainPatchList[index].setPatrons(oldCpLabel)
        k += 1
            
    # Add minimum markers if they don't overlap with anything and add segmentation if minimum marker overlaps with 0 or itself
    for minMarker in unique_nonzero(minima, return_counts=False):
        index_minMarker = np.unravel_index((np.where(minima.flatten() == minMarker),), minima.shape)
        product = minMarker * markers[index_minMarker]
        # If minMarker overlaps with 0
        if product == 0:
            markers[index_minMarker] = minMarker
            segmentation = np.where(oldCps == minMarker, 1, segmentation)
            #print("minMarker " + str(minMarker) + " overlaps 0")

        # If minMarker overlaps with itself (due to 100% overlap between rain and old cold pool)
        if product == minMarker*minMarker:
            segmentation = np.where(oldCps == minMarker, 1, segmentation)
            #print("minMarker " + str(minMarker) + " overlaps itself")
        

    return markers, segmentation





