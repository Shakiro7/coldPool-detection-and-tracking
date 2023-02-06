#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:21:53 2022

@author: jannik
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from utils import invert01, unique_nonzero
from skimage.measure import label



# Function to segment the domain in either "CP possible" or "CP not possible/allowed"
def segmentDomain(tv, u, v, w, rint, minSize=50, patchCheck=False, horResolution=200, fuzzThresh=40, periodicDomain=True):
    # Filter rint field
    rint_filtered = filters.gaussian(rint, sigma=2.0)
    # Assign segmentation input data
    # Virtual temperature input        
    tv_filtered = filters.gaussian(tv, sigma=1.0)
    input_tv = (tv_filtered-np.mean(tv_filtered)).flatten().reshape(-1,1)
    input_tv = invert01(MinMaxScaler().fit_transform(input_tv))   
    
    # Horizontal wind speed input
    u_filtered = filters.gaussian(u, sigma=1.0)
    v_filtered = filters.gaussian(v, sigma=1.0)
    input_uv = np.sqrt((u_filtered-np.mean(u_filtered))**2+(v_filtered-np.mean(v_filtered))**2)
    input_uv = MinMaxScaler().fit_transform(filters.gaussian(input_uv, sigma=1.0).flatten().reshape(-1,1))
    
    # Vertical wind speed input
    # w_filtered = filters.gaussian(w, sigma=2.0)
    # input_w = (w_filtered-np.mean(w_filtered)).flatten().reshape(-1,1)
    # input_w = np.absolute(MinMaxScaler(feature_range=(-1, 1)).fit_transform(input_w))     
    
    # Combine input fields
    input_data = input_tv + input_uv
    
    # Initialize KMeans object specifying the number of desired clusters
    model = KMeans(n_clusters=2, init="k-means++")
    
    # Learning the clustering from the input
    segmentation = model.fit_predict(input_data)

    # Calculated average silhouette score
    # silhouette_avg = silhouette_score(input_data, segmentation,n_jobs=-1)
    # print("Silhouette avg: " + str(silhouette_avg))
    
    segmentation = segmentation.reshape(tv.shape[0],tv.shape[1])     
    
    # Verify and/or match label 1 to cold pools and 0 to the rest
    cpindex = np.where(tv == np.min(tv))    
    if segmentation[cpindex].all() == 0:
        segmentation = np.where((segmentation==0)|(segmentation==1), segmentation^1, segmentation)     


    # Check that all segmentation blobs are not smaller than minSize
    if minSize is not None:
        # Label each segmentation patch
        segmentation_labelled = label(segmentation,connectivity=1)    
        # Take care of periodic BC for patches
        if periodicDomain:
            for k in range(segmentation_labelled.shape[0]):
                if segmentation_labelled[k, 0] > 0 and segmentation_labelled[k, -1] > 0:
                    segmentation_labelled[segmentation_labelled == segmentation_labelled[k, -1]] = segmentation_labelled[k, 0]
            for k in range(segmentation_labelled.shape[1]):
                if segmentation_labelled[0, k] > 0 and segmentation_labelled[-1, k] > 0:
                   segmentation_labelled[segmentation_labelled == segmentation_labelled[-1, k]] = segmentation_labelled[0, k]        
        blobs, counts = unique_nonzero(segmentation_labelled,return_counts=True)    
        # Set blobs smaller minSize to 0 (no CP)
        countsSuff_bool = counts >= minSize
        blobsSuff = blobs[countsSuff_bool]
        countsSuff = counts[countsSuff_bool]
        blobsSmall = np.delete(blobs, countsSuff_bool)    
        segmentationSmall = np.isin(segmentation_labelled,blobsSmall)     
        segmentation[segmentationSmall] = 0       

        # fig, ax = plt.subplots(figsize=(10,10))
        # cmap = plt.cm.nipy_spectral  
        # ax.imshow(segmentation_labelled, cmap=cmap)
        # ax.set_title('Segmentation labelled')
        # plt.show()    
    
        if patchCheck:
            # Check that all segmentation blobs are divergent (div > 0), convergent at the boundary (div < 0)
            # and not fuzzy (U/sqrt(A) > 40)
            dx = dy = horResolution
            # Pad u and v, apply a central difference scheme to compute the derivatives, and compute the divergence
            u_pad = filters.gaussian(np.pad(u,((0,0),(1,1)),mode='wrap'), sigma=1.0)
            v_pad = filters.gaussian(np.pad(v,((1,1),(0,0)),mode='wrap'), sigma=1.0)    
            dudx = (u_pad[:, 2:] - u_pad[:, :-2]) / (2*dx)
            dvdy = (v_pad[2:, :] - v_pad[:-2, :]) / (2*dy)
            div = dudx + dvdy
            div_noGf = np.ma.masked_where(div < np.mean(div)-1.645*np.std(div), div) # z = 1.645 -> 95th percentile
            
            # Loop over all remaining patches and check whether all criterions are fulfilled; if not set the segmentation to 0 (no CP)
            i = 0
            for blob in blobsSuff:
                pixelBlob = segmentation_labelled == blob
                boundaryThick_blob_bool = find_boundaries(pixelBlob, connectivity=1, mode='thick', background=0)
                boundaryOuter_blob_bool = np.where(pixelBlob==True,False,boundaryThick_blob_bool)
                #boundaryOuter_blob_bool = find_boundaries(pixelBlob, connectivity=1, mode='outer', background=0)
                blob_div = np.ma.masked_where(np.where(boundaryThick_blob_bool==True,False,pixelBlob)==False,div_noGf)
                #print("Blob div: " + str(np.nanmean(blob_div)))
                #print("Boundary div: " + str(np.nanmean(boundaryThick_blob_bool*div)))
                if blob_div.mask.all():
                    notdiv = False
                else:
                    notdiv = np.nanmean(blob_div) <= 0
                boundaryThickDiv = np.ma.masked_where(boundaryThick_blob_bool==False,boundaryThick_blob_bool*div)
                #notconv = np.nanmean(np.ma.masked_where(div > np.mean(div)+2*np.std(div),boundaryThickDiv)) >= 0
                # z = 0.675 -> 75th percentile
                boundary_conv = np.ma.masked_where((rint_filtered >= 1.0) | (div > np.mean(div)+0.675*np.std(div)),boundaryThickDiv)
                if boundary_conv.mask.all(): # necessary since fully masked arrays lead to ValueError in np.nanmean()
                    notconv = False
                else:
                    notconv = np.nanmean(boundary_conv) >= 0
                toofuzzy = np.sum(boundaryOuter_blob_bool)/np.sqrt(countsSuff[i]) > fuzzThresh              
                if notdiv or notconv or toofuzzy:
                    # print("Blob"+str(blob)+" failed patchCheck: "+str(notdiv)+", "+str(notconv)+", "+str(toofuzzy))
                    segmentation[pixelBlob] = 0
                i += 1 
    
    # Plot segmentation
    plot_seg = False
    if plot_seg:
        fig, ax = plt.subplots()
        cmap = plt.cm.gray
        cmap.set_bad(color='red')          
        ax.imshow(np.ma.masked_where(w > np.mean(w) + 2*np.std(w), segmentation), cmap=cmap)
        ax.set_title('kmeans segmentation')
        ax.axis('off')
        plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
        plt.show()
        
    return segmentation 











