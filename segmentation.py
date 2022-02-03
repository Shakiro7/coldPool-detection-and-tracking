#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:21:53 2022

@author: jannik
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import invert01




# Function to segment the domain in either "CP possible" or "CP not possible/allowed"
def segmentDomain(tv, u, v, w, rint, oldCps=None):
    # Assign segmentation input data
    # Virtual temperature input        
    #input_tv = filters.gaussian(tv, sigma=1.0).flatten().reshape(-1,1)
    tv_filtered = filters.gaussian(tv, sigma=1.0)
    input_tv = (tv_filtered-np.mean(tv_filtered)).flatten().reshape(-1,1)
    input_tv = invert01(MinMaxScaler().fit_transform(input_tv))   
    
    # Horizontal velocity input
    #input_uv = MinMaxScaler().fit_transform(filters.gaussian(np.sqrt(u**2+v**2), sigma=1.0).flatten().reshape(-1,1)) 
    u_filtered = filters.gaussian(u, sigma=1.0)
    v_filtered = filters.gaussian(v, sigma=1.0)
    input_uv = np.sqrt((u_filtered-np.mean(u_filtered))**2+(v_filtered-np.mean(v_filtered))**2)
    input_uv = MinMaxScaler().fit_transform(filters.gaussian(input_uv, sigma=1.0).flatten().reshape(-1,1))
    
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
    #segmentation = model.labels_.reshape(tv.shape[0],tv.shape[1])
    
    # Verify and/or match label 1 to cold pools and 0 to the rest
    cpindex = np.where(tv == np.min(tv))    
    if segmentation[cpindex].all() == 0:
        segmentation = np.where((segmentation==0)|(segmentation==1), segmentation^1, segmentation)
    

    # Make sure that all pixel with rint > 2mm/h are segmented as possible CP
    #segmentation = np.where(filters.gaussian(rint, sigma=2.0) >= 2, 1, segmentation)        

    # Segment old CP areas as possible CP if existing
    # if oldCps is not None:
    #     segmentation = np.where(oldCps != 0, 1, segmentation)
        
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











