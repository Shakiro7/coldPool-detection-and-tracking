#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:37:35 2022

@author: jannik
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
from skimage import filters
from skimage.measure import find_contours
from dataloader import DataLoader



# Timesteps to be analyzed
start = 610
end = 660


# Dataset   
path = ("/home/jannik/PhD/Programming/gust_front/Romain_data/cp-detection/"
        "diurnal4K_200m/diurnal4K_200m_240x240km2.nc")
ds = nc.Dataset(path,mode="r")




# Fields to plot
# =============================================================================
postprocessingDict = {
    "tv": False,
    "rint": False,
    "showDynGustFront": False, # for tv and rint fields
    
    "rintContouredTv": True,
    "rintContouredW": True,
    "rintContouredWMasked": False
}
save = True
# =============================================================================


# Set seaborn background formatting
sns.set()  

for i in range(end-start):
                    
    print('Timestep ' + str(i+1) + ' / ' + str(end-start))

    # Create DataLoader object    
    dataloader = DataLoader(ds,start+i)
    
         

    # Plot specified fields
    if postprocessingDict["showDynGustFront"]:
        w = filters.gaussian(dataloader.getW(), sigma=2.0)
        w_mean = np.mean(w)
        w_std = np.std(w)
                                         
        if postprocessingDict["tv"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis
            cmap.set_bad(color='red')    
            im=ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), filters.gaussian(dataloader.getTv(), sigma=1.0)), cmap=cmap)
            ax.set_title('Virtual temperature @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if save:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_tv.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["rint"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis
            cmap.set_bad(color='red')    
            im=ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), filters.gaussian(dataloader.getRint(), sigma=2.0)), cmap=cmap)
            ax.set_title('Surface rain intensity @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if save:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_rint.png",bbox_inches='tight')            
            plt.show()
    
    else:
            
        if postprocessingDict["tv"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis  
            im=ax.imshow(filters.gaussian(dataloader.getTv(), sigma=1.0), cmap=cmap)
            ax.set_title('Virtual temperature @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if save:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_tv.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["rint"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis   
            im=ax.imshow(filters.gaussian(dataloader.getRint(), sigma=2.0), cmap=cmap)
            ax.set_title('Surface rain intensity @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if save:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_rint.png",bbox_inches='tight')
            plt.show()


    if postprocessingDict["rintContouredTv"]:    
        # Find contours at a constant value of 1 and 2 mm/h
        contours1 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 1)
        contours2 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 2)
        contours5 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 5)
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots(figsize=(10,10))
        cmap = plt.cm.viridis  
        #cmap.set_bad(color='red')
        im = ax.imshow(filters.gaussian(dataloader.getTv(), sigma=1.0), cmap=cmap)
        fig.colorbar(im)     
        for contour in contours1:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        for contour in contours2:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='purple')
        for contour in contours5:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='black')                 
        ax.set_title('Rint contoured tv @ timestep ' + str(dataloader.getTimestep()))  
        if save:
            plt.savefig("Plots/"+str(dataloader.getTimestep())+"_rintContouredTv.png",bbox_inches='tight')
        plt.show() 
    

    if postprocessingDict["rintContouredW"]:    
        # Find contours at a constant value of 1 and 2 mm/h
        contours1 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 1)
        contours2 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 2)
        contours5 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 5)
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots(figsize=(10,10))
        cmap = plt.cm.viridis  
        #cmap.set_bad(color='red')
        im = ax.imshow(filters.gaussian(dataloader.getW(), sigma=2.0), cmap=cmap)
        fig.colorbar(im)     
        for contour in contours1:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        for contour in contours2:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='purple')
        for contour in contours5:
            if contour.shape[0] > 1:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='black') 
        ax.set_title('Rint contoured w @ timestep ' + str(dataloader.getTimestep()))  
        if save:
            plt.savefig("Plots/"+str(dataloader.getTimestep())+"_rintContouredW.png",bbox_inches='tight')
        plt.show() 


    if postprocessingDict["rintContouredWMasked"]:    
        # Find contours at a constant value of 1 and 2 mm/h
        contours1 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 1)
        contours2 = find_contours(filters.gaussian(dataloader.getRint(), sigma=2.0), 2)
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots(figsize=(10,10))
        cmap = plt.cm.viridis  
        cmap.set_bad(color='red')
        im = ax.imshow(np.ma.masked_where(dataloader.getW()<0,filters.gaussian(dataloader.getW(), sigma=2.0)), cmap=cmap)
        fig.colorbar(im)     
        for contour in contours1:
            if contour.shape[0] > 100:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='purple')
        for contour in contours2:
            if contour.shape[0] > 100:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='black') 
        ax.set_title('Rint contoured w masked @ timestep ' + str(dataloader.getTimestep()))  
        if save:
            plt.savefig("Plots/"+str(dataloader.getTimestep())+"_rintContouredWMasked.png",bbox_inches='tight')
        plt.show() 


    # Delete current dataloader object
    del dataloader

    







