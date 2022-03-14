#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:09:41 2022

@author: jannik
"""

import numpy as np
import copy
from datetime import datetime
import netCDF4 as nc
import matplotlib.pyplot as plt
from dataloader import DataLoader
from rain import RainField
from segmentation import segmentDomain
from coldpools import ColdPoolField
from utils import createMarkers, unique_nonzero
from postprocessing import plotFields, plotCpStats, plotFamilyStats, plotDomainStats
from postprocessing import createCpDf, createFamilyDf, createDomainStatsDf 
from postprocessing import exportDfs, exportFields



# Timesteps to be analyzed
start = 429
end = 1081

# Dataset
simulation = "rce0K_200m"   
path = ("/home/jannik/PhD/Programming/gust_front/Romain_data/cp-detection/"+
        simulation+"/"+simulation+"_240x240km2.nc")
ds = nc.Dataset(path,mode="r")


"""
SETUP INFORMATION

rintThresh: Min. surface rain intensity to be considered part of a rain patch.
    Default is 1 mm/h to be less restrictive and enhance the detection of relations between cold 
    pools. Can be changed to 2 mm/h if not robust enough. Furthermore, rainPatchMinSize can be
    increased to overcome stability issues.
mergeThresh: Proportion of a cold pool that needs to be overruled by another cold pool to get
    merged and be assigned a new cold pool ID together. If 1.0 no merging of cold pools takes
    place. Don't choose values < 0.5 as they can cause problems in identifying the predator 
    during merges. The predator is the overruling cold pool that inherits its family to the 
    new formed cold pool.
    Default is 1.0.
rainPatchMinSize: Min. number of adjacent pixels above rintThresh to become a rain patch.
    Default is 1 km² (here 25 pixels).
    Smaller values can lead to finer dissipation behaviour, but do increase the 
    computation time. Very small values ( e.g. 0) may also lead to artificially high
    rain patch numbers, since also individual pixels close to other rain patches get individual
    IDs. Higher numbers can improve the robustness and speed up the computation, but
    lead to coarse dissipation behaviour and an underestimation of cold pool numbers.
    Drager & van den Heever [2017] used a min. area of 8 km² in their algorithm.
dissipationThresh: Min. number of time steps a rain patch is kept (still gets a marker) during 
    its dissipation (no active rain anymore and the segmentation does not allow the full patch 
    as cold pool anymore).
    Default is 0 (dissipating rain patches are kept as long as they are not dissipated, meaning 
    that the segmentation does not allow the whole patch anymore; so no "bonus" if the number of 
    time steps they were dissipating is below the dissipationThresh; However, cold pools that
    were still non-dissipating the last time step can dissapear without changing into the diss-
    ipating state before. The threshold can be used to overcome this (real cold pools do not 
    dissapear without dissipation). A value of 3 seems to improve dissipation behavior without
    adverse effects.
        
"""

# Setup
rintThresh = 2          # mm/h
mergeThresh = 1.0       # overlap for merge
rainPatchMinSize = 25   # min. no. of pixel
dissipationThresh = 3   # number of time steps
periodicDomain = True

# Post-processing Options
# =============================================================================
postprocessingDict = {
    "simulation_name": simulation,
    # Domain statistics over time ---------------------------------------------
    "domain": False,
    "save_domain": False,
    # Fields ------------------------------------------------------------------
    "labeledCps": True,
    "labeledCpsNonDiss": False,
    "stateLabels": True,
    "labeledFamilies": True,
    "labeledFamiliesNonDiss": False,
    "tv": False,
    "rint": False,
    "showDynGustFront": True, # in the above fields
    "save_fields": True,
    # Cold pool & family statistics -------------------------------------------
    "cp": True,
    "family": True,
    "save_statistics": True,
    
    # Export of dataframes ----------------------------------------------------
    "export_domainDf": True, # domain needs to be True as well
    "export_cpDf": True,
    "export_familyDf": True,
    
    # Export of fields (as compressed single file for each tstep)--------------
    "export_rawDataMl": True,
    "export_analysisData": True   
}
# =============================================================================








# Store and print the starting time
tstart = datetime.now()
print("Start time: " + str(tstart))

# Create list to store RainFields
rainfield_list = []

for i in range(end-start):
                    
    print('Timestep ' + str(i+1) + ' / ' + str(end-start))

    # Create DataLoader object    
    dataloader = DataLoader(ds,start+i)
    
    if i == 0:
        # Create RainField and add to rainfield list
        rainfield = RainField(timestep=start+i,
                              rainIntensity=dataloader.getRint(),
                              rainIntensityThreshold=rintThresh,
                              rainPatchMinSize=rainPatchMinSize,
                              periodicDomain=periodicDomain)
        rainfield_list.append(rainfield)
        
        # Segment the domain
        segmentation = segmentDomain(tv=dataloader.getTv(), 
                                     u=dataloader.getU(), 
                                     v=dataloader.getV(), 
                                     w=dataloader.getW(), 
                                     rint=dataloader.getRint())

        # Create markers from rain patches
        markers, segmentation = createMarkers(rainfield_list=rainfield_list,
                                              rainPatchList=RainField.rainpatch_list,
                                              segmentation=segmentation,
                                              dataset=ds,
                                              dissipationThresh=dissipationThresh,
                                              periodicDomain=periodicDomain)
        
        # Create ColdPoolField and add to coldpoolfield list
        coldpoolfield = ColdPoolField(timestep=start+i,
                                      markers=markers,
                                      rainPatchList=RainField.rainpatch_list,
                                      rainMarkers=rainfield.getRainMarkers(),
                                      dataloader=dataloader,
                                      mask=segmentation,
                                      periodicDomain=periodicDomain,
                                      domainStats=postprocessingDict["domain"],
                                      fillOnlyBackgroundHoles=False,
                                      mergeThreshold=mergeThresh)                                      
        coldpoolfield_temp = copy.deepcopy(coldpoolfield)   
    
    else:
        # Create RainField and add to rainfield list
        rainfield = RainField(timestep=start+i,
                              rainIntensity=dataloader.getRint(),
                              rainMarkersOld=rainfield_list[i-1].getRainMarkers(),
                              oldCps=coldpoolfield_temp.getLabeledCps(),
                              rainIntensityThreshold=rintThresh,
                              rainPatchMinSize=rainPatchMinSize,
                              periodicDomain=periodicDomain)    
        rainfield_list.append(rainfield)
        
        # Segment the domain
        segmentation = segmentDomain(tv=dataloader.getTv(), 
                                     u=dataloader.getU(), 
                                     v=dataloader.getV(), 
                                     w=dataloader.getW(), 
                                     rint=dataloader.getRint(), 
                                     oldCps=coldpoolfield_temp.getLabeledCps())            
        
        # Combine the markers from rain and old cold pools and adapt the segmentation 
        # in case old cold pools are still active
        markers, segmentation = createMarkers(rainfield_list=rainfield_list,
                                              rainPatchList=RainField.rainpatch_list,
                                              oldCps=coldpoolfield_temp.getLabeledCps(),
                                              coldPoolList=ColdPoolField.coldpool_list,
                                              segmentation=segmentation,
                                              dataset=ds,
                                              dissipationThresh=dissipationThresh,
                                              periodicDomain=periodicDomain)
        
        # Create ColdPoolField and add to coldpoolfield list
        coldpoolfield = ColdPoolField(timestep=start+i,
                                      markers=markers,
                                      rainPatchList=RainField.rainpatch_list,
                                      rainMarkers=rainfield.getRainMarkers(),
                                      dataloader=dataloader,
                                      mask=segmentation,
                                      oldCps=coldpoolfield_temp.getLabeledCps(),
                                      periodicDomain=periodicDomain,
                                      domainStats=postprocessingDict["domain"],
                                      fillOnlyBackgroundHoles=False,
                                      mergeThreshold=mergeThresh)  
        coldpoolfield_temp = copy.deepcopy(coldpoolfield)     

        

    # Plot specified fields
    plotFields(postprocessingDict=postprocessingDict,
               dataloader=dataloader,
               coldpoolfield=coldpoolfield)
    
    # Export fields if specified
    exportFields(postprocessingDict=postprocessingDict,
                 dataloader=dataloader,
                 coldpoolfield=coldpoolfield)    
    
    # Delete current dataloader object
    del dataloader

    


# Plot time-dependent domain statistics
plotDomainStats(postprocessingDict=postprocessingDict,
                domainStatsDict=ColdPoolField.domainstats_dict)

# Plot statistics for individual CPs (cold pool level) 
plotCpStats(postprocessingDict=postprocessingDict,
            coldpool_list=ColdPoolField.coldpool_list,
            rainpatch_list=RainField.rainpatch_list)

# Plot statistics for CP families (family level) 
plotFamilyStats(postprocessingDict=postprocessingDict,
                coldpoolfamily_list=ColdPoolField.coldpoolfamily_list,
                coldpool_list=ColdPoolField.coldpool_list,
                rainpatch_list=RainField.rainpatch_list)



# Create dataframes and export them if specified
domainStats_df = createDomainStatsDf(domainStatsDict=ColdPoolField.domainstats_dict)

cp_df = createCpDf(coldpool_list=ColdPoolField.coldpool_list,
                   rainpatch_list=RainField.rainpatch_list)

family_df = createFamilyDf(coldpoolfamily_list=ColdPoolField.coldpoolfamily_list,
                           coldpool_list=ColdPoolField.coldpool_list,
                           rainpatch_list=RainField.rainpatch_list)

exportDfs(postprocessingDict=postprocessingDict,
          domainStats_df=domainStats_df,
          cp_df=cp_df,
          family_df=family_df)

# Store the end time
tend = datetime.now()
print("End time: " +str(tend))
print("Execution time [s]: " + str((tend-tstart).seconds))


