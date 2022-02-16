#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:09:41 2022

@author: jannik
"""

import numpy as np
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
start = 610
end = 624

# Dataset
simulation = "diurnal4K_200m"   
path = ("/home/jannik/PhD/Programming/gust_front/Romain_data/cp-detection/"+
        simulation+"/"+simulation+"_240x240km2.nc")
ds = nc.Dataset(path,mode="r")


"""
SETUP INFORMATION

rintThresh: Min. surface rain intensity to be considered part of a rain patch.
    Default is 2 mm/h since this is typically required for downdrafts.
rainPatchMinSize: Min. number of adjacent pixels above rintThresh to become a rain patch.
    Default is 1 km² (here 25 pixels).
    Smaller values can lead to finer dissipation behaviour, but do increase the 
    computation time. Very small values ( e.g. 0) may also lead to artificially high
    rain patch numbers, since also individual pixels close to other rain patches get individual
    IDs. Higher numbers can improve the robustness and speed up the computation, but
    lead to coarse dissipation behaviour and an underestimation of cold pool numbers.
dissipationThresh: Min. number of time steps a rain patch is kept (still gets a marker) during 
    its dissipation (no active rain anymore and the segmentation does not allow the full patch 
    as cold pool anymore).
    Default is 0 (dissipating rain patches are kept as long as they are not dissipated, meaning 
    that the segmentation does not allow the whole patch anymore; so no "bonus" if the number of 
    time steps they were dissipating is below the dissipationThresh; realistic since small rain
    patches or their cold pools might dissipate faster than larger ones).    
"""

# Setup
rintThresh = 2          # mm/h
mergeThresh = 1.0       # overlap for merge
rainPatchMinSize = 25   # min. no. of pixel
dissipationThresh = 0   # number of time steps
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
    "save_fields": False,
    # Cold pool & family statistics -------------------------------------------
    "cp": False,
    "family": False,
    "save_statistics": False,
    
    # Export of dataframes ----------------------------------------------------
    "export_domainDf": False, # domain needs to be True as well
    "export_cpDf": False,
    "export_familyDf": False,
    
    # Export of fields (as compressed single file for each tstep)--------------
    "export_rawDataMl": False,
    "export_analysisData": False   
}
# =============================================================================











# Create lists to store RainFields and ColdPoolFields
rainfield_list = []
coldpoolfield_list = []

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
        coldpoolfield_list.append(coldpoolfield)   
    
    else:
        # Create RainField and add to rainfield list
        rainfield = RainField(timestep=start+i,
                              rainIntensity=dataloader.getRint(),
                              rainMarkersOld=rainfield_list[i-1].getRainMarkers(),
                              oldCps=coldpoolfield_list[i-1].getLabeledCps(),
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
                                     oldCps=coldpoolfield_list[i-1].getLabeledCps())            
        
        # Combine the markers from rain and old cold pools and adapt the segmentation 
        # in case old cold pools are still active
        markers, segmentation = createMarkers(rainfield_list=rainfield_list,
                                              rainPatchList=RainField.rainpatch_list,
                                              oldCps=coldpoolfield_list[i-1].getLabeledCps(),
                                              coldPoolList=ColdPoolField.coldpool_list,
                                              segmentation=segmentation,
                                              dissipationThresh=dissipationThresh,
                                              periodicDomain=periodicDomain)
        
        # Create ColdPoolField and add to coldpoolfield list
        coldpoolfield = ColdPoolField(timestep=start+i,
                                      markers=markers,
                                      rainPatchList=RainField.rainpatch_list,
                                      rainMarkers=rainfield.getRainMarkers(),
                                      dataloader=dataloader,
                                      mask=segmentation,
                                      oldCps=coldpoolfield_list[i-1].getLabeledCps(),
                                      periodicDomain=periodicDomain,
                                      domainStats=postprocessingDict["domain"],
                                      fillOnlyBackgroundHoles=False,
                                      mergeThreshold=mergeThresh)  
        coldpoolfield_list.append(coldpoolfield)    

        

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





