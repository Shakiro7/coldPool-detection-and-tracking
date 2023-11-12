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
start = 0
end = 19

# Dataset
simulation = "diu4K_t761-780"   
path = ("/home/jannik/PhD/Programming/gust_front/Romain_data/cp-detection/"+
        "diurnal4K_200m/"+simulation+".nc")
ds = nc.Dataset(path,mode="r")


"""
SETUP INFORMATION

rintThresh: Min. surface rain intensity to be considered part of a rain patch.
    Default is 2 mm/h. Can be changed to 1 mm/h if not fine enough.
rainPatchMinSize: Min. number of adjacent pixels above rintThresh to become a rain patch.
    Default is 2 km² (here 50 pixels).
    Smaller values can lead to finer dissipation behaviour, but do increase the 
    computation time. Very small values ( e.g. 0) may also lead to artificially high
    rain patch numbers, since also individual pixels close to other rain patches get individual
    IDs. Higher numbers can improve the robustness and speed up the computation, but
    lead to coarse dissipation behaviour and an underestimation of cold pool numbers.
    Drager & van den Heever [2017] used a min. area of 8 km² in their algorithm.
dissipationThresh: Min. number of time steps a cold pool receives markers during 
    its dissipation (no active rain anymore and the segmentation does not allow the full last 
    rain patch as cold pool anymore).
    Default is 3. Prevents cold pools from disappearing without dissipation stage.
    Helps to prevent cold pools from occupying the region of former cold pools when
    parts of their region are still segmented as potential cold pool by the segmentation.
    A value of 3 seems to improve dissipation behavior without adverse effects.
    Not applied if a cold pool was only there for one time step. A cold pool needs to be
    there for at least two time steps to be saved by the dissipationThresh.
coldPoolMinSize: Min. number of adjacent cold pool pixels to become a cold pool.
    Default is rainPatchMinSize, which requires cold pools to be fully developed since
    very cold regions within rain patches are dropped as long as no gust front hasn't
    developed and kept up with the rain patch.
    Smaller patches in the segmentation will be dropped. Checked again for individual
    cold pools after the labeled cold pool field has been created. 
onlyNew: Only affects the starting time step. If "True", only cold pools that are in the 
    beginning of their lifecycle will be detected. If "False", all cold pools that can be
    linked with rain events will be detected.
    Default is False.
patchCheck: If "True", every patch segmented as potential cold pool by the segmentation
    will be checked and only kept, if the center is divergent (div > 0), the boundary
    is convergent (div < 0) and the ratio of perimeter and area is smaller or equal to
    fuzzThresh (only relevant for time steps without cold pools or rainfall).
    Default is True.
fuzzThresh: Only relevant if patchCheck = True. Threshold for the fuzzyness ratio perimeter/sqrt(area).
    Default is 40. 
    Since the kmeans algorithm applied in the segmentation part does always find two groups
    (cold pool and no cold pool) it has problems in time steps without any cold pool activity.
    However, since the boundaries between the two groups are very noisy in this case, the fuzzyness
    ratio may help to drop these patches.
    Note: As long as no rain patch that satisfies the above criteria is present, no cold pools are 
    created from the potential cold pool patches identified by the segmentation.
horResolution: Only relevant if patchCheck = True. Horizontal resolution in lowest model layer in meters.
    Default is 200 m.    
        
"""

# Setup
rintThresh = 2                              # mm/h
rainPatchMinSize = 50                       # min. no. of pixel
dissipationThresh = 3                       # number of time steps
coldPoolMinSize = 1 * rainPatchMinSize      # min. no. of pixel
onlyNew = False                             # True or False
patchCheck = True                           # True or False (only possible if coldPoolMinSize is not None)
fuzzThresh = 40                             # max. perimeter/sqrt(area) ratio (only possible if patchCheck is True)
horResolution = 200                         # m (only needed if patchCheck=True)
periodicDomain = True                       # True or False

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
    "cp": True,
    "family": True,
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
                                     rint=dataloader.getRint(),
                                     minSize=coldPoolMinSize,
                                     patchCheck=patchCheck,
                                     horResolution=horResolution,
                                     fuzzThresh=fuzzThresh,
                                     periodicDomain=periodicDomain)

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
                                      minSize=coldPoolMinSize,
                                      onlyNew=onlyNew,
                                      periodicDomain=periodicDomain,
                                      domainStats=postprocessingDict["domain"],
                                      fillOnlyBackgroundHoles=False)                                     
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
                                     minSize=coldPoolMinSize, 
                                     patchCheck=patchCheck,
                                     horResolution=horResolution,
                                     fuzzThresh=fuzzThresh,
                                     periodicDomain=periodicDomain)            
        
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
                                      minSize=coldPoolMinSize,
                                      onlyNew=onlyNew,
                                      oldCps=coldpoolfield_temp.getLabeledCps(),
                                      periodicDomain=periodicDomain,
                                      domainStats=postprocessingDict["domain"],
                                      fillOnlyBackgroundHoles=False)  
        coldpoolfield_temp = copy.deepcopy(coldpoolfield)     

        

    # Plot specified fields
    plotFields(postprocessingDict=postprocessingDict,
               dataloader=dataloader,
               coldpoolfield=coldpoolfield)
    
    # Export fields if specified
    exportFields(postprocessingDict=postprocessingDict,
                 dataloader=dataloader,
                 coldpoolfield=coldpoolfield,
                 rainMarkers=rainfield.getRainMarkers())    
    
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


