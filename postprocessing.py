#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:40:50 2022

@author: jannik
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from skimage import filters
import pandas as pd
import seaborn as sns
from utils import unique_nonzero




def plotFields(postprocessingDict,dataloader,coldpoolfield):

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
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_tv.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["rint"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis
            cmap.set_bad(color='red')    
            im=ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), filters.gaussian(dataloader.getRint(), sigma=2.0)), cmap=cmap)
            ax.set_title('Surface rain intensity @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_rint.png",bbox_inches='tight')            
            plt.show()

        if postprocessingDict["labeledCps"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), coldpoolfield.getLabeledCps()), cmap=cmap)
            ax.set_title('Labeled CPs @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledCps.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["labeledCpsNonDiss"]:      
            activeCps = coldpoolfield.getLabeledCpsActive()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), activeCps), cmap=cmap)
            ax.set_title('Labeled CPs without dissipating @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledCpsActive.png",bbox_inches='tight')
            plt.show()

        if postprocessingDict["stateLabels"]:      
            binaryLabels = np.where(coldpoolfield.getLabeledCps() != 0, 1, 0)
            stateLabels = np.where((binaryLabels==1) & (coldpoolfield.getLabeledCpsActive() != 0), 2, binaryLabels)            
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), stateLabels), cmap=cmap)
            ax.set_title('Labeled CP state @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_stateLabels.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["labeledFamilies"]:      
            labeledFamilies = coldpoolfield.getLabeledFamilies()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), labeledFamilies), cmap=cmap)
            ax.set_title('Labeled families @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledFamilies.png",bbox_inches='tight')
            plt.show()

        if postprocessingDict["labeledFamiliesNonDiss"]:      
            activeFamilies = coldpoolfield.getLabeledFamiliesActive()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(np.ma.masked_where(w > (w_mean + 2*w_std), activeFamilies), cmap=cmap)
            ax.set_title('Labeled families without dissipating members @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledFamiliesActive.png",bbox_inches='tight')
            plt.show()
            
    else:
            
        if postprocessingDict["tv"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis  
            im=ax.imshow(dataloader.getTv(), cmap=cmap)
            ax.set_title('Virtual temperature @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_tv.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["rint"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.viridis   
            im=ax.imshow(dataloader.getRint(), cmap=cmap)
            ax.set_title('Surface rain intensity @ timestep ' + str(dataloader.getTimestep()))
            fig.colorbar(im) 
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_rint.png",bbox_inches='tight')
            plt.show()

        if postprocessingDict["labeledCps"]:
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral   
            ax.imshow(coldpoolfield.getLabeledCps(), cmap=cmap)
            ax.set_title('Labeled CPs @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledCps.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["labeledCpsNonDiss"]:      
            activeCps = coldpoolfield.getLabeledCpsActive()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral  
            ax.imshow(activeCps, cmap=cmap)
            ax.set_title('Labeled CPs without dissipating @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledCpsActive.png",bbox_inches='tight')
            plt.show()             

        if postprocessingDict["stateLabels"]:      
            binaryLabels = np.where(coldpoolfield.getLabeledCps() != 0, 1, 0)
            stateLabels = np.where((binaryLabels==1) & (coldpoolfield.getLabeledCpsActive() != 0), 2, binaryLabels)            
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            cmap.set_bad(color='red')    
            ax.imshow(stateLabels, cmap=cmap)
            ax.set_title('Labeled CP state @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_stateLabels.png",bbox_inches='tight')
            plt.show()
            
        if postprocessingDict["labeledFamilies"]:      
            labeledFamilies = coldpoolfield.getLabeledFamilies()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral  
            ax.imshow(labeledFamilies, cmap=cmap)
            ax.set_title('Labeled families @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledFamilies.png",bbox_inches='tight')
            plt.show()

        if postprocessingDict["labeledFamiliesNonDiss"]:      
            activeFamilies = coldpoolfield.getLabeledFamiliesActive()
            fig, ax = plt.subplots(figsize=(10,10))
            cmap = plt.cm.nipy_spectral
            ax.imshow(activeFamilies, cmap=cmap)
            ax.set_title('Labeled families without dissipating members @ timestep ' + str(dataloader.getTimestep()))
            if postprocessingDict["save_fields"]:
                plt.savefig("Plots/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_labeledFamiliesActive.png",bbox_inches='tight')
            plt.show()



def plotCpStats(postprocessingDict,coldpool_list,rainpatch_list):
    
    if postprocessingDict["cp"]:

        cp_df = createCpDf(coldpool_list=coldpool_list,rainpatch_list=rainpatch_list)
                 
        # Set seaborn background formatting
        sns.set()        

        # Plots
        fig, ax = plt.subplots()
        splot = sns.histplot(data=cp_df,x="Max_age",stat="percent", discrete=True)
        splot.set(xlabel="Max. CP age")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_cpMaxAge.png",bbox_inches='tight')
        plt.show()  


        fig, ax = plt.subplots()
        splot = sns.histplot(data=cp_df,x="Generation",stat="percent", discrete=True)
        splot.set(xlabel="CP generation")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_cpGeneration.png",bbox_inches='tight')
        plt.show()                     

        fig, ax = plt.subplots()
        splot = sns.histplot(data=cp_df,x="Max_area",stat="percent",log_scale=True)
        splot.set(xlabel="Max. CP area")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_cpMaxAreaLog.png",bbox_inches='tight')
        plt.show()          


 




def plotFamilyStats(postprocessingDict,coldpoolfamily_list,coldpool_list,rainpatch_list):

    if postprocessingDict["family"]:      
        
        family_df = createFamilyDf(coldpoolfamily_list=coldpoolfamily_list, 
                                   coldpool_list=coldpool_list, 
                                   rainpatch_list=rainpatch_list)

        # Set seaborn background formatting
        sns.set()        

        # Plots
        fig, ax = plt.subplots()
        splot = sns.histplot(data=family_df,x="Max_age",stat="percent", discrete=True)
        splot.set(xlabel="Max. family age")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_familyMaxAge.png",bbox_inches='tight')
        plt.show()         
        

        fig, ax = plt.subplots()
        splot = sns.histplot(data=family_df,x="Generations",stat="percent", discrete=True)
        splot.set(xlabel="Family generations")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_familyGenerations.png",bbox_inches='tight')
        plt.show() 


        fig, ax = plt.subplots()
        splot = sns.histplot(data=family_df,x="No_familyMembers",stat="percent", discrete=True)
        splot.set(xlabel="Family members")
        if postprocessingDict["save_statistics"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_histplot_familyNoFamilyMembers.png",bbox_inches='tight')
        plt.show() 
        
        
        
        
def plotDomainStats(postprocessingDict,domainStatsDict):
    
    if postprocessingDict["domain"]:

        # Set seaborn background formatting
        sns.set() 
        
        domainStats_df = createDomainStatsDf(domainStatsDict)
        
        
        # Plots
        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.coverageCpInFamily,
                        domainStats_df.coverageCpNotInFamily,
                        labels=['CPs with family', 'CPs without family'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Domain coverage")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-domainCoverage-cpFamilyHue.png",
                        bbox_inches='tight')        
        plt.show() 


        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.noCpInFamily,
                        domainStats_df.noCpNotInFamily,
                        labels=['CPs with family', 'CPs without family'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-number-cpFamilyHue.png",
                        bbox_inches='tight')        
        plt.show() 


        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.coverageCpActive,
                        domainStats_df.coverageCpDissipating,
                        labels=['active CPs', 'dissipating CPs'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Domain coverage")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-domainCoverage-cpStateHue.png",
                        bbox_inches='tight')        
        plt.show() 


        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.noCpActive,
                        domainStats_df.noCpDissipating,
                        labels=['active CPs', 'dissipating CPs'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-number-stateHue.png",
                        bbox_inches='tight')        
        plt.show() 


        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.noCpIsolated,
                        domainStats_df.noCpIntersecting,
                        labels=['isolated CPs', 'intersecting CPs'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-number-contactHue.png",
                        bbox_inches='tight')        
        plt.show() 


        fig, ax = plt.subplots()
        im=ax.stackplot(domainStats_df.timestep, domainStats_df.noFamiliesActive,
                        domainStats_df.noFamiliesInactive,
                        labels=['active families', 'inactive families'])           
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Number")
        ax.legend(loc='upper left')
        if postprocessingDict["save_domain"]:
            plt.savefig("Plots/"+postprocessingDict["simulation_name"]+"_stackplot_timestep-number-familiesHue.png",
                        bbox_inches='tight')        
        plt.show() 




def createCpDf(coldpool_list,rainpatch_list):
    
    cp_df = pd.DataFrame(list(zip([obj.getId() for obj in coldpool_list],
                                  [obj.getOrigin() for obj in coldpool_list],
                                  [obj.getArea() for obj in coldpool_list],
                                  [obj.getAge() for obj in coldpool_list],
                                  [obj.getTvMean() for obj in coldpool_list],
                                  [obj.getIntersecting() for obj in coldpool_list],
                                  [len(obj.getParents()) for obj in coldpool_list],
                                  [len(obj.getChildren()) for obj in coldpool_list],
                                  [len(obj.getPatrons()) for obj in coldpool_list],
                                  [obj.getGeneration() for obj in coldpool_list],
                                  [obj.getFamily() for obj in coldpool_list],
                                  )),
                         columns=['CP_ID','Origin','Max_area','Max_age',
                                  'Initial_tv_mean','Initial_contact',
                                  'No_parents','No_children','No_patrons',
                                  'Generation','Family_ID'])
    
    cp_df['Initial_contact'] = pd.Categorical(cp_df.Initial_contact)
    
    # Identify the parent rain duration and the initial rint mean for each cold pool
    rainDuration_list = []
    initialRintMean_list = []    
    for cp in cp_df.CP_ID:        
        for i, obj in enumerate(rainpatch_list):
            if obj.getId() == cp:
                index = i
                break        
        rainDuration_list.append(rainpatch_list[index].getAge())
        initialRintMean_list.append(rainpatch_list[index].getRintMean())
        
    # Add both variables to the cold pool dataframe
    cp_df['Initial_rint_mean'] = initialRintMean_list
    cp_df['Rain_duration'] = rainDuration_list

    return cp_df
    



def createFamilyDf(coldpoolfamily_list,coldpool_list,rainpatch_list):
   
    family_df = pd.DataFrame(list(zip([obj.getId() for obj in coldpoolfamily_list],
                                      [obj.getStart() for obj in coldpoolfamily_list],
                                      [obj.getAge() for obj in coldpoolfamily_list],
                                      [obj.getFounder() for obj in coldpoolfamily_list],
                                      [len(obj.getFamilyMembers()) for obj in coldpoolfamily_list],
                                      [obj.getGenerations() for obj in coldpoolfamily_list],
                                      )),
                             columns=['Family_ID','Start_tstep','Max_age','Founder',
                                      'No_familyMembers','Generations'])
    
    # Identify the properties of the family founder(s)
    founderInitialRintMean_list = []
    founderRainDuration_list = []
    founderInitialTvMean_list = []    
    for indexFamily in range(len(family_df.Family_ID)):
        founderInitialRintMean = 0
        founderRainDuration = 0
        founderInitialTvMean = 0
        for founder in family_df.Founder[indexFamily]:
            for i, obj in enumerate(rainpatch_list):
                if obj.getId() == founder:
                    index = i
                    break
            founderInitialRintMean += rainpatch_list[index].getRintMean()
            founderRainDuration += rainpatch_list[index].getAge()
            for i, obj in enumerate(coldpool_list):
                if obj.getId() == founder:
                    index = i
                    break
            founderInitialTvMean += coldpool_list[index].getTvMean()
        
        founderInitialRintMean_list.append(founderInitialRintMean/len(family_df.Founder[indexFamily]))
        founderRainDuration_list.append(founderRainDuration/len(family_df.Founder[indexFamily]))
        founderInitialTvMean_list.append(founderInitialTvMean/len(family_df.Founder[indexFamily]))
        
    # Add the variables to the family dataframe
    family_df['Initial_rint_mean'] = founderInitialRintMean_list
    family_df['Rain_duration'] = founderRainDuration_list
    family_df['Initial_tv_mean'] = founderInitialTvMean_list

    return family_df




def createDomainStatsDf(domainStatsDict):
    
    domainStats_df = pd.DataFrame(domainStatsDict)

    return domainStats_df



  
def exportDfs(postprocessingDict,domainStats_df=None,cp_df=None,family_df=None):    
    
    if postprocessingDict["export_domainDf"] and domainStats_df is not None:
        domainStats_df.to_pickle("Dataframes/"+postprocessingDict["simulation_name"]+"_domainStats_df.pkl")

    if postprocessingDict["export_cpDf"] and cp_df is not None:
        cp_df.to_pickle("Dataframes/"+postprocessingDict["simulation_name"]+"_cp_df.pkl")
        
    if postprocessingDict["export_familyDf"] and family_df is not None:
        family_df.to_pickle("Dataframes/"+postprocessingDict["simulation_name"]+"_family_df.pkl")




def exportFields(postprocessingDict,dataloader,coldpoolfield):
    
    if postprocessingDict["export_rawDataMl"]:
        binaryLabels = np.where(coldpoolfield.getLabeledCps() != 0, 1, 0)
        stateLabels = np.where((binaryLabels==1) & (coldpoolfield.getLabeledCpsActive() != 0), 2, binaryLabels)
        np.savez_compressed("Arrays/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_rawDataMl", 
                            rint=dataloader.getRint(), 
                            tb=dataloader.getTb(),
                            binaryLabels=binaryLabels, 
                            stateLabels=stateLabels)

    if postprocessingDict["export_analysisData"]:
        np.savez_compressed("Arrays/"+str(dataloader.getTimestep())+"_"+postprocessingDict["simulation_name"]+
                            "_analysisData", 
                            labeledCps=coldpoolfield.getLabeledCps(),
                            labeledCpsNonDiss=coldpoolfield.getLabeledCpsActive(),
                            labeledFamilies=coldpoolfield.getLabeledFamilies(), 
                            labeledFamiliesNonDiss=coldpoolfield.getLabeledFamiliesActive())





      