#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:03:30 2022

@author: jannik
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed, find_boundaries
from skimage import filters
from scipy import ndimage as ndi
from skimage.measure import label
from utils import unique_nonzero, searchBlobMin, checkBlobContact, scale01, invert01
from tracking import mergeNew






class ColdPool:
    
    def __init__(self,identificationNumber,origin,startTimestep,area,virtualTemp_mean,parents,merged,age=1,dissipating=0,intersecting=False,
                 generation=1,family=None):
        
        self.__id = identificationNumber
        self.__origin = origin
        self.__startTimestep = startTimestep
        self.__area = area
        self.__tv_mean = virtualTemp_mean
        self.__age = age
        self.__dissipating = dissipating
        self.__intersecting = intersecting
        self.__generation = generation
        self.__parents = parents
        self.__merged = merged
        self.__children = []
        self.__patrons = []
        self.__family = family
        

    def __del__(self):
        """
        Deletes the ColdPool object
        """

    def getId(self):
        
        return self.__id
    
    def getOrigin(self):
        
        return self.__origin
    
    def getStart(self):
        
        return self.__startTimestep
    
    def getArea(self):
        
        return self.__area

    def getTvMean(self):
        
        return self.__tv_mean
    
    def getAge(self):
        
        return self.__age
    
    def getState(self):
        
        return self.__dissipating

    def getIntersecting(self):
        
        return self.__intersecting

    def getGeneration(self):
        
        return self.__generation
    
    def getParents(self):
        
        return self.__parents
    
    def getMerged(self):
        
        return self.__merged 

    def getChildren(self):
        
        return self.__children

    def getPatrons(self):
        
        return self.__patrons

    def getFamily(self):
        
        return self.__family 



    def setArea(self,newArea):
        """
        Allows to modify the area of an existing ColdPool
        """          
        self.__area = newArea

    def setAge(self):
        """
        Increases the age of an existing ColdPool by one
        """          
        self.__age += 1

    def setState(self):
        """
        Increases the time steps that a ColdPool is dissipating by one
        """          
        self.__dissipating += 1

    def setChildren(self,child):
        """
        Adds the ID of a child ColdPool to the corresponding parent ColdPool
        """          
        self.__children.append(child)

    def setPatrons(self,patron):
        """
        Adds the ID of a patron ColdPool to the corresponding recipient ColdPool
        """          
        self.__patrons.append(patron)

    def setFamily(self,newFamilyId):
        """
        Sets or changes the family of a ColdPool
        """          
        self.__family = newFamilyId



class ColdPoolFamily:
    
    def __init__(self,identificationNumber,founder,startTimestep,familyMembers,age=1,generations=2):
        
        self.__id = identificationNumber
        self.__founder = founder
        self.__startTimestep = startTimestep
        self.__familyMembers = familyMembers
        self.__age = age
        self.__generations = generations
        
    def __del__(self):
        """
        Deletes the ColdPoolFamily object
        """

    def getId(self):
        
        return self.__id

    def getFounder(self):
        
        return self.__founder
    
    def getStart(self):
        
        return self.__startTimestep

    def getFamilyMembers(self):
        
        return self.__familyMembers
    
    def getAge(self):
        
        return self.__age
    
    def getGenerations(self):
        
        return self.__generations

    def setFounder(self,newFoundersList):
        """
        Adds a new founder/founders to the ColdPoolFamily
        """          
        self.__founder.extend(newFoundersList)
    
    def setFamilyMembers(self,newMember):
        """
        Adds a new familyMember or new family members to the ColdPoolFamily
        """          
        if isinstance(newMember, list):
            self.__familyMembers.extend(newMember)
        else:
            self.__familyMembers.append(newMember)

    def setAge(self,age=None):
        """
        If no age is specified, the age of an existing ColdPoolFamily is increased by one.
        Otherwise, the specified age is assigned as new age.
        """          
        if age is not None:
            self.__age = age
        else:
            self.__age += 1

    def setGenerations(self,generations=None):
        """
        If generations is not specified, the generations of an existing ColdPoolFamily are increased by one.
        Otherwise, the specified generations are assigned as new generations.
        """          
        if generations is not None:
            self.__generations = generations
        else:
            self.__generations += 1
            


class ColdPoolField:
    
    # Create empty lists to store all ColdPools (and ColdPoolFamilies)
    # The coldpool_list is a class variable (shared by all ColdPoolFields) and stores all ColdPools for the selected time steps
    coldpool_list = []
    coldpoolfamily_list = []
    domainstats_dict = {
        "timestep": [],
        "coverageCpInFamily": [],
        "coverageCpNotInFamily": [],
        "coverageCpActive": [],
        "coverageCpDissipating": [],
        "noCpInFamily": [],
        "noCpNotInFamily": [],
        "noCpActive": [],
        "noCpDissipating": [],
        "noCpIsolated": [],
        "noCpIntersecting": [],
        "noFamiliesActive": [],
        "noFamiliesInactive": []}
    
    def __init__(self,timestep,markers,rainPatchList,rainMarkers,dataloader,mask,oldCps=None,
                 periodicDomain=True,domainStats=False,fillOnlyBackgroundHoles=False,mergeThreshold=0.5):
        
        self.__tstep = timestep
        markers = markers
        rainPatchList = rainPatchList
        rainMarkers = rainMarkers
        q01 = scale01(dataloader.getQ())
        tv = dataloader.getTv()
        tv01 = scale01(tv)
        w01 = scale01(filters.gaussian(dataloader.getW(), sigma=2.0))
        mask = mask
        labeledCpsOld = oldCps
        periodicBc = periodicDomain
        stats = domainStats
        fillOnlyBackgroundHoles = fillOnlyBackgroundHoles
        mergeThresh = mergeThreshold
      
                    
        # # Plot markers over w field
        # markers_masked = np.ma.masked_array(markers,markers==0)
        # w_masked = np.ma.masked_array(filters.gaussian(dataloader.getW(), sigma=2.0),markers!=0)        
        # fig, ax = plt.subplots(figsize=(10,10))
        # markers_im = ax.imshow(markers_masked,cmap=plt.cm.viridis)
        # cbmarkers = plt.colorbar(markers_im)
        # w_im = ax.imshow(w_masked,cmap=plt.cm.Reds)
        # cbw = plt.colorbar(w_im)        
        # ax.set_title('Markers over w @ timestep ' + str(self.__tstep))
        # cbw.set_label('w [m/s]')
        # cbmarkers.set_label('labeled markers')
        # plt.savefig("Plots/"+str(self.__tstep)+"_markersOverW.png",bbox_inches='tight')
        # plt.show()

        # # Plot markers over tv field
        # tv_masked = np.ma.masked_array(tv,markers!=0)        
        # fig, ax = plt.subplots(figsize=(10,10))
        # markers_im = ax.imshow(markers_masked,cmap=plt.cm.viridis)
        # cbmarkers = plt.colorbar(markers_im)
        # tv_im = ax.imshow(tv_masked,cmap=plt.cm.Reds)
        # cbtv = plt.colorbar(tv_im)        
        # ax.set_title('Markers over tv @ timestep ' + str(self.__tstep))
        # cbtv.set_label('tv [K]')
        # cbmarkers.set_label('labeled markers')
        # plt.savefig("Plots/"+str(self.__tstep)+"_markersOverTv.png",bbox_inches='tight')
        # plt.show()
        
        # Fill the elevation map (here virtualTemp) at markers and mask it
        # In case of periodic BC wrap the whole domain and slice after flooding
        if periodicBc:
            pad_width = (tv.shape[0], tv.shape[1])
            self.__labeledCps = watershed(np.pad(tv01+w01,pad_width,mode='wrap'), np.pad(markers,pad_width,mode='wrap'), 
                                          mask=np.pad(mask,pad_width,mode='wrap'))
            if fillOnlyBackgroundHoles:
                labeledCpsCenter = self.__labeledCps[tv.shape[0]:tv.shape[0]*2, 
                                                     tv.shape[1]:tv.shape[1]*2]
                for cp in unique_nonzero(self.__labeledCps, return_counts=False):
                    # Fill possible holes in the cp if they are 0 (not other cold pools) and only surrounded by the cp itself
                    filled_cp = ndi.binary_fill_holes(self.__labeledCps == cp)
                    filled_cp = filled_cp[tv.shape[0]:tv.shape[0]*2, 
                                          tv.shape[1]:tv.shape[1]*2]
                    if not np.array_equal(filled_cp, (labeledCpsCenter == cp)):
                        filled_cp_zeros = (filled_cp == True) & (labeledCpsCenter == 0)
                        fill_blobs = filled_cp_zeros & (labeledCpsCenter != cp)
                        labeled_fill_blobs = label(fill_blobs)
                        for blob in unique_nonzero(labeled_fill_blobs):
                            boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                            boundary_blob = boundary_blob_bool * labeledCpsCenter
                            if all(unique_nonzero(boundary_blob)==cp):
                                labeledCpsCenter[labeled_fill_blobs==blob] = cp
                self.__labeledCps = labeledCpsCenter
            else:
                for cp in unique_nonzero(self.__labeledCps, return_counts=False):
                    # Fill possible holes in the cp
                    filled_cp = ndi.binary_fill_holes(self.__labeledCps == cp)
                    self.__labeledCps = np.where(filled_cp, cp, self.__labeledCps)
                self.__labeledCps = self.__labeledCps[tv.shape[0]:tv.shape[0]*2, 
                                                      tv.shape[1]:tv.shape[1]*2]
        else:
            self.__labeledCps = watershed(tv01+w01, markers, mask=mask)
            if fillOnlyBackgroundHoles:
                for cp in unique_nonzero(self.__labeledCps, return_counts=False):
                    # Fill possible holes
                    filled_cp = ndi.binary_fill_holes(self.__labeledCps == cp)
                    if not np.array_equal(filled_cp, (self.__labeledCps == cp)):
                        filled_cp_zeros = (filled_cp == True) & (self.__labeledCps == 0)
                        fill_blobs = filled_cp_zeros & (self.__labeledCps != cp)
                        labeled_fill_blobs = label(fill_blobs)
                        for blob in unique_nonzero(labeled_fill_blobs):
                            boundary_blob_bool = find_boundaries(labeled_fill_blobs==blob, connectivity=1, mode='outer', background=0)
                            boundary_blob = boundary_blob_bool * self.__labeledCps
                            if all(unique_nonzero(boundary_blob)==cp):
                                self.__labeledCps[labeled_fill_blobs==blob] = cp
            else:
                for cp in unique_nonzero(self.__labeledCps, return_counts=False):
                    # Fill possible holes in the cp
                    filled_cp = ndi.binary_fill_holes(self.__labeledCps == cp)
                    self.__labeledCps = np.where(filled_cp, cp, self.__labeledCps)
        

        # Merge cold pools based on previous ColdPoolField
        if labeledCpsOld is not None:
            mergedField, mergeDict, mergeList = mergeNew(self.__labeledCps,labeledCpsOld,
                                                         rainMarkers,rainPatchList,
                                                         ColdPoolField.coldpool_list,mergeThresh)
            
            # If merging took place create new ColdPools for those 
            if len(mergeDict["newLabels"]) > 0:
                print("Merged CPs: " + str(len(mergeDict["newLabels"])))
                l = 0
                for cp in mergeDict["newLabels"]:
                    print("CP " + str(mergeDict["predator"][l]) + " overruled CP " + 
                          str(mergeDict["prey"][l]) + ". New label: " + str(mergeDict["newLabels"][l]))
                    cp_region = mergedField == cp
                    predator = mergeDict["predator"][l]
                    # Copy the family related properties from its predator
                    for j, obj in enumerate(ColdPoolField.coldpool_list):
                        if obj.getId() == predator:
                            index_predator = j
                            break                
                    generation = ColdPoolField.coldpool_list[index_predator].getGeneration()
                    cp_parents = ColdPoolField.coldpool_list[index_predator].getParents()
                    family = ColdPoolField.coldpool_list[index_predator].getFamily()
                    state = ColdPoolField.coldpool_list[index_predator].getState()
                    mergers = [mergeDict["predator"][l]] + mergeDict["prey"][l]
                       
                    coldpool = ColdPool(identificationNumber=cp,origin=None,startTimestep=self.__tstep,
                                        area=np.count_nonzero(cp_region),virtualTemp_mean=np.mean(tv[cp_region]),dissipating=state,
                                        parents=cp_parents,merged=mergers,intersecting=checkBlobContact(cp_region, mergedField),
                                        generation=generation,family=family)                   
                    ColdPoolField.coldpool_list.append(coldpool)
                    l += 1

        # Create ColdPools for all unique labels in labeledCps and store them in the list
        cp_labels, cp_counts = unique_nonzero(self.__labeledCps, return_counts=True)
        self.__numberOfColdPools = len(cp_labels)
        n = 0                   
        if labeledCpsOld is not None:
            # Increase the age of all already exisiting (unique) families of the current cold pool field by one 
            if any(obj.getFamily() is not None for obj in ColdPoolField.coldpool_list):
                notNoneFamilies = [x for x in [obj.getFamily() for obj in ColdPoolField.coldpool_list] if x is not None]
                families = unique_nonzero(notNoneFamilies)
                for family in families:
                    ColdPoolField.coldpoolfamily_list[family-1].setAge()
                    
            # Create lists to store child cold pool IDs along with the family which they are born in
            childcp_list = []
            family_list = []
            
            for cp in cp_labels:
                if any(obj.getId() == cp for obj in ColdPoolField.coldpool_list):
                # Only modify existing ColdPools
                    if cp in mergeList:
                        merge = True
                    else:
                        merge = False
                    for i, obj in enumerate(ColdPoolField.coldpool_list):
                        if obj.getId() == cp:
                            index_cp = i
                            break
                    if not merge:
                        ColdPoolField.coldpool_list[index_cp].setAge()
                        if cp_counts[n] > ColdPoolField.coldpool_list[index_cp].getArea():
                            ColdPoolField.coldpool_list[index_cp].setArea(cp_counts[n])
                    # Check if cp is parent, if yes add all children to the cp
                    children_list = []
                    
                    for i, obj in enumerate(rainPatchList):
                         if cp in obj.getParents():
                             index_child = i
                             child = rainPatchList[index_child].getId()
                             children_list.append(child)
                             
                             # Family stuff
                             # Check if a child is not yet assigned to a family
                             if child not in childcp_list:
                                 # Check if child has only the cp as parent
                                 if len(rainPatchList[index_child].getParents()) == 1:
                                     # If cp is the only parent and has no family yet, start a new one
                                     if ColdPoolField.coldpool_list[index_cp].getFamily() is None:
                                         #print("Start founding family")
                                         familyId = len(ColdPoolField.coldpoolfamily_list) + 1
                                         childcp_list.append(child)
                                         family_list.append(familyId)
                                         ColdPoolField.coldpool_list[index_cp].setFamily(familyId)
                                         founder = [cp]
                                         familyMembers = founder + [child]
                                         family = ColdPoolFamily(identificationNumber=familyId, founder=founder, 
                                                                 startTimestep=self.__tstep, familyMembers=familyMembers)
                                         ColdPoolField.coldpoolfamily_list.append(family)
                                         #print("Family " + str(familyId) + " founded successfully")
                                         #print(" Founder: " + str(family.getFounder()))
                                         #print(" Members: " + str(family.getFamilyMembers()))
                                    # If cp is the only parent and has already a family, add the child to it if not yet done
                                     else:
                                         familyLabl = ColdPoolField.coldpool_list[index_cp].getFamily()
                                         if child not in ColdPoolField.coldpoolfamily_list[familyLabl-1].getFamilyMembers():
                                             childcp_list.append(child)
                                             #print("Adding child " + str(child) + " to family " + str(familyLabl))
                                             family_list.append(familyLabl)
                                             ColdPoolField.coldpoolfamily_list[familyLabl-1].setFamilyMembers(child)
                                             # If parent of child is from last family generation, increase family generation
                                             if ColdPoolField.coldpool_list[index_cp].getGeneration() == ColdPoolField.coldpoolfamily_list[familyLabl-1].getGenerations():
                                                 ColdPoolField.coldpoolfamily_list[familyLabl-1].setGenerations()
                                 # If child has more than one parent  
                                 else:
                                    parent_list = rainPatchList[index_child].getParents()
                                    indexParents_list = []
                                    for parent in parent_list:
                                        for j, obj1 in enumerate(ColdPoolField.coldpool_list):
                                            if obj1.getId() == parent:
                                                index_parent = j
                                                indexParents_list.append(index_parent)
                                                break
                                    parentFamilies_list = []
                                    for indexParent in indexParents_list:
                                        parentFamilies_list.append(ColdPoolField.coldpool_list[indexParent].getFamily())
                                    if all(family == None for family in parentFamilies_list):
                                        #print("Start founding family with multiple parents")
                                        familyId = len(ColdPoolField.coldpoolfamily_list) + 1
                                        childcp_list.append(child)
                                        family_list.append(familyId)                                        
                                        for indexParent in indexParents_list:
                                            ColdPoolField.coldpool_list[indexParent].setFamily(familyId)
                                        founder = parent_list
                                        familyMembers = founder + [child]
                                        family = ColdPoolFamily(identificationNumber=familyId, founder=founder, 
                                                                 startTimestep=self.__tstep, familyMembers=familyMembers)
                                        ColdPoolField.coldpoolfamily_list.append(family)                                        
                                        #print("Family " + str(familyId) + " founded successfully")
                                        #print(" Founder: " + str(family.getFounder()))
                                        #print(" Members: " + str(family.getFamilyMembers()))
                                    else:
                                        notNoneParentFamilies = [x for x in parentFamilies_list if x is not None]
                                        uniqueParentFamilies = unique_nonzero(notNoneParentFamilies)
                                        # Find the oldest of the parent families (find the lowest familyId)
                                        oldestFamily = min(uniqueParentFamilies)
                                        if child not in ColdPoolField.coldpoolfamily_list[oldestFamily-1].getFamilyMembers():
                                            #print("Adding child " + str(child) + " to oldest family " + str(oldestFamily))
                                            childcp_list.append(child)
                                            family_list.append(oldestFamily)
                                            ColdPoolField.coldpoolfamily_list[oldestFamily-1].setFamilyMembers(child)
                                            maxGenerations = 1
                                            # Loop over all other parent families and integrate them into the oldest family
                                            if len(uniqueParentFamilies) > 1:
                                                for parentFamily in uniqueParentFamilies:
                                                    if parentFamily != oldestFamily:
                                                        ColdPoolField.coldpoolfamily_list[oldestFamily-1].setFounder(
                                                            ColdPoolField.coldpoolfamily_list[parentFamily-1].getFounder())
                                                        ColdPoolField.coldpoolfamily_list[oldestFamily-1].setFamilyMembers(
                                                            ColdPoolField.coldpoolfamily_list[parentFamily-1].getFamilyMembers())
                                                        if ColdPoolField.coldpoolfamily_list[parentFamily-1].getGenerations() > maxGenerations:
                                                            maxGenerations = ColdPoolField.coldpoolfamily_list[parentFamily-1].getGenerations()
                                            coldpool_list_subset = [ColdPoolField.coldpool_list[index] for index in indexParents_list]
                                            if any(obj2.getGeneration() > maxGenerations for obj2 in coldpool_list_subset):
                                                maxGenerations = max([obj2.getGeneration() for obj2 in coldpool_list_subset]) + 1
                                            else:
                                                maxGenerations += 1
                                            ColdPoolField.coldpoolfamily_list[oldestFamily-1].setGenerations(maxGenerations)

                        
                    if len(children_list) > 0: 
                        for child in children_list:
                            if child not in ColdPoolField.coldpool_list[index_cp].getChildren():
                                 ColdPoolField.coldpool_list[index_cp].setChildren(child)                             
                                 
                    # Check if cp has a patron (= another cp that interacted with the rain patch to keep it going)
                    for i, obj in enumerate(rainPatchList):
                        if (obj.getId() == cp) & (len(obj.getPatrons()) > 0):
                            index_rain = i
                            cp_patrons = rainPatchList[index_rain].getPatrons().copy()
                            for patron in cp_patrons:
                                if patron not in ColdPoolField.coldpool_list[index_cp].getPatrons():
                                    ColdPoolField.coldpool_list[index_cp].setPatrons(patron)
                            break
                        
                    
                else:
                # Create new ColdPool and append it to coldpool_list
                    cp_region = self.__labeledCps == cp
                    generation = 1
                    cp_parents = []
                    family = None
                    # Check based on the RainPatch if cp is child, if yes assign the parents of the RainPatch also to the cp
                    for i, obj in enumerate(rainPatchList):
                        if (obj.getId() == cp) & (len(obj.getParents()) > 0):
                            index_rain = i
                            cp_parents = rainPatchList[index_rain].getParents().copy()
                            # Check the generation of the parents and assign their max. generation + 1 to the cp
                            for parent in cp_parents:
                                for k, obj1 in enumerate(ColdPoolField.coldpool_list):
                                    if obj1.getId() == parent:
                                        index_parent = k
                                        break
                                if ColdPoolField.coldpool_list[index_parent].getGeneration() >= generation:
                                    generation = ColdPoolField.coldpool_list[index_parent].getGeneration() + 1
                                if ColdPoolField.coldpool_list[index_parent].getFamily() != family:
                                    family = ColdPoolField.coldpool_list[index_parent].getFamily()
                            break                    
                    coldpool = ColdPool(identificationNumber=cp,origin=searchBlobMin(pixelBlob=cp_region,field=tv),
                                        startTimestep=self.__tstep,area=cp_counts[n],virtualTemp_mean=np.mean(tv[cp_region]),
                                        parents=cp_parents,merged=[],intersecting=checkBlobContact(cp_region, self.__labeledCps),
                                        generation=generation,family=family)                   
                    ColdPoolField.coldpool_list.append(coldpool)                     

                n += 1
                
            # If merging took place, update the field
            self.__labeledCps = mergedField
              

                                        
        else:
            # Make sure all class lists are empty at first
            ColdPoolField.coldpool_list = []
            ColdPoolField.coldpoolfamily_list = []
            ColdPoolField.domainstats_dict = {
                "timestep": [],
                "coverageCpInFamily": [],
                "coverageCpNotInFamily": [],
                "coverageCpActive": [],
                "coverageCpDissipating": [],
                "noCpInFamily": [],
                "noCpNotInFamily": [],
                "noCpActive": [],
                "noCpDissipating": [],
                "noCpIsolated": [],
                "noCpIntersecting": [],
                "noFamiliesActive": [],
                "noFamiliesInactive": []}
            for cp in cp_labels:
                # Create new ColdPool for all cold pools and append them to coldpool_list
                cp_region = self.__labeledCps == cp
                coldpool = ColdPool(identificationNumber=cp,origin=searchBlobMin(pixelBlob=cp_region,field=tv),
                                    startTimestep=self.__tstep,area=cp_counts[n],virtualTemp_mean=np.mean(tv[cp_region]),
                                    parents=[],merged=[],intersecting=checkBlobContact(cp_region, self.__labeledCps))
                ColdPoolField.coldpool_list.append(coldpool) 
                n += 1
                
        
        # Evaluate the domain statistics if stats = true
        if stats:
            ColdPoolField.domainstats_dict["timestep"].append(self.__tstep)
            coverageCpNotInFamily = 0
            noCpNotInFamily = 0
            coverageCpInFamily = 0
            noCpInFamily = 0
            noCpActive = 0
            noCpDissipating = 0
            noCpIsolated = 0
            noCpIntersecting = 0
            noFamiliesActive = 0
            noFamiliesInactive = 0
            cp_labels, cp_counts = unique_nonzero(self.__labeledCps, return_counts=True)
            n = 0
            # Loop over current cold pool field and evaluate the statistics
            for cp in cp_labels:
                for i, obj in enumerate(ColdPoolField.coldpool_list):
                    if obj.getId() == cp:
                        index = i
                        break
                if ColdPoolField.coldpool_list[index].getFamily() is None:
                    coverageCpNotInFamily += cp_counts[n]
                    noCpNotInFamily += 1
                else:
                    coverageCpInFamily += cp_counts[n]
                    noCpInFamily += 1
                if ColdPoolField.coldpool_list[index].getState() == 0:
                    noCpActive += 1
                else:
                    noCpDissipating += 1
                if ColdPoolField.coldpool_list[index].getIntersecting() == False:
                    noCpIsolated += 1
                else:
                    noCpIntersecting += 1
                n += 1
            ColdPoolField.domainstats_dict["coverageCpNotInFamily"].append(
                coverageCpNotInFamily/(self.__labeledCps.shape[0]*self.__labeledCps.shape[1]))
            ColdPoolField.domainstats_dict["coverageCpInFamily"].append(
                coverageCpInFamily/(self.__labeledCps.shape[0]*self.__labeledCps.shape[1]))
            ColdPoolField.domainstats_dict["coverageCpActive"].append(np.count_nonzero(self.getLabeledCpsActive())/
                                                                      (self.__labeledCps.shape[0]*self.__labeledCps.shape[1]))
            ColdPoolField.domainstats_dict["coverageCpDissipating"].append(np.count_nonzero(self.__labeledCps)/
                                                                           (self.__labeledCps.shape[0]*self.__labeledCps.shape[1])-
                                                                           ColdPoolField.domainstats_dict["coverageCpActive"][-1])            
            ColdPoolField.domainstats_dict["noCpNotInFamily"].append(noCpNotInFamily)
            ColdPoolField.domainstats_dict["noCpInFamily"].append(noCpInFamily)
            ColdPoolField.domainstats_dict["noCpActive"].append(noCpActive)
            ColdPoolField.domainstats_dict["noCpDissipating"].append(noCpDissipating)
            ColdPoolField.domainstats_dict["noCpIsolated"].append(noCpIsolated)
            ColdPoolField.domainstats_dict["noCpIntersecting"].append(noCpIntersecting)
            
            # Find the number of families of the current cold pool field
            if any(obj.getFamily() is not None for obj in ColdPoolField.coldpool_list):
                notNoneTotalFamilies = [x for x in [obj.getFamily() for obj in ColdPoolField.coldpool_list] 
                                   if x is not None]
                notNoneActiveFamilies = [x for x in [obj.getFamily() for obj in ColdPoolField.coldpool_list] 
                                   if (x is not None) and 
                                   (x in [obj.getFamily() for obj in ColdPoolField.coldpool_list if obj.getId() in cp_labels])]
                noFamiliesActive = len(unique_nonzero(notNoneActiveFamilies))
                noFamiliesInactive = len(unique_nonzero(notNoneTotalFamilies))-noFamiliesActive

            ColdPoolField.domainstats_dict["noFamiliesActive"].append(noFamiliesActive)
            ColdPoolField.domainstats_dict["noFamiliesInactive"].append(noFamiliesInactive)

    
    def __del__(self):
        """
        Deletes the ColdPoolField object
        """
 
    def getTimestep(self):
        
        return self.__tstep        
 
    def getLabeledCps(self):
        
        return self.__labeledCps
    
    def getNumberColdPools(self):
        
        return self.__numberOfColdPools
    
    def getLabeledCpsActive(self):                
        activeCps = self.__labeledCps.copy()
        for cp in unique_nonzero(self.__labeledCps):
            # Find the old cold pool and save its index
            for k, obj in enumerate(ColdPoolField.coldpool_list):
                if obj.getId() == cp:
                    index = k
                    break
            state = ColdPoolField.coldpool_list[index].getState()
            if state != 0:
                activeCps = np.where(activeCps == cp, 0, activeCps)
        
        return activeCps
        
    def getLabeledFamilies(self):
        labeledFamilies = self.__labeledCps.copy()
        for cp in unique_nonzero(self.__labeledCps):
            # Find the old cold pool and save its index
            for k, obj in enumerate(ColdPoolField.coldpool_list):
                if obj.getId() == cp:
                    index = k
                    break
            family = ColdPoolField.coldpool_list[index].getFamily()
            if family is not None:
                labeledFamilies = np.where(self.__labeledCps == cp, family, labeledFamilies)
            else:
                labeledFamilies = np.where(self.__labeledCps == cp, 0, labeledFamilies)
            
        return labeledFamilies        
        
    def getLabeledFamiliesActive(self):
        labeledFamiliesActive = self.__labeledCps.copy()
        for cp in unique_nonzero(self.__labeledCps):
            # Find the old cold pool and save its index
            for k, obj in enumerate(ColdPoolField.coldpool_list):
                if obj.getId() == cp:
                    index = k
                    break
            family = ColdPoolField.coldpool_list[index].getFamily()
            state = ColdPoolField.coldpool_list[index].getState()
            if (family is not None) and (state == 0):
                labeledFamiliesActive = np.where(self.__labeledCps == cp, family, labeledFamiliesActive)
            else:
                labeledFamiliesActive = np.where(self.__labeledCps == cp, 0, labeledFamiliesActive)
            
        return labeledFamiliesActive 





        