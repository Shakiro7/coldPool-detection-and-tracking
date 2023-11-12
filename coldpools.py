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
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label
from utils import unique_nonzero, searchBlobMin, checkBlobContact, scale01, invert01, searchOrigin, findObjIndex
from utils import createLabeledCps






class ColdPool:
    
    def __init__(self,identificationNumber,origin,startTimestep,area,virtualTemp_mean,parents,age=1,dissipating=0,intersecting=False,
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
    
    def __init__(self,identificationNumber,founder,startTimestep,familyMembers,age=1):
        
        self.__id = identificationNumber
        self.__founder = founder
        self.__startTimestep = startTimestep
        self.__familyMembers = familyMembers
        self.__age = age
        
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
    
    def __init__(self,timestep,markers,rainPatchList,rainMarkers,dataloader,mask,minSize=50,onlyNew=False,oldCps=None,
                 periodicDomain=True,domainStats=False,fillOnlyBackgroundHoles=False,minCpRpFactor=1,maxCpRpFactor=3):
        
        self.__tstep = timestep
        markers = markers
        rainPatchList = rainPatchList
        rainMarkers = rainMarkers
        tv = dataloader.getTv()      
        mask = mask
        minSize = minSize
        onlyNew = onlyNew
        labeledCpsOld = oldCps
        periodicBc = periodicDomain
        stats = domainStats
        fillOnlyBackgroundHoles = fillOnlyBackgroundHoles
        
        # Define valid proportions between rain patch and cold pool area
        minCpRpFactor = minCpRpFactor
        maxCpRpFactor = maxCpRpFactor
        
        # Compute the elevation for the watershed filling
        q01filt = scale01(filters.gaussian(dataloader.getQ(), sigma=1.0))
        t01filt = scale01(filters.gaussian(dataloader.getT(), sigma=1.0))
        w01filt = scale01(filters.gaussian(dataloader.getW(), sigma=2.0))
        elevationMap = t01filt+q01filt**2+w01filt
        
        # Compute the field for the center of mass evaluation of the origin
        rintFieldCenter = False
        if rintFieldCenter:
            field = filters.gaussian(dataloader.getRint(), sigma=2.0)
        else:
            field = invert01(scale01(t01filt+w01filt))     
      
                    
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
        
        # Create the labeled cold pool field by filling the elevation map at markers and masking it
        self.__labeledCps = createLabeledCps(markers=markers, elevationMap=elevationMap, mask=mask,
                                             periodicDomain=periodicBc,fillOnlyBackgroundHoles=False)       

        # fig, ax = plt.subplots(figsize=(10,10))
        # cmap = plt.cm.nipy_spectral  
        # ax.imshow(self.__labeledCps, cmap=cmap)
        # ax.set_title('Cps after watershed')
        # plt.show()  

        # Create ColdPools for all unique labels in labeledCps and store them in the list
        cp_labels, cp_counts = unique_nonzero(self.__labeledCps, return_counts=True)
        
        if labeledCpsOld is not None:        
            # Derive a list which holds only the old CPs that were also(!) there the tstep before
            cp_labels_old, cp_counts_old = unique_nonzero(labeledCpsOld,return_counts=True)
            cp_labels_new = np.setdiff1d(cp_labels,cp_labels_old)
            labeledCpsOldAndNew = np.where(np.isin(self.__labeledCps,cp_labels_new),0,self.__labeledCps)
            cp_labels_oldAndNew, cp_counts_oldAndNew = unique_nonzero(labeledCpsOldAndNew,return_counts=True)
        
            # Check if the old CPs fulfill the minSize criterion (new CPs are treated separately later)
            # CPs that do not fulfill the criterion are either overwritten (if overlapped by other old CP) or dropped (else)
            l = 0
            for cp_oldAndNew in cp_labels_oldAndNew:
                if cp_counts_oldAndNew[l] < minSize:
                    replace = False
                    cp_region = labeledCpsOld == cp_oldAndNew
                    overlap_cp = cp_region * self.__labeledCps
                    unique, number = unique_nonzero(overlap_cp, return_counts=True)                  
                    itemindex = np.where(unique==cp_oldAndNew)
                    unique = np.delete(unique, itemindex)
                    number = np.delete(number, itemindex)                        
                    if len(unique) > 0:
                        # Only take action if the largest overlapper is not a new cp (they are treated separately)
                        #if any(obj.getId() == unique[np.argmax(number)] for obj in ColdPoolField.coldpool_list):
                        if unique[np.argmax(number)] in cp_labels_oldAndNew:
                            updated_label = unique[np.argmax(number)]
                            replace = True
                    # If no other cp overlaps its old area, drop it
                    else:
                        updated_label = 0
                        replace = True
                    if replace:
                        #print("CP" + str(cp_oldAndNew) + " with " + str(cp_counts_oldAndNew[l]) + "pixel replaced by " + str(updated_label))
                        self.__labeledCps = np.where(self.__labeledCps == cp_oldAndNew, updated_label, self.__labeledCps)
                l += 1

        # fig, ax = plt.subplots(figsize=(10,10))
        # cmap = plt.cm.nipy_spectral  
        # ax.imshow(self.__labeledCps, cmap=cmap)
        # ax.set_title('Cps after minSize')
        # plt.show() 
        
        # Update cp_labels
        cp_labels, cp_counts = unique_nonzero(self.__labeledCps, return_counts=True)
        
        self.__numberOfColdPools = len(cp_labels)
        n = 0                   
        if labeledCpsOld is not None:
            # Increase the age of all already exisiting (unique) families of the current cold pool field by one
            currentFamilies = [obj.getFamily() for obj in ColdPoolField.coldpool_list if obj.getId() in cp_labels]
            if any(obj is not None for obj in currentFamilies):
                notNoneFamilies = [x for x in currentFamilies if x is not None]
                families = unique_nonzero(notNoneFamilies)
                for family in families:
                    ColdPoolField.coldpoolfamily_list[family-1].setAge()
                    
            # Create lists to store child cold pool IDs along with the family which they are born in
            childcp_list = []
            family_list = []
            for cp in cp_labels:
                if any(obj.getId() == cp for obj in ColdPoolField.coldpool_list):
                # Only modify existing ColdPools                    
                    index_cp = findObjIndex(ColdPoolField.coldpool_list,cp) 
                    if cp_counts[n] > ColdPoolField.coldpool_list[index_cp].getArea():
                        ColdPoolField.coldpool_list[index_cp].setArea(cp_counts[n])
                    ColdPoolField.coldpool_list[index_cp].setAge()
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
                                    # If cp is the only parent and has already a family, add the child to it if not yet done
                                     else:
                                         familyLabl = ColdPoolField.coldpool_list[index_cp].getFamily()
                                         if child not in ColdPoolField.coldpoolfamily_list[familyLabl-1].getFamilyMembers():
                                             childcp_list.append(child)
                                             #print("Adding child " + str(child) + " to family " + str(familyLabl))
                                             family_list.append(familyLabl)
                                             ColdPoolField.coldpoolfamily_list[familyLabl-1].setFamilyMembers(child)
                                             
                                 # If child has more than one parent  
                                 else:
                                    parent_list = rainPatchList[index_child].getParents()
                                    mainParent = rainPatchList[index_child].getMainParent()
                                    indexMainParent = findObjIndex(ColdPoolField.coldpool_list,mainParent)
                                    # In case main parent doesn't have a family yet, found a new one
                                    if ColdPoolField.coldpool_list[indexMainParent].getFamily() is None:
                                        familyId = len(ColdPoolField.coldpoolfamily_list) + 1
                                        childcp_list.append(child)
                                        family_list.append(familyId) 
                                        ColdPoolField.coldpool_list[indexMainParent].setFamily(familyId)
                                        founder = parent_list
                                        familyMembers = founder + [child]                                        
                                        family = ColdPoolFamily(identificationNumber=familyId, founder=founder, 
                                                                 startTimestep=self.__tstep, familyMembers=familyMembers)
                                        ColdPoolField.coldpoolfamily_list.append(family)
                                        for parent in [x for x in parent_list if x != mainParent]:
                                            index_parent = findObjIndex(ColdPoolField.coldpool_list, parent)
                                            if ColdPoolField.coldpool_list[index_parent].getFamily() is None:
                                                ColdPoolField.coldpool_list[index_parent].setFamily(familyId)
                                    # In case main parent has a family already, integrate the child (and the other parents) into it if not yet done     
                                    else:
                                        familyLabl = ColdPoolField.coldpool_list[indexMainParent].getFamily()
                                        if child not in ColdPoolField.coldpoolfamily_list[familyLabl-1].getFamilyMembers():
                                             childcp_list.append(child)
                                             family_list.append(familyLabl)
                                             ColdPoolField.coldpoolfamily_list[familyLabl-1].setFamilyMembers(child)
                                             for parent in [x for x in parent_list if x != mainParent]:
                                                 index_parent = findObjIndex(ColdPoolField.coldpool_list, parent)
                                                 if ColdPoolField.coldpool_list[index_parent].getFamily() is None:
                                                     ColdPoolField.coldpool_list[index_parent].setFamily(familyLabl)
                                                     ColdPoolField.coldpoolfamily_list[familyLabl-1].setFamilyMembers(parent)
                                                 else:
                                                     if parent not in ColdPoolField.coldpoolfamily_list[familyLabl-1].getFamilyMembers():
                                                         ColdPoolField.coldpoolfamily_list[familyLabl-1].setFamilyMembers(parent)
                                            
                    # If cp has children, add them as children to the cp object    
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
                    rain_region = rainMarkers == cp
                    intersecting = checkBlobContact(cp_region, self.__labeledCps)
                    cp_parents = []
                    replaced = False
                    # Check based on the RainPatch if cp is child
                    for i, obj in enumerate(rainPatchList):
                        if (obj.getId() == cp) & (len(obj.getParents()) > 0):
                            index_rain = i
                            cp_parents = rainPatchList[index_rain].getParents().copy()
                            break
                    # Check if the proportions of rain patch (rp) and the resulting cold pool are reasonable        
                    if not intersecting:
                        # If the cp is isolated, keep it if rp <= cp <= 4*rp, else drop it
                        if (np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region)) or (np.count_nonzero(cp_region) < minCpRpFactor * np.count_nonzero(rain_region)):
                            self.__labeledCps = np.where(self.__labeledCps == cp, 0, self.__labeledCps)
                            replaced = True
                    else:
                        # If the cp is intersecting, check whether it has parents
                        if len(cp_parents) == 0:
                            # If the cp has no parents, keep it if rp <= cp <= 4*rp
                            # In case cp > 4*rp, assign it the label of the largest overlapped cp from the previous tstep
                            if np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region):
                                overlap_cp = cp_region * labeledCpsOld
                                unique, number = unique_nonzero(overlap_cp, return_counts=True)                  
                                if len(unique) > 0:
                                        updated_label = unique[np.argmax(number)]
                                # If the cp overlaps no old cp, drop it
                                else:
                                    updated_label = 0                            
                                self.__labeledCps = np.where(self.__labeledCps == cp, updated_label, self.__labeledCps)
                                replaced = True   
                            # In case cp < rp, assign it the label of the cp that stole most of its (rain patch) area
                            elif np.count_nonzero(cp_region) < minCpRpFactor * np.count_nonzero(rain_region):
                                overlap_rain = rain_region * self.__labeledCps
                                unique, number = unique_nonzero(overlap_rain, return_counts=True)                  
                                itemindex = np.where(unique==cp)
                                unique = np.delete(unique, itemindex)
                                number = np.delete(number, itemindex)
                                if len(unique) > 0:
                                        updated_label = unique[np.argmax(number)]
                                # If no other cp overlaps that cp's rp, just drop it
                                else:
                                    updated_label = 0
                                self.__labeledCps = np.where(self.__labeledCps == cp, updated_label, self.__labeledCps)
                                replaced = True
                        elif len(cp_parents) == 1:
                            # If the cp has one parent, keep it if rp <= cp <= 4*rp, else replace with parent
                            if (np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region)) or (np.count_nonzero(cp_region) < 1 * np.count_nonzero(rain_region)):
                                self.__labeledCps = np.where(self.__labeledCps == cp, cp_parents[0], self.__labeledCps)
                                replaced = True                           
                        else:
                            # If the cp has multiple parents, keep it if rp <= cp <= 4*rp
                            # In case cp > 4*rp, replace with parent that occupied most of its region before
                            if np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region):
                                overlap_cp = cp_region * labeledCpsOld
                                unique, number = unique_nonzero(overlap_cp, return_counts=True)                  
                                itemindex = np.isin(unique,cp_parents)
                                unique_parents = unique[itemindex]
                                number_parents = number[itemindex]
                                if len(unique_parents) > 0:
                                        updated_label = unique_parents[np.argmax(number_parents)]
                                # If the cp overlaps no old parent, use the largest rp overlap as in the other case
                                else:
                                    updated_label = rainPatchList[index_rain].getMainParent()
                                self.__labeledCps = np.where(self.__labeledCps == cp, updated_label, self.__labeledCps)
                                replaced = True
                            # In case cp < rp, replace with parent with largest rp overlap (stored as main parent)
                            elif np.count_nonzero(cp_region) < minCpRpFactor * np.count_nonzero(rain_region):
                                updated_label = rainPatchList[index_rain].getMainParent()
                                self.__labeledCps = np.where(self.__labeledCps == cp, updated_label, self.__labeledCps)
                                replaced = True                            
                                
                    # If the cp survived the check, create its object
                    if not replaced:
                        generation = 1   
                        family = None                     
                        # Check the generation of parents (if any) and assign their max. generation + 1 to the cp
                        if len(cp_parents) > 0:
                            for parent in cp_parents:
                                index_parent = findObjIndex(ColdPoolField.coldpool_list,parent)
                                # If parent is main parent, assign its family to the new cp
                                if parent == rainPatchList[index_rain].getMainParent():
                                    if ColdPoolField.coldpool_list[index_parent].getFamily() is not None:
                                        family = ColdPoolField.coldpool_list[index_parent].getFamily()
                                    # In case the main parent has no family yet (because it was superseeded or below minSize this tstep)
                                    # -> create the family now and add the new cp as child to all parents
                                    else:
                                        # print("CP " + str(cp) + " is an orphan")
                                        familyId = len(ColdPoolField.coldpoolfamily_list) + 1
                                        childcp_list.append(cp)
                                        family_list.append(familyId) 
                                        ColdPoolField.coldpool_list[index_parent].setFamily(familyId)
                                        ColdPoolField.coldpool_list[index_parent].setChildren(cp)   
                                        founder = cp_parents
                                        familyMembers = founder + [cp]
                                        ColdPoolField.coldpoolfamily_list.append(ColdPoolFamily(identificationNumber=familyId, founder=founder, 
                                                                                                startTimestep=self.__tstep, familyMembers=familyMembers))                                        
                                        family = familyId
                                        if len(cp_parents) > 1:
                                            for p in [x for x in cp_parents if x != parent]:
                                                index_p = findObjIndex(ColdPoolField.coldpool_list, p)
                                                if ColdPoolField.coldpool_list[index_p].getFamily() is None:
                                                    ColdPoolField.coldpool_list[index_p].setFamily(familyId)
                                                ColdPoolField.coldpool_list[index_p].setChildren(cp)  
                                # Let the new cp obtain the largest parent generation + 1
                                if ColdPoolField.coldpool_list[index_parent].getGeneration() >= generation:
                                    generation = ColdPoolField.coldpool_list[index_parent].getGeneration() + 1
                        origin = searchOrigin(pixelBlob=rain_region,field=field,periodicDomain=periodicBc)
                        coldpool = ColdPool(identificationNumber=cp,origin=origin,
                                            startTimestep=self.__tstep,area=cp_counts[n],virtualTemp_mean=np.mean(tv[cp_region]),
                                            parents=cp_parents,intersecting=intersecting,generation=generation,family=family)                   
                        ColdPoolField.coldpool_list.append(coldpool)
                        # Check if the newly appended ColdPool breaks the sorting and if yes, sort again
                        try:
                            if coldpool.getId() < ColdPoolField.coldpool_list[-2].getId():
                                ColdPoolField.coldpool_list = sorted(ColdPoolField.coldpool_list, key=lambda x: x.getId(), reverse=False)
                        except:
                            pass
                    # If the cold pool was dropped reduce the number of cps by 1
                    else:
                        self.__numberOfColdPools -= 1

                n += 1
                
              

                                        
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
                rain_region = rainMarkers == cp
                intersecting = checkBlobContact(cp_region, self.__labeledCps)
                replaced = False
                # Check if the proportions of rain patch (rp) and the resulting cold pool are reasonable        
                if not intersecting:
                    # If the cp is isolated, keep it if rp <= cp <= 4*rp, else drop it (if onlyNew==False, only check that cp >= rp)
                    if onlyNew:
                        if np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region):
                            self.__labeledCps = np.where(self.__labeledCps == cp, 0, self.__labeledCps)
                            replaced = True
                    if not replaced:
                        if np.count_nonzero(cp_region) < minCpRpFactor * np.count_nonzero(rain_region):
                            self.__labeledCps = np.where(self.__labeledCps == cp, 0, self.__labeledCps)
                            replaced = True                        
                else:
                    # If the cp is intersecting, keep it if rp <= cp <= 4*rp (if onlyNew==False, only check that cp >= rp)
                    # In case cp > 4*rp, drop it
                    if onlyNew:
                        if np.count_nonzero(cp_region) > maxCpRpFactor * np.count_nonzero(rain_region):                          
                            self.__labeledCps = np.where(self.__labeledCps == cp, 0, self.__labeledCps)
                            replaced = True   
                    # In case cp < rp, assign it the label of the cp that stole most of its (rain patch) area
                    if not replaced:
                        if np.count_nonzero(cp_region) < minCpRpFactor * np.count_nonzero(rain_region):
                            overlap_rain = rain_region * self.__labeledCps
                            unique, number = unique_nonzero(overlap_rain, return_counts=True)                  
                            itemindex = np.where(unique==cp)
                            unique = np.delete(unique, itemindex)
                            number = np.delete(number, itemindex)
                            if len(unique) > 0:
                                    updated_label = unique[np.argmax(number)]
                            # If no other cp overlaps that cp's rp, just drop it
                            else:
                                updated_label = 0
                            self.__labeledCps = np.where(self.__labeledCps == cp, updated_label, self.__labeledCps)
                            replaced = True                    
                if not replaced:
                    origin = searchOrigin(pixelBlob=rain_region,field=field,periodicDomain=periodicBc)
                    coldpool = ColdPool(identificationNumber=cp,origin=origin,
                                        startTimestep=self.__tstep,area=cp_counts[n],virtualTemp_mean=np.mean(tv[cp_region]),
                                        parents=[],intersecting=intersecting)
                    ColdPoolField.coldpool_list.append(coldpool)
                # If the cold pool was dropped reduce the number of cps by 1
                else:
                    self.__numberOfColdPools -= 1
                n += 1
                
        # fig, ax = plt.subplots(figsize=(10,10))
        # cmap = plt.cm.nipy_spectral  
        # ax.imshow(self.__labeledCps, cmap=cmap)
        # ax.set_title('Cps after family routine')
        # plt.show()         
        
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
                index = findObjIndex(ColdPoolField.coldpool_list,cp)
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
            index = findObjIndex(ColdPoolField.coldpool_list,cp)
            state = ColdPoolField.coldpool_list[index].getState()
            if state != 0:
                activeCps = np.where(activeCps == cp, 0, activeCps)
        
        return activeCps
        
    def getLabeledFamilies(self):
        labeledFamilies = self.__labeledCps.copy()
        for cp in unique_nonzero(self.__labeledCps):
            # Find the old cold pool and save its index
            index = findObjIndex(ColdPoolField.coldpool_list,cp)
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
            index = findObjIndex(ColdPoolField.coldpool_list,cp)
            family = ColdPoolField.coldpool_list[index].getFamily()
            state = ColdPoolField.coldpool_list[index].getState()
            if (family is not None) and (state == 0):
                labeledFamiliesActive = np.where(self.__labeledCps == cp, family, labeledFamiliesActive)
            else:
                labeledFamiliesActive = np.where(self.__labeledCps == cp, 0, labeledFamiliesActive)
            
        return labeledFamiliesActive 





        