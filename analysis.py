#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:37:30 2022

@author: jannik
"""





        # fig, ax = plt.subplots()
        # splot = sns.displot(data=cp_df,x="Max_area", hue="qInitial_rint_mean",kind="kde",
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # splot.set(xscale="log")
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_cpMaxAreaLog-qInitialRintMeanHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.displot(data=cp_df,x="Max_area", hue="qInitial_tv_mean",kind="kde",
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # splot.set(xscale="log")
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_cpMaxAreaLog-qInitialTvMeanHue.png",bbox_inches='tight')        
        # plt.show()     

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="qRain_duration", y="Max_area", data=cp_df)
        # splot.set(yscale="log")
        # if save:
        #     plt.savefig("Plots/boxplot_qRainDuration-cpMaxAreaLog.png",bbox_inches='tight')
        # plt.show() 
               
        # fig, ax = plt.subplots()
        # splot = sns.kdeplot(data=cp_df, x="Max_area", hue="Initial_contact",warn_singular=False)
        # splot.set(xscale="log")
        # if save:
        #     plt.savefig("Plots/kde_cpMaxAreaLog-initialContactHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="qInitial_rint_mean", y="Max_area", data=cp_df)
        # splot.set(yscale="log")
        # if save:
        #     plt.savefig("Plots/boxplot_qInitialRintMean-cpMaxAreaLog.png",bbox_inches='tight')
        # plt.show()

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="qInitial_tv_mean", y="Max_area", data=cp_df)
        # splot.set(yscale="log")
        # if save:
        #     plt.savefig("Plots/boxplot_qInitialTvMean-cpMaxAreaLog.png",bbox_inches='tight')
        # plt.show()

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="No_children_binned", y="Max_area", data=cp_df)
        # splot.set(yscale="log")
        # if save:
        #     plt.savefig("Plots/boxplot_noChildrenBinned-cpMaxAreaLog.png",bbox_inches='tight')
        # plt.show()           

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="No_patrons_binned", y="Max_area", data=cp_df)
        # splot.set(yscale="log")
        # if save:
        #     plt.savefig("Plots/boxplot_noPatronsBinned-cpMaxAreaLog.png",bbox_inches='tight')
        # plt.show() 
        
        
        
        
        
        # fig, ax = plt.subplots()
        # splot = sns.displot(data=family_df,x="Max_age", hue="qInitial_rint_mean",kind="kde",warn_singular=False,
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_familyMaxAge-qInitialRintMeanHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.displot(data=family_df,x="Max_age", hue="qInitial_tv_mean",kind="kde",warn_singular=False,
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_familyMaxAge-qInitialTvMeanHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.scatterplot(data=family_df, x="Max_age", y="Rain_duration", hue='Generations_binned', 
        #                         edgecolor = 'none', alpha = .75)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # if save:
        #     plt.savefig("Plots/scatter_familyMaxAge-rainDuration-generationsBinnedHue.png",bbox_inches='tight')
        # plt.show()

        # fig, ax = plt.subplots()
        # splot = sns.scatterplot(data=family_df, x="No_familyMembers", y="Rain_duration", hue='Generations_binned', 
        #                         edgecolor = 'none', alpha = .75)
        # splot.set(xscale="log")
        # if save:
        #     plt.savefig("Plots/scatter_noFamilyMembersLog-rainDuration-generationsBinnedHue.png",bbox_inches='tight')
        # plt.show()
        
        # fig, ax = plt.subplots()
        # splot = sns.scatterplot(data=family_df, x="No_familyMembers", y="Initial_rint_mean", hue='Generations_binned', 
        #                         edgecolor = 'none', alpha = .75)
        # splot.set(xscale="log")
        # if save:
        #     plt.savefig("Plots/scatter_noFamilyMembersLog-initialRintMean-generationsBinnedHue.png",bbox_inches='tight')
        # plt.show()
        
        # fig, ax = plt.subplots()
        # splot = sns.scatterplot(data=family_df, x="No_familyMembers", y="Initial_tv_mean", hue='Generations_binned', 
        #                         edgecolor = 'none', alpha = .75)
        # splot.set(xscale="log")
        # if save:
        #     plt.savefig("Plots/scatter_noFamilyMembersLog-initialTvMean-generationsBinnedHue.png",bbox_inches='tight')
        # plt.show() 



        # # Calculate some quartiles and categories
        # family_df["qInitial_rint_mean"] = pd.qcut(family_df["Initial_rint_mean"], q=[0, .25, .5, .75, 1.], 
        #                                       precision=3, duplicates='drop')
        # family_df["qInitial_tv_mean"] = pd.qcut(family_df["Initial_tv_mean"], q=[0, .25, .5, .75, 1.], 
        #                                     precision=3, duplicates='drop') 
        # family_df["qRain_duration"] = pd.qcut(family_df["Rain_duration"], q=[0, .25, .5, .75, 1.], 
        #                                   precision=3, duplicates='drop')  

        # family_df["Generations_binned"] = pd.cut(family_df['Generations'], bins=[2, 3, 4, 10000], include_lowest=True, 
        #                                          labels=['2', '3', '>3'])
        # family_df['Generations_binned'] = pd.Categorical(family_df.Generations_binned)
        
        
        
        
        # fig, ax = plt.subplots()
        # splot = sns.displot(data=cp_df,x="Max_age", hue="qInitial_rint_mean",kind="kde",
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_cpMaxAge-qInitialRintMeanHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.displot(data=cp_df,x="Max_age", hue="qInitial_tv_mean",kind="kde",
        #                     multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75")
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.close(1)
        # if save:
        #     plt.savefig("Plots/ckde_cpMaxAge-qInitialTvMeanHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="qRain_duration", y="Max_age", data=cp_df)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # if save:
        #     plt.savefig("Plots/boxplot_qRainDuration-cpMaxAge.png",bbox_inches='tight')
        # plt.show() 
        
        # fig, ax = plt.subplots()
        # splot = sns.kdeplot(data=cp_df, x="Max_age", hue="Initial_contact",warn_singular=False)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # if save:
        #     plt.savefig("Plots/kde_cpMaxAge-initialContactHue.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="No_children_binned", y="Max_age", data=cp_df)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # if save:
        #     plt.savefig("Plots/boxplot_noChildrenBinned-cpMaxAge.png",bbox_inches='tight')
        # plt.show() 

        # fig, ax = plt.subplots()
        # splot = sns.boxplot(x="No_patrons_binned", y="Max_age", data=cp_df)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # if save:
        #     plt.savefig("Plots/boxplot_noPatronsBinned-cpMaxAge.png",bbox_inches='tight')
        # plt.show() 





        # # Calculate some quartiles and categories
        # cp_df["qInitial_rint_mean"] = pd.qcut(cp_df["Initial_rint_mean"], q=[0, .25, .5, .75, 1.], 
        #                                       precision=3, duplicates='drop')
        # cp_df["qInitial_tv_mean"] = pd.qcut(cp_df["Initial_tv_mean"], q=[0, .25, .5, .75, 1.], 
        #                                     precision=3, duplicates='drop')
        # cp_df["qMax_age"] = pd.qcut(cp_df["Max_age"], q=[0, .25, .5, .75, 1.], 
        #                             precision=3, duplicates='drop')    
        # cp_df["qMax_area"] = pd.qcut(cp_df["Max_area"], q=[0, .25, .5, .75, 1.], 
        #                              precision=3, duplicates='drop')  
        # cp_df["qRain_duration"] = pd.qcut(cp_df["Rain_duration"], q=[0, .25, .5, .75, 1.], 
        #                                   precision=3, duplicates='drop')   
        
        # cp_df["No_children_binned"] = pd.cut(cp_df['No_children'], bins=[0, 1, 2, 10000], include_lowest=True, 
        #                                      labels=['0', '1', '>1'])
        # cp_df['No_children_binned'] = pd.Categorical(cp_df.No_children_binned)
        
        # cp_df["No_parents_binned"] = pd.cut(cp_df['No_parents'], bins=[0, 1, 2, 10000], include_lowest=True, 
        #                                     labels=['0', '1', '>1'])
        # cp_df['No_parents_binned'] = pd.Categorical(cp_df.No_parents_binned)
        
        # cp_df["No_patrons_binned"] = pd.cut(cp_df['No_patrons'], bins=[0, 1, 2, 10000], include_lowest=True, 
        #                                     labels=['0', '1', '>1'])
        # cp_df['No_patrons_binned'] = pd.Categorical(cp_df.No_patrons_binned)        
        