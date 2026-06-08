""" this module contains the function to compute the statistics of numpy matrix"""

import scikit_posthocs as sp
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal
from ..selective_classifiers import fancy_names
import numpy as np
import seaborn as sns
def remap_names(ranking_df,colors_dict=None):
    '''this function is used to remap the names of the methods to more meaningful names. It is used to plot the critical difference diagram.
    '''
    # create colors_dict as a dictionary that maps the color based on the content of the content of ranking_df
    res_colors = {fancy_names(k):"black" for k,v in ranking_df.items()}
    # rankin_df is a dataframe that has as many columns called as the methods 
    # to remap the names of the methods we use the res_d dictionary
    ranking_df.columns = [fancy_names(col) for col in ranking_df.columns]
    print(res_colors,"res_colors")
    print(ranking_df.columns,"columns of the ranking_df")
    return ranking_df,res_colors


def create_df_and_check(methods_names,
                        values,
                        title="",
                        title_size=18,
                        show=True,
                        exp="",
                        metrica='',
                        dt_name ='dt_name',vmin=0,vmax=0.4,hmargin=0.1,vertica_image_size=5,
                        colors_dict=None):
    '''methods_names is a list of the names of the methods.
    values is a list containing the values of the methods. It has as many columns as the number of methods and as many rows as the number of metric domanins. For example if the metrics rre Frob and Kendalltau, the values will have 2 rows and as many rows as the number of proposals.'''
    # create the dataframe
    df = pd.DataFrame(values,columns=methods_names)
    # compute the ranks for the friedman test
    ranks = df.rank(axis=1)
    # check the shape of the ranks to be analogous to the methods_names
    assert ranks.values.shape[1] == len(methods_names), "the ranks have a different number of columns than the methods names"

    #print(ranks.shape,"shape of the ranks")
    #print(df.head())
    #print(ranks.head())
    
    # Convert p-value to scientific notation
    # now check the friedman test

    _, p = ss.friedmanchisquare(*values)
    cd = sp.posthoc_nemenyi_friedman(ranks)

    if p < 0.05:
        #print("the differences are significant\n"+title)
        # prepare the CD diagram
        if  len(values)>=2:
            fig,ax = plt.subplots(2,1,figsize=(10,vertica_image_size))
            axes = ax.flatten()

            ranks_res,colors_res = remap_names(ranking_df=ranks,colors_dict=colors_dict)
            if (np.sum(cd<0.05)>0).any():
                # plot the CD diagram using the ranks and the critical difference
                cd.columns = [fancy_names(col) for col in cd.columns]
                # here it gives an error because the adjiance matrix is not symmetric
                # to solve this we need to make the adjiance matrix symmetric
                # the index of the adjiance matrix should be the same as the columns
                cd.index = cd.columns
                label_props = {"fontsize":title_size-1}
                print("plotting CD diagram")
                # all in black  
                black_colors = {k:"black" for k in colors_res.keys()}
                axes[1].set_xlim(1,len(methods_names))
                # set the xlim to be in the 1-len(methods_names)-1 range
                # (there is never a first and last method that is significantly different)
                sp.critical_difference_diagram(ranks_res.T.mean(axis=1),cd,label_props=label_props,color_palette=black_colors,ax=axes[1],
                                               text_h_margin=hmargin)

            else:
                # now write big on the image "no significant difference"
                axes[1].text(0.5,0.5,"no significant difference",fontsize=20)
            # plot the boxplot of the values getting the pathces from the colors dictionary
            if metrica != "$ShapGap_{L2}$":
                values= -values
                df = -df
                if metrica != "Faithfulness":
                    ylab = "Similarity"
                
                else:
                    ylab = "Faithfulness"
            else:
                ylab = "Distance"
            
            axes[0].grid(axis='y')
            print("plotting boxplot")
            print(df)
            #for j in range(len(df.columns)):
            #    axes[0].scatter(j*np.ones(len(df)),df.iloc[:,j])
            #axes[0].boxplot(values,patch_artist=True,showfliers=True)
            sns.boxenplot(data=df,ax=axes[0],#palette=[colors_dict[k] for k in df.columns],
                          fill=False)

            '''
            pathches = axes[0].boxplot(values,patch_artist=True,showfliers=True)
            for pathch, mn in zip(pathches['boxes'],methods_names):
                c = colors[mn]
                
                pathch.set_facecolor(c)
            '''
            # set the xticks to be the names of the methods
            axes[0].set_xticks(np.arange(len(methods_names)),ranks_res.columns,rotation=30,fontsize=15)
            # set the title of the boxplot
            #axes[0].set_title("Boxplot of the values\n"+dt_name+" "+" ".join([str(elemento) for elemento in exp]),fontsize=title_size)
            # the y grid
            # set the y label
            axes[0].set_ylabel(ylab,fontsize=title_size)
            # set ylim
            #axes[0].set_ylim(vmin,vmax)

            
            p = np.sum(cd<0.05)
        print(cd.round(2))
        # remove the diagonal of the cd matrix as a new cd1 matrix
        cd1 = cd.copy()
        np.fill_diagonal(cd1.values,0)
        # do the sum
        s = np.sum(cd1.values,axis=1)
        print(s)
        # get the index of the lowest value that is lower than 0.05*len(methods_names)
        idx = np.where(s<0.05*len(methods_names))[0]
        print(s[idx])
        if len(idx)>0:
            p = min(s[idx])
        
            # transform the p value to scientific notation like 0.3E-3
            p_str = "{:.3E}".format(Decimal(str(p)))
            title += " - p value "+p_str
        else:
            title += " - No significant difference"

        plt.suptitle(title,fontsize=title_size)
        plt.tight_layout()
        
    print()