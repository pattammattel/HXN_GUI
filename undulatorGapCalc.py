#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[20]:


def calcGap(targetE = 12, harmonics = 5):
    
    "calculate the gap for a given E and harmonics using the polynomial fit"
    
    E1 = targetE/harmonics
     
    gap = 0.655+(4.380*E1)-(1.085*E1**2)+(0.202*E1**3) #equation from curve fit
    
    return np.around(gap*1000,1) #one decimal 




def FindGap(targetE =12):
    
    "Estimate a suitable gap for target energy"
    
    harmonics = [3,5,7,9] #harmonics options
    
    if targetE>5.5 and targetE<25:     # if dialed a wrong number
    
        opt = np.array([calcGap(targetE = targetE, harmonics = hm) for hm in harmonics])
    
        return opt[np.where(np.logical_and(opt>=6000, opt<=10000))][-1] #last one has lowest gap in the list
    
    else:
        print(" Requested Energy is out of range")
        return

        
def moveEnergy(E = 12):
    
    ugap_ = findGap(targetE = E)
    yield from bps.mov(e, E)
    yield from bps.mov(ugap, ugap_)
    
    print(f"Current energy is {e.position} ")
    

    
def optimizePitch():
    
    #find curvefit eqn for HFM and Pitch
    #go to appr. pitch
    #find beam at ssa2(open),  follow ic1
    #close ssa horz. and center HFM coarse
    #scan the pitch 
    
    pass
    
    


# In[39]:


df = pd.DataFrame(columns= ['Energy', 'calcGap'], dtype=object,index=range(50))

for i in range(0,30):
    df.loc[i].Energy = i
    df.loc[i].calcGap = suggestAGap(targetE = i)


# In[40]:


df


# In[23]:


df


# In[ ]:




