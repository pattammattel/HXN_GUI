


import time, tqdm
from epics import caget, caput

def triggerPV(PV_name):
    caput(PV_name,1)
    time.sleep(2)
    
def waitWithProgessBar(time_in_minues):
    
    print(f"Timer started for {time_in_minues} minutes")
    
    for _ in tqdm.trange(time_in_minues*60):
        time.sleep(1)
    

def StartPumpingProtocol():

        #pumping PVs
    yield from bps.sleep(1)
    slowVentClose = 'XF:03IDC-VA{ES:1-SlowVtVlv:Stg2}Cmd:Cls-Cmd'
    fastVentClose = 'XF:03IDC-VA{ES:1-FastVtVlv:Stg3}Cmd:Cls-Cmd'

    slowVentStatus = 'XF:03IDC-VA{ES:1-SlowVtVlv:Stg2}Sts:Cls-Sts'
    fastVentStatus = 'XF:03IDC-VA{ES:1-FastVtVlv:Stg3}Sts:Cls-Sts'

    pumpAON = 'XF:03IDC-VA{ES:1-FrPmp:A}Cmd:Start-Cmd'
    pumpBON = 'XF:03IDC-VA{ES:1-FrPmp:B}Cmd:Start-Cmd'

    pumpASlowOpen = 'XF:03IDC-VA{ES:1-SlowFrVlv:A}Cmd:Opn-Cmd'
    pumpAFastOpen = 'XF:03IDC-VA{ES:1-FastFrVlv:A}Cmd:Opn-Cmd'

    pumpBSlowOpen = 'XF:03IDC-VA{ES:1-SlowFrVlv:B}Cmd:Opn-Cmd'
    pumpBFastOpen = 'XF:03IDC-VA{ES:1-FastFrVlv:B}Cmd:Opn-Cmd'

    pumpAOFF = 'XF:03IDC-VA{ES:1-FrPmp:A}Cmd:Stop-Cmd'
    pumpBOFF = 'XF:03IDC-VA{ES:1-FrPmp:B}Cmd:Stop-Cmd'

    pumpASlowClose = 'XF:03IDC-VA{ES:1-SlowFrVlv:A}Cmd:Cls-Cmd'
    pumpAFastClose = 'XF:03IDC-VA{ES:1-FastFrVlv:A}Cmd:Cls-Cmd'

    pumpBSlowClose = 'XF:03IDC-VA{ES:1-SlowFrVlv:B}Cmd:Cls-Cmd'
    pumpBFastClose = 'XF:03IDC-VA{ES:1-FastFrVlv:B}Cmd:Cls-Cmd'
    

    #make sure vents are closed
    
    [triggerPV(pv) for pv in [fastVentClose,slowVentClose]]
    
    #turn on pumps 
    #make sure vents are closed
    if caget(fastVentStatus)==1 and caget(slowVentStatus)==1:
        
        [triggerPV(pv) for pv in [pumpAON,pumpASlowOpen,pumpBON,pumpBSlowOpen]]
        
        #wait for vaccum to reach below 300 for fast open
        waitWithProgessBar(12)
            
        print("FAST Open triggered")
        [triggerPV(pv) for pv in [pumpBFastOpen,pumpAFastOpen]]

        #wait for vaccum to reach ~1  
        waitWithProgessBar(18)
        
        #close pump valves
        [triggerPV(pv) for pv in [pumpBFastClose,pumpAFastClose,
                                  pumpBSlowClose,pumpASlowClose]]

        #tun off the pumps
        [triggerPV(pv) for pv in [pumpAOFF,pumpBOFF]]
        
        #Done!
        
        print("Pumping completed Successfully, Ready for He Backfill")
        
    else: print("Closing the vents failed; Try Manually closing them")
    
def StartAutoHeBackFill():
    yield from bps.sleep(1)
    #pump valve status

    pumpBSlowStats = 'XF:03IDC-VA{ES:1-SlowFrVlv:B}Sts:Cls-Sts'
    pumpASlowStats = 'XF:03IDC-VA{ES:1-SlowFrVlv:A}Sts:Cls-Sts'
    
    pumpBFastStats = 'XF:03IDC-VA{ES:1-FastFrVlv:B}Sts:Cls-Sts'
    pumpAFastStats = 'XF:03IDC-VA{ES:1-FastFrVlv:A}Sts:Cls-Sts'
    
    #He backfill PVs
    startAutoHeBackfill = 'XF:03IDC-VA{ES:1-AutoVt:He}Cmd:Start-Cmd'
    
    HeFillReadiness = [pumpBSlowStats,pumpASlowStats,pumpBFastStats,pumpAFastStats]

    if [caget(pvs) == 1 for pvs in HeFillReadiness]:
    
        readyForHe = True
    
    print("He backfill strats in 30 seconds; Make sure the cylider is open")
    
    time.sleep(30)
    #only execute if pump vales are closed
    if readyForHe:    
        triggerPV(startAutoHeBackfill)
        waitWithProgessBar(15)
        print("He backfilled; Please close the cyclinder")
        
    else: print("One or more valves is not closed; try again")
            
    
def ventChamber():
    
    slowVentOpen = 'XF:03IDC-VA{ES:1-SlowVtVlv:Stg2}Cmd:Opn-Cmd'
    fastVentOpen = 'XF:03IDC-VA{ES:1-FastVtVlv:Stg3}Cmd:Opn-Cmd'
    
    #make sure fluorescence detector is out while relasing the vaccuum
    yield from bps.mov(fdet1.x, -107)
    
    triggerPV(slowVentOpen)
    waitWithProgessBar(10)
    
    triggerPV(fastVentOpen)
    waitWithProgessBar(2)
    
    
        
        
    

