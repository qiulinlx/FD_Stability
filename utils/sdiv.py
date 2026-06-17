import pandas as pd
import math

def species_richness(sp_loc: pd.DataFrame) -> pd.DataFrame:
    pID = []
    Srich = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for i, species in enumerate(species_PID):
        pid = species_PID.index[i]
        Srich.append(len(species))
        pID.append(pid)

    return pd.DataFrame({'PID': pID, 'Species Richness': Srich})

def shannon_diversity(sp_loc):
    pID = []
    shannon = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):
        ab= sp_loc[sp_loc.index == pid]
        ab=ab.to_numpy().flatten()
        ab = ab[ab > 0]  
        ab= ab/ab.sum()  
        total = -sum(x*math.log(x) for x in ab)
        shannon.append(total)
        pID.append(pid)

    shannon_df = pd.DataFrame({"PID": pID, "Shannon Diversity": shannon})
    
    return shannon_df


def simpsons_index(sp_loc):
    '''
    Docstring for simpsons_index
    
    Measure of Dominance /
    '''

    pID = []
    simpsons = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):
        ab= sp_loc[sp_loc.index == pid]
        ab=ab.to_numpy().flatten()
        ab = ab[ab > 0]   
        total = sum(x*(x-1) for x in ab)
        total = 1 - (total/ (ab.sum()*(ab.sum()-1)))
        simpsons.append(total)
        pID.append(pid)

    simpsons_df = pd.DataFrame({"PID": pID, "Simpson's Index": simpsons})
    
    return simpsons_df


def shannon_equitability(sp_loc):
    """
    Also known as Pielou's Evenness. Measure of Evennes 
    Calculate the Shannon Equivalence (also known as the Effective Number of Species)
    
    """
    pID = []
    shannon = []

    species_PID = sp_loc.apply(lambda row: row.index[row != 0].tolist(), axis=1)

    for pid, species in zip(species_PID.index,species_PID):
        S= len(species)
        ab= sp_loc[sp_loc.index == pid]
        ab=ab.to_numpy().flatten()
        ab = ab[ab > 0]  
        ab= ab/ab.sum()  
        total = -sum(x*math.log(x) for x in ab)
        equitabiilty= total /math.log(S) if S > 1 else 0
        shannon.append(equitabiilty)
        pID.append(pid)

    shannon_equi_df = pd.DataFrame({"PID": pID, "Shannon Equitabiltiy Index": shannon})
    
    return shannon_equi_df