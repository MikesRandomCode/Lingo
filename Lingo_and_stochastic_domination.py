#Standard Lingo CDF [0.       , 0.0625054, 0.1250665, 0.1982667, 0.2911126, 0.4093409, 0.5512241, 0.703243 , 0.841298 , 0.9392275, 0.9868886, 0.9990809,1.       , 1.       , 1.       , 1.       ])
#Standard Lingo map np.array([1,3,7,8,11,13,19,20,23])
import numpy as np
import matplotlib.pyplot as plt

def find_CDF_sheet(sheet, itno):
    """
    Generates the CDF of the amount of tries it takes to get Lingo (i.e. five
    crosses in a row) as a vector. It does this using a Monte Carlo simulation.

    Parameters
    ----------
    sheet : Square np.array() with {0, 1} entries;
        The Lingo sheet you want to know the CDF of. Zeros indicate crossed
        numbers, while ones are open numbers.
    itNo : int;
        The number of times you repeat the Lingo finale.

    Returns
    -------
    CDF : one dimensional np.array() with float entries;
        If X is the number of balls drawn before you get Lingo, then
        CDF[i] = Prob(X <= i + 1).

    """
    dim = sheet.shape[0]
    numbers_left = np.nonzero(np.reshape(sheet, dim**2))[0]
    tally = np.zeros(len(numbers_left))
    
    for it in range(itno):
        newsheet = sheet.copy()
        np.random.shuffle(numbers_left)

        
        for index, choice in enumerate(numbers_left):
            no_col_lingo = np.all(np.sum(newsheet, axis = 0))
            no_row_lingo = np.all(np.sum(newsheet, axis = 1))
            no_diag_lingo = np.trace(newsheet)
            no_anidiag_lingo = np.trace(np.fliplr(newsheet))
            
            if not np.all([no_col_lingo, no_row_lingo, no_diag_lingo, no_anidiag_lingo]):
                tally[index:] += 1
                break
    
            newsheet[np.unravel_index(numbers_left[index], (dim, dim))] = 0
        
    CDF = tally / itno

    return CDF 

def find_CDF_sheet_coupled(sheet1, sheet2, itno):
    """
    Generates the CDF of the amount of tries it takes to get Lingo (i.e. five
    crosses in a row) as a vector. It does this using a Monte Carlo simulation.

    Parameters
    ----------
    sheet1 : Square np.array() with {0, 1} entries;
        The reference Lingo sheet you want to know the CDF of. Zeros indicate crossed
        numbers, while ones are open numbers.
    sheet2 : Square np.array() with {0, 1} entries;
        The new sheet you want to know the CDF of.
    itNo : int;
        The number of times you repeat the Lingo finale.

    Returns
    -------
    CDF1 : one dimensional np.array() with float entries;
        If X is the number of balls drawn before you get Lingo, then
        CDF[i] = Prob(X <= i + 1). This correspondes to sheet 1.
    CDF2 : one dimensional np.array() with float entries;
        CDF corresponding to sheet 2.
    """
    dim = sheet1.shape[0]
    dim_alt = sheet2.shape[0]
    if dim != dim_alt:
        raise ValueError("Dimensions do not agree")
        
        
    numbers_left1 = np.nonzero(np.reshape(sheet1, dim**2))[0]
    numbers_left2 = np.nonzero(np.reshape(sheet2, dim**2))[0]
    tally1 = np.zeros(len(numbers_left1))
    tally2 = np.zeros(len(numbers_left2))
    
    for it in range(itno):
        print(it)
        newsheet1 = sheet1.copy()
        newsheet2 = sheet2.copy()
        choices = np.arange(len(numbers_left1))
        np.random.shuffle(choices)

        done1 = False
        done2 = False
        for index, choice in enumerate(choices):
            no_col_lingo = np.all(np.sum(newsheet1, axis = 0))
            no_row_lingo = np.all(np.sum(newsheet1, axis = 1))
            no_diag_lingo = np.trace(newsheet1)
            no_anidiag_lingo = np.trace(np.fliplr(newsheet1))
            
            no_col_lingo2 = np.all(np.sum(newsheet2, axis = 0))
            no_row_lingo2 = np.all(np.sum(newsheet2, axis = 1))
            no_diag_lingo2 = np.trace(newsheet2)
            no_anidiag_lingo2 = np.trace(np.fliplr(newsheet2))
            
            if not np.all([no_col_lingo, no_row_lingo, no_diag_lingo, no_anidiag_lingo]) and not done1:
                tally1[index:] += 1
                done1 = True
                
            if not np.all([no_col_lingo2, no_row_lingo2, no_diag_lingo2, no_anidiag_lingo2]) and not done2:
                tally2[index:] += 1
                done2 = True
                
            if (done1 and done2):
                break
    
            newsheet1[np.unravel_index(numbers_left1[choice], (dim, dim))] = 0
            newsheet2[np.unravel_index(numbers_left2[choice], (dim, dim))] = 0
        
    CDF1 = tally1 / itno
    CDF2 = tally2 / itno
    
    return CDF1, CDF2

def plot_save_CDF(CDF, save = False, savename = "no_name_given"):
    """
    Plots the CDF of a distribution function. Also, saves the plot if indicated.

    Parameters
    ----------
    CDF : One dimenstional np.array() with float entries.
        The CDF to be plotted as a vector of probabilities.
    save : bool, optional;
        Indicates whether figure should be saved. The default is False.
    savename : str, optional
        The name of the savefile. The default is "no_name_given".

    Returns
    -------
    None.

    """
    plt.plot(CDF, 'ok-')
    #plt.ylim((0,1))
    #plt.xlim((0,12))
    plt.grid(True)
    plt.xlabel("Number of balls won in finale")
    plt.ylabel("Probability of getting Lingo")
    plt.title("The CDF of the standard Lingo card")
    
    if save:
        plt.savefig("{}.png".format(savename))
    return

def Find_worse_map(CDF_ref, dim, tries, itno, must_have_initial_lingo = False):
    i = 0
    Sheets = []
    while i < tries:
        print(i)
        base_sheet = np.ones((dim, dim))
        initial = np.sort(np.random.choice(dim ** 2, (dim - 2)**2, replace = False))
        base_sheet[np.unravel_index(initial, (dim, dim))] = 0
        
        if initial.tolist() in Sheets:
            continue
        
        if must_have_initial_lingo:
            col_lingo = np.any(np.sum(base_sheet, axis = 0) == 1)
            row_lingo = np.any(np.sum(base_sheet, axis = 1) == 1)
            diag_lingo = (np.trace(base_sheet) == 1)
            antidiag_lingo = (np.trace(np.fliplr(base_sheet)) == 1)
            
            if not (col_lingo or row_lingo or diag_lingo or antidiag_lingo):
                continue
        
        Sheets.append(initial.tolist())
        
        CDF = find_CDF_sheet(base_sheet, itno)
        
        if np.all(CDF[2:] <= CDF_ref[2:]):
            print("I've found a worse one!")
            return initial, CDF
            
        i += 1
        print(CDF)
        
    print("I could not find a worse one...")
    return None, None

if __name__ == "__main__":
    standardLingoLoc = np.array([1,3,7,8,11,13,19,20,23])
    standardLingoSheet = np.ones(25)
    standardLingoSheet[standardLingoLoc] = 0
    itNo = 10000000
    lingoCDF = find_CDF_sheet(standardLingoSheet.reshape((5,5)), itNo)
    print(f"The Lingo sheet CDF is {lingoCDF}")
    plot_save_CDF(lingoCDF, save = True, savename="Final_CDF_Lingo")
