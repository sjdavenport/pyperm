"""
A file to compute and save the FPR
"""

# Import statements
import numpy as np
import pyrft as pr

# Set the location to save the results
saveloc = 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes\\pyrft\\results\\'

# Set the save file name
savefilename = 'FPRresults_all'
saveloc = saveloc + savefilename

# Initialize the contrast matrix
C = np.array([[1,-1,0],[0,1,-1]]); 

nsubj_vec = np.arange(10,101,10)
dim_sides = np.array([25,50,100,200])
nsubj_vec = np.array([100])
dim_sides = np.array([100])
 
# Initialize matrices to store the estimated FPRs
store_JER = np.zeros((len(nsubj_vec), len(dim_sides)))
store_FWER = np.zeros((len(nsubj_vec), len(dim_sides)))

# Choose the smoothness, the number of bootstraps and the number of iterations to use
FWHM = 0; B = 100; niters = 1000
for J in np.arange(len(dim_sides)):
  print('J:', J)
  Dim = (dim_sides[J], dim_sides[J])
  
  # In the single voxel case use this dimension
  if Dim == (1,1):
      Dim = 1
      
  for I in np.arange(len(nsubj_vec)):
    print('I:', I)
    FWER_FPR, JER_FPR = pr.bootFPR(Dim, nsubj_vec[I], C, FWHM, 0, B)
    store_JER[I,J] = JER_FPR
    store_FWER[I,J] = FWER_FPR
    np.savez(saveloc + '.npz', JER_FPR  = store_JER, FWER_FPR = store_FWER)
    
# %%
# Set the location to save the results
saveloc = 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes\\pyrft\\results\\'

# Set the save file name
savefilename = 'FPRresults_singlecontrast_all'
#savefilename = 'testing'
saveloc = saveloc + savefilename

# Initialize the contrast matrix
C = np.array([[1,-1,0]]); 

nsubj_vec = np.arange(10,101,10)
dim_sides = np.array([25,50,100,200])
 
# Initialize matrices to store the estimated FPRs
store_JER = np.zeros((len(nsubj_vec), len(dim_sides)))
store_FWER = np.zeros((len(nsubj_vec), len(dim_sides)))

# Choose the smoothness, the number of bootstraps and the number of iterations to use
FWHM = 0; B = 100; niters = 1000
for J in np.arange(len(dim_sides)):
  print('J:', J)
  Dim = (dim_sides[J], dim_sides[J])
  
  # In the single voxel case use this dimension
  if Dim == (1,1):
      Dim = 1
      
  for I in np.arange(len(nsubj_vec)):
    print('I:', I)
    FWER_FPR, JER_FPR = pr.bootFPR(Dim, nsubj_vec[I], C, FWHM, 0, B)
    store_JER[I,J] = JER_FPR
    store_FWER[I,J] = FWER_FPR
    print(FWER_FPR)
    print(JER_FPR)
    print(store_JER)
    print(store_FWER)
    np.savez(saveloc + '.npz', JER_FPR  = store_JER, FWER_FPR = store_FWER)
    
# %%
# Set the location to save the results
saveloc = 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes\\pyrft\\results\\'

# Set the save file name
savefilename = 'FPRresults_X'
saveloc = saveloc + savefilename

# Initialize the contrast matrix
C = np.array([1,-1,0]); 

nsubj_vec = np.arange(10,101,10)
dim_sides = np.array([1,5,10,25,50,100])
 
# Initialize matrices to store the estimated FPRs
store_JER = np.zeros((len(nsubj_vec), len(dim_sides)))
store_FWER = np.zeros((len(nsubj_vec), len(dim_sides)))

# Choose the smoothness, the number of bootstraps and the number of iterations to use
FWHM = 0; B = 100; niters = 1000
for J in np.arange(len(dim_sides)):
  print('J:', J)
  Dim = (dim_sides[J], dim_sides[J])
  
  # In the single voxel case use this dimension
  if Dim == (1,1):
      Dim = 1
      
  for I in np.arange(len(nsubj_vec)):
      
    print('I:', I)
    FWER_FPR, JER_FPR = pr.bootfpr(Dim, nsubj_vec[I], C, FWHM, 0, B)
    store_JER[I,J] = JER_FPR
    store_FWER[I,J] = FWER_FPR
    np.savez(saveloc + '.npz', JER_FPR  = store_JER, FWER_FPR = store_FWER)
    
# %%
#load_results = np.load('./FPRresults_all.npz')
load_results = np.load('./FPRresults_singlecontrast_all.npz')

JER_FPR = load_results['JER_FPR']
FWER_FPR = load_results['FWER_FPR']

nsubj_vec = np.arange(10,101,10)
dim_sides = np.array([25,50,100,200])

# Plot results:
for I in np.arange(3):
    if I == 2:
        plt.plot(nsubj_vec[0:-1], JER_FPR[0:-1,I], label = '(' + str(dim_sides[I]) + ',' + str(dim_sides[I]) + ')')
    else:
        plt.plot(nsubj_vec, JER_FPR[:,I], label = '(' + str(dim_sides[I]) + ',' + str(dim_sides[I]) + ')')
    
plt.legend(loc="lower right")