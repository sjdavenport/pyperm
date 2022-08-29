"""
Testing bootstrapping in the linear model
"""
import pyrft as pr
import numpy as np
import sansouci as sa
Dim = (10,10); N = 30; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); C = np.array([[1,-1,0],[0,1,-1]]); lat_data = pr.wfield(Dim,N)

minP, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C)

orig_pvalues_sorted = np.array([np.sort(np.ravel(orig_pvalues.field))])
    
# Get the minimum p-value over voxels and contrasts (include the orignal in the permutation set)
print(orig_pvalues_sorted[0,0])
    # Obtain the pivotal statistic used for JER control
print(np.amin(sa.t_inv_linear(orig_pvalues_sorted)))

# %%
### One sample, one voxel test
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 20; categ = np.zeros(N)
X = pr.group_design(categ); C = np.array(1); 

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C, 100)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

# %% Multiple contrasts - global null is true
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 30; 
categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); 
C = np.array([[1,-1,0],[0,1,-1]]); 
B = 100

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

# %% Multiple contrasts - global null is false
alpha = 0.1; niters = 1000;
Dim = (50,50); N = 30; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
categ = rng.choice(3, N, replace = True)
X = pr.group_design(categ); 
C = np.array([1,-1,0]); 
B = 100

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I)
    lat_data = pr.wfield(Dim,N)
    minPperm, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters
 
# Note that this give inflated false positives for low N! E.g. N = 30! This gets better
# as N is increased but worse and worse as Dim increases so Anderson may have missed
# it in his FL paper as I'm fairly sure that the tests there were only done in 1D!!

# %%
C = np.array([[1,-1,0]]); 
FWER_FPR, JER_FPR = pr.bootFPR(Dim, N, C)

# %% Multiple contrasts - testing strong control
alpha = 0.1; niters = 1000;
Dim = (10,10); N = 150; 
from sklearn.utils import check_random_state
rng = check_random_state(101)
categ = rng.choice(3, N, replace = True)
X = pr.group_design(categ); 
C = np.array([1,-1,0]); 
w2 = np.where(categ==2) # I.e. there is signal but not in the constrasts of interest
B = 100
signal = 4;

number_of_false_positives = 0
store_origs = np.zeros((1,niters))
for I in np.arange(niters):
    print(I) 
    lat_data = pr.wfield(Dim,N)
    lat_data.field[:,:,w2]=  lat_data.field[:,:,w2] + signal
    minPperm, orig_pvalues, pivotal_stats, _ = pr.boot_contrasts(lat_data, X, C, B)
    alpha_quantile = np.quantile(minPperm, alpha)
    store_origs[0,I] = minPperm[0]
    if minPperm[0] < alpha_quantile:
        number_of_false_positives = number_of_false_positives + 1
        
FPR = number_of_false_positives/niters

# %% Testing bootstorage
import pyrft as pr
import numpy as np
Dim = (8,8); N = 100; categ = np.random.multinomial(2, [1/3,1/3,1/3], size = N)[:,1]
X = pr.group_design(categ); C = np.array([1,-1,0]); lat_data = pr.statnoise(Dim,N,4)
B = 100;

minP, orig_pvalues, pivotal_stats, bs = pr.boot_contrasts(lat_data, X, C, B, 'linear', True, 1)


for b in np.arange(B):
    plt.plot(bs[:,b],color="blue")
    
# Calculate reference families
m = bs.shape[0]
lamb = np.arange(11)/10
k = np.arange(m+1)

for l in np.arange(len(lamb)):
    plt.plot(lamb[l]*k/m,color="black")

