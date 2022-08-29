dim = (4,4); nsubj = 100; C = np.array([[1,-1,0],[0,1,-1]]);
power, power_sd = pr.bootpower(dim, nsubj, C, 4, 0, 100, 1000, 0.8, simtype = 1)