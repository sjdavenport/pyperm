  # Obtain the size of the array
  sD = np.shape(data)
  
  # Obtain the dimensions
  Dim = sD[0:-1]
  
  # Obtain the number of dimensions
  D = len(Dim)
  
  # Obtain the number of subjects
  nsubj = sD[-1]
  
  # Get the mean and stanard deviation along the number of subjects
  xbar = data.mean(D) # Remember in python D is the last dimension of a D+1 array
  
  # Calculate the standard deviation (multiplying to ensure the population std is used!)
  std_dev = data.std(D)*np.sqrt((nsubj/(nsubj-1.)))
  
  # Compute Cohen's d
  cohensd = xbar/std_dev
  tstat = np.sqrt(nsubj)*cohensd
  
  return(tstat, xbar, std_dev)