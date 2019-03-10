import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib

# bimodal_theta
def bimodal_theta(x, theta):
	para1 = theta[0,:]
	para2 = theta[1,:]

	xdot = -para2*x*(x - para1)*(x + para1)

	return xdot

# observational operator
def h(x):
	y = x # in this case it is just indentity matrix

	return y

# sample params from a prior, which is uniformly distributed in this case
def samp_param(pmin, pmax, pdim, N):
	# pmin, pmax - min and max guesses in column vector pdim by 1
	# N - total number of particles
	# pdim - dimension of the parameter

	theta = np.random.uniform(pmin,pmax,(pdim, N))

	return theta

# Euler-Maruyama scheme to get ensemble truth solution
def EM_hist_theta(f, tvals, dt_truth, xt0, sigma, true_param):
	t0 = tvals[0]
	M = int((tvals[-1] - tvals[0])/dt_truth) # total number of time steps
	N = int((tvals[1] - tvals[0])/dt_truth)  # number of time steps between observations
	dimx = len(xt0) # dimension of state variables
	x = np.zeros((dimx, M+1))
	x[:,0] = xt0

	for n in range(M):
		dx = dt_truth * f(x[:, n], true_param)
		x[:, n+1] = x[:, n] + dx + sigma*np.sqrt(dt_truth)*np.random.normal(0, 1, dimx)

	x = x[:, 0:-1:N]
	return x

# Euler-Maruyama scheme to get a solution
def EM_theta(f, tvals, dt, x, sigma, theta):
	t0 = tvals[0]
	M = int(numpy.ceil((tvals[-1] - tvals[0])/dt))
	
	x1, x2 = x.shape
	# print(np.random.normal(0,1,(x1,x2)))
	
	for n in range(M):
		dx = dt * f(x, theta)
		w = np.random.normal(0,1,(x1,x2))
		x = x + dx + sigma*np.sqrt(dt)*w
		

	return x

# sample from the discrete distribution using unequal probabilities
# and resample to equal probability
def ResampSimp(W, N):
	# W: normalized weight vector
	# N: total number of samples
	cdf = np.cumsum(W) # CDF of particles
	
	rU = np.sort(np.random.uniform(0,1,N)) # draw uniformly distributed random variables

	outIndex = np.zeros((N, 1))
	
	j = 0
	for i in range(N):
		while cdf[j] < rU[i]:
			j = j + 1
			pass
		# for j in range(N):

		# 	if cdf[j] >= rU[i]:
		# 		break
		outIndex[i] = j

	return outIndex



# main function
def pf(true_para, pmin, pmax):
	# true_para - column vector of true values of parameters, these are to be estimated
	# pmin - initial min value of the uniformly distributed particles 
	# pmax - initial max value of the uniformly distributed particles

	# critical parameters
	N = 20000 # number of particles in the filter

	o_sigma = 0.01 # observational error std

	resamp_thresh = 0.5 # threshold for resampling
	wiggle = 0.01 # noise added on resampling for simple resampling scheme

	sigma = 0.05 # std of model noise

	## model parameters and model

	# stochatic bimodal model
	model = bimodal_theta # model system, can be adapted
	mdim = 1 # state dimension

	# truth and observations
	truth = model # truth system

	obsdim = 1 # observations dimension (number of state variable)
	R = o_sigma**2*np.eye(obsdim) # observation covariance matrix

	## Numerical integration parameters
	T = 0.4 # final time
	truth_step = 0.01 # time step for integrating for truth
	obs_step = 0.1 # time interval between observations, coarser than truth
	m = 1
	mdt = m*truth_step # `model' time step

	tvals = np.linspace(0, T, int(T/obs_step)) # time interval
	tdim = len(tvals) # number of time steps in observations
	xt0 = [0.1] # initial condition for truth
	xp = np.random.normal(xt0, o_sigma, mdim) # initial condition for model state variable
	xp = np.matlib.repmat(xp, 1, N) # forecast associated with each particle - dimension mdim by N

	## generate particle distribution
	pdim = len(true_para) # dimension of parameters
	particle = samp_param(pmin, pmax, pdim, N)
	W = 1/N*np.ones((N,1)) # initial weights

	

	## generate truth and observations
	xt = EM_hist_theta(truth, tvals, truth_step, xt0, sigma, true_para)
	obshist = h(xt + o_sigma*np.random.normal(0, 1, (mdim, tdim)))
	# print(obshist)
	# allocation memory for particles and weights ensemble data 
	phist = np.zeros((pdim, N, tdim))
	Whist = np.zeros((N, tdim))
	phist[:,:,0] = particle
	
	Whist[:,0] = W.reshape((N,))

	resampcount = 0 # count how many times the PF resampled

	## THE REAL PARTICLE FILTER STEP
	for tau in range(tdim-1):

		# Forecast step
		xp = EM_theta(model, tvals[tau:tau+2], mdt, xp, sigma, particle)
		
		obs = obshist[:, tau+1] # observation at time tvals[tau+1]

		innov = np.absolute(np.matlib.repmat(obs,1,N) - h(xp))
		
		Wtmp = -0.5*np.sum(innov**2/R[0], axis = 0) #NOTICE ASSUMPTIONS: 1) R is diagonal 2) every diagonal entry is identical  .... ie R = o_sigma*eye(obsdim)
		Wtmp = Wtmp.reshape((N, 1))
		Wmax = np.max(Wtmp, axis = 0)[0]
		
		Wtmp = Wtmp - Wmax
		
		W = W*np.exp(Wtmp)
		
		W = W/np.sum(W) # normalize weights
		
		
		# resampling if resampling threshold is met
		resamp_cond = 1/sum(W**2)/N < resamp_thresh
		if resamp_cond:
			resampcount = resampcount + 1
			p = []
			for Windex in range(len(W)):
				p.append(W[Windex][0])


			sampIndex = np.random.choice(N, N, p = p)
			
			# sampIndex = ResampSimp(W, N)
			for pIndex in range(pdim):
				particle[pIndex,:] = particle[pIndex, sampIndex] + wiggle*np.random.normal(0,1, N)	
			
			xp = xp[:, sampIndex]
			
			W = 1/N*np.ones((N,1))

		phist[:,:,tau+1] = particle
		
		Whist[:, tau+1] = W.reshape((N,))


	return phist, Whist, tvals, resampcount


# main function
def nested_pf(true_para, pmin, pmax):
	# true_para - column vector of true values of parameters, these are to be estimated
	# pmin - initial min value of the uniformly distributed particles 
	# pmax - initial max value of the uniformly distributed particles

	# critical parameters
	N = 200 # number of particles in the filter
	M = 10 # number of state particles per parameter particle

	o_sigma = 0.01 # observational error std

	resamp_thresh = 0.5 # threshold for resampling
	wiggle = 0.01 # noise added on resampling for simple resampling scheme

	sigma = 0.01 # std of model noise

	## model parameters and model

	# stochatic bimodal model
	model = bimodal_theta # model system, can be adapted
	mdim = 1 # state dimension

	# truth and observations
	truth = model # truth system

	obsdim = 1 # observations dimension (number of state variable)
	R = o_sigma**2*np.eye(obsdim) # observation covariance matrix

	## Numerical integration parameters
	T = 1.9 # final time
	truth_step = 0.01 # time step for integrating for truth
	obs_step = 0.1 # time interval between observations, coarser than truth
	m = 1
	mdt = m*truth_step # `model' time step

	tvals = np.linspace(0, T, int(T/obs_step)) # time interval
	tdim = len(tvals) # number of time steps in observations
	xt0 = [0.1] # initial condition for truth
	xp = np.random.normal(xt0, o_sigma, (mdim, M)) # initial condition for model state variable
	xp = np.matlib.repeat(xp[:,:,np.newaxis], N, axis=2) # forecast associated with each particle - dimension mdim by N
	# xp = np.matlib.repmat(xp, 1, N) # forecast associated with each particle - dimension mdim by N


	## generate particle distribution
	pdim = len(true_para) # dimension of parameters
	particle_param = samp_param(pmin, pmax, pdim, N) # initialize parameters
	Wpara = 1/N*np.ones((N,1)) # initial weights, size N (num of particles) by M (num of state particles)
	Wstate = 1/M*np.ones((M,1))

	## generate truth and observations
	xt = EM_hist_theta(truth, tvals, truth_step, xt0, sigma, true_para) # true state solution, size model dimension by time steps 
	obshist = h(xt + o_sigma*np.random.normal(0, 1, (mdim, tdim))) # observation time series, size model dimension by time steps
	
	# allocation memory for parameter particles and weights ensemble data 
	parahist = np.zeros((pdim, N, tdim))
	Wparahist = np.zeros((N, tdim))
	Wstatehist = np.zeros((mdim*M, N, tdim))
	xi = np.zeros((mdim, N))
	parahist[:,:,0] = particle_param
	Wparahist[:,0] = Wpara.reshape((N,))

	resampcount = 0 # count how many times the PF resampled

	## THE REAL PARTICLE FILTER STEP
	for tau in range(tdim-1):

		obs = obshist[:, tau+1]
		# step (a)
		particle_param = particle_param # jittering FIX
		
		# loop through particles
		for ii in range(N):
			# forecasting
			xp[:,:,ii] = EM_theta(model, tvals[tau:tau+1], mdt, xp[:,:,ii], sigma, particle_param[:,ii])			
			xi[ii] = 1/M*np.sum(xp[:,:,ii], axis = 1) # construct xi
			innov_state = np.absolute(np.matlib.repmat(obs,1,M) - h(xp[:,:,ii]))
			Wstatetmp = -0.5*np.sum(innov_state**2)/R[0]
			Wstatemax = np.max(Wstatetmp)
			Wstatetmp = Wstatetmp - Wstatemax
			Wstate = Wstate*np.exp(Wstatetmp.T)
			Wstate = Wstate/np.sum(Wstate)

			# resample
			sampIndex = ResampSimp(Wstate, M)
			xp[:,:,ii] = xp[:,sampIndex,ii]
			Wstate = 1/M*np.ones((M,1)) # construct discrete distribution with equal weights

		# step (b)
		innov = np.absolute(np.matlib.repmat(obs,1,N) - h(xi))
		Wtmp = -0.5*np.sum(innov**2)/R[0] #NOTICE ASSUMPTIONS: 1) R is diagonal 2) every diagonal entry is identical  .... ie R = o_sigma*eye(obsdim)

		Wmax = np.max(Wtmp)
		Wtmp = Wtmp - Wmax
		Wpara = Wpara*np.exp(Wtmp.T)
		Wpara = Wpara/np.sum(Wpara) # normalize weights

		# resampling if resampling threshold is met
		resamp_cond = 1/np.sum(Wpara**2)/N < resamp_thresh
		if resamp_cond:
			resampcount = resampcount + 1
			print(sampIndex)
			sampIndex = ResampSimp(Wpara, N)

			particle_param = particle_param[:, sampIndex] + wiggle*np.random.normal(0,1 (pdim, N))
			xp = xp[:, :, sampIndex]
			Wpara = 1/N*np.ones(N,1)

		# parahist[:,:,tau+1] = particle_param
		
		# Whist[:, tau+1] = W.reshape((N,))


	return particle_param, Wpara

true_para = np.array([[2],[3]])
pmin = np.array([[1.8],[2.8]])
pmax = np.array([[2.2],[3.2]])

# p,W = nested_pf(true_para, pmin, pmax)
phist, Whist, tvals, resampcount = pf(true_para, pmin, pmax)
print(resampcount)

p = phist[:,:,-1]
W = Whist[:,-1]

print(np.sum(p[0,:]*W))
print(np.sum(p[1,:]*W))

### some plotting












	
