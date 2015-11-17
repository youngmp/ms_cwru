# By Youngmin Park MS/BS CWRU 2013
# The code is designed to numerically check the iPRC of the PML model

import matplotlib.pylab as mp
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
mp.rcParams.update({'font.size': 22})
from lib import *
import math

def Am(i):
    """
    returns matrix A in Coombes 2008 for use in odeint
    i: region
    """
    if i==1 or i == 3:
        return np.array([[1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[1], -1.0]])
    if i==2:
        return np.array([[-1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[1], -1.0]])
    if i==4:
        return np.array([[1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[0], -1.0]])


def b_vector(i,iapp):
    """
    returns the vector b for use in odeint
    i: region
    iapp: applied current value
    """
    #print i,iapp
    if i==1 or i == 3:
        return np.array([(iapp-pml_params()[3])/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[1]])
    if i==2:
        return np.array([(1.+iapp)/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[1]])
    if i==4:
        return np.array([(iapp-pml_params()[3])/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[0]])

def checkregion(x0):
    """
    # x0: an arbitrary coordinate in R2
    # checkregion returns the region x0 is in (1,2, or 4)
    """
    x_coord = x0[0]
    
    # check region here
    # these conditions are from pg1103 Coombes 2008
    if x_coord <= pml_params()[4]:
        return 4
    elif x_coord > pml_params()[4] and x_coord <= (1.+pml_params()[3])/2.:
        return 1 # region 2 == region 3
    elif x_coord > (1.+pml_params()[3])/2.:
        return 2

        
def pml_system(x0,t,iapp):
    """
    # x0: initial condition of IVP
    # t: time to evaluate
    get region, select matrix A and vector b.
    """
    region = checkregion(x0)
    #print region
    #print region, np.dot(Am(region),x0)+b_vector(region,iapp)
    return np.dot(Am(region),x0)+b_vector(region,iapp)

def pml_limit_cycle(iapp=.1, max_time=200,
        max_steps=100000):
    """
    original function by Kendrick Shaw.
    This function finds the limit cycle solution of the PML system
    """

    # run for a while                                                                                                                                                  
    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(pml_system,
            [.4,.2], #initial condition
            t, args=(iapp,))

    # calculate the most recent time a new cycle was started                                                                                                           
    x_section = .5 # set to .5 for now for simplicity
    crossings = ((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section)
            * (vals[1:,1] > .4))


    if crossings.sum() < 2:
        raise RuntimeError("No complete cycles")

    # linearly interpolate between the two nearest points                                                                                                              
    crossing_fs = ((vals[1:,0][crossings] - x_section)
            / (vals[1:,0][crossings]-vals[:-1,0][crossings]) )
    crossing_ys = (crossing_fs * vals[:-1,1][crossings]
            + (1-crossing_fs) * vals[1:,1][crossings])
    crossing_times = (crossing_fs * t[:-1][crossings]
            + (1-crossing_fs) * t[1:][crossings])

    # Period T, x0, y0, err
    return ( crossing_times[-1] - crossing_times[-2],x_section, crossing_ys[-1],
            abs(crossing_ys[-1]- crossing_ys[-2]) )


def pml_phase_reset(phi, dx=0., dy=0., iapp=.1,
        steps_per_cycle = 10000, num_cycles = 10, return_intermediates=False,
        y0 = None,x0=None, T = None):
    print phi

    if (y0 is None or x0 is None) or T is None:
        T, x0, y0, error = pml_limit_cycle(iapp)

    steps_before = int(phi * steps_per_cycle) + 1

    # run up to the perturbation
    t1 = np.linspace(0, phi * T, steps_before)
    vals1 = integrate.odeint(pml_system,
            [x0, y0],
            t1, args=(iapp,))
    #print T,x0,y0
    # run after the perturbation                                                                                                           
    t2 = np.linspace(phi * T, T * num_cycles,
            steps_per_cycle * num_cycles - steps_before)
    vals2 = integrate.odeint(pml_system,
            list(vals1[-1,:] + np.array([dx, dy])),
            t2, args=(iapp,))
    
    #x_section = .5 # set to .5 for now for simplicity
    #crossings = ((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section)
    #        * (vals[1:,1] > .4))

    # calculate the most recent time a new cycle was started
    x_section = .5
    #print sum((vals2[:-1,0] > x_section) * (vals2[1:,0] <= x_section))
    crossings = ((vals2[:-1,0] > x_section) * (vals2[1:,0] <= x_section)
            * (vals2[1:,1] > .4))

    if crossings.sum() == 0:
        raise RuntimeError("No complete cycles after the perturbation")
    #print sum(crossings)
    crossing_fs = ((vals2[1:,0][crossings] - x_section)
            / (vals2[1:,0][crossings]-vals2[:-1,0][crossings]) )
    crossing_times = (crossing_fs * t2[:-1][crossings]
            + (1-crossing_fs) * t2[1:][crossings])

    crossing_phases = np.fmod(crossing_times, T)/T #* 2 * math.pi
    #print crossing_phases
    crossing_phases[crossing_phases > .5] -= 1#2*math.pi
    if return_intermediates:
        return dict(t1=t1, vals1=vals1, t2=t2, vals2=vals2,
                crossings=crossings,
                crossing_times=crossing_times,
                crossing_phases=crossing_phases)
    else:
        #print crossing_phases
        return -crossing_phases[-1]

        
    
def main():
    # gather parameters for use in nullcline data
    gamma1, gamma2, C, a, b, b_star, dt = pml_params()
    
    # define applied current
    iapp = .1
    #print gamma1, gamma2, C, a, b, b_star, dt

    # solve for pwl system
    x0 = np.array([.4,.2]) # initiaon condition
    maxiter = 1000 # total number of iterations
    tfinal = 50 # final time
    
    t = np.linspace(0,tfinal,maxiter) # create time array to evaluate odeint
    sol = integrate.odeint(pml_system,x0,t,args=(iapp,)) # solve the IVP
    #print sol
    """
    vrange = [0,.8] # information for nullclines
    # create nullclines
    nullcline_w = nullclinew(vrange=vrange, b_star=b_star, gamma1=gamma1, gamma2=gamma2, b=b, dt=dt)
    nullcline_v = nullclinev(vrange=vrange, iapp=iapp, a=a, C=C, dt=dt)
    # make sure nullclines are culled to equal length
    if len(nullcline_w[0]) > len(nullcline_w[1]): 
        nullcline_w[0] = np.delete(nullcline_w[0], -1)


    mp.figure(1)
    # plot nullclines
    mp.plot(nullcline_w[0],nullcline_w[1]) 
    mp.plot(nullcline_v[0],nullcline_v[1])

    # plot solution and show plot
    mp.plot(sol[:,0],sol[:,1])
    
    mp.figure(3)
    mp.plot(t,sol[:,0])
    """

    # PRC calculations
    n_phis = np.linspace(0, 1, 50)
    mp.figure(2)
    dx = 0.
    dy = 1e-4
    n_prc_o = np.array([
            pml_phase_reset(phi, dx=dx, dy=dy, iapp=iapp,x0=None, y0=None, T=None,
                                  steps_per_cycle=100000)
            for phi in n_phis
            ])
    mp.plot(n_phis,n_prc_o/dy,'o',color='.5',markeredgecolor='0.8')
    mp.plot(n_phis,n_prc_o/dy,'-',color='.5')
    mp.xlabel('$\phi$')
    mp.ylabel('$\Delta\phi$ (normalized)')


    dx = 1e-4
    dy = 0.
    n_prc_o = np.array([
            pml_phase_reset(phi, dx=dx, dy=dy, iapp=iapp,x0=None, y0=None, T=None,
                                  steps_per_cycle=100000)
            for phi in n_phis
            ])
    mp.plot(n_phis,n_prc_o/dx,'bo',markeredgecolor='b')
    mp.plot(n_phis,n_prc_o/dx,'b-')


    #mp.figure(3)

    mp.show()

if __name__ == '__main__':
    main()
    
