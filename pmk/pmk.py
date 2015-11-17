# By Youngmin Park MS/BS CWRU 2013
# The code is designed to numerically check the iPRC of the PML model
import matplotlib.pylab as mp
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
mp.rcParams.update({'font.size': 22})
from lib import *
import math

# reminder: pmk_params returns gamma,c,a,dt in that order.

def Am(i):
    """
    returns matrix A in Coombes 2008 for use in odeint
    i: region
    """
    if i==1 or i == 3:
        return np.array([[1./pmk_params()[1], -1./pmk_params()[1]],[1., -pmk_params()[0]]])
    if i==2 or i == 4:
        return np.array([[-1./pmk_params()[1], -1./pmk_params()[1]],[1., -pmk_params()[0]]])


def b_vector(i,iapp):
    """
    returns the vector b for use in odeint
    i: region
    iapp: applied current value
    """
    #print i,iapp
    if i==1 or i == 3:
        return np.array([(iapp-pmk_params()[2])/pmk_params()[1], 0])
    if i==2:
        return np.array([(1.+iapp)/pmk_params()[1], 0])
    if i==4:
        return np.array([iapp/pmk_params()[1],0])

def checkregion(x0):
    """
    # x0: an arbitrary coordinate in R2
    # checkregion returns the region x0 is in (1,2, or 4)
    """
    x_coord = x0[0]
    # pml_params() -> gamma, c, a, dt
    # check region here
    # these conditions are from pg1103 Coombes 2008
    if x_coord < pmk_params()[2]/2:
        return 4
    elif x_coord >= pmk_params()[2]/2 and x_coord <= (1.+pmk_params()[2])/2.:
        return 1 # region 2 == region 3
    elif x_coord > (1.+pmk_params()[2])/2.:
        return 2

# this function may be defunct since it might be easier to solve the ODE instead...
def analytic_fn(x0,t):
    """
    # x0: arbitrary coordinate in R2 (initial condition)
    # t: arbitrary scalar in R+ (time)
    # Returns the point x(t) after x0
    # The soltn in each region is analytic, so no need to treat all of this like an IVP
    """
    pass
        
def pmk_system(x0,t,iapp):
    """
    # x0: initial condition of IVP
    # t: time to evaluate
    get region, select matrix A and vector b.
    """
    region = checkregion(x0)
    #print region,x0
    #print region
    #print region, np.dot(Am(region),x0)+b_vector(region,iapp)
    return np.dot(Am(region),x0)+b_vector(region,iapp)


def pmk_limit_cycle(iapp=.1, max_time=200,
        max_steps=100000):
    """
    original function by Kendrick Shaw.
    This function finds the limit cycle solution of the PML system
    """

    # run for a while                                                                                                                                                  
    t = np.linspace(0, max_time, max_steps)
    vals = integrate.odeint(pmk_system,
            [.6,-.2], #initial condition
            t, args=(iapp,))

    # calculate the most recent time a new cycle was started                                                                                                           
    x_section = .5 # set to .5 for now for simplicity
    #print np.amin(vals[1:,1])
    #print sum((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section) * (vals[1:,1] < .3))
    crossings = ((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section)
            * (vals[1:,1] > .4))

    #crossings_x = ((vals[:-1,0] > x_section) * (vals[1:,0] <= x_section))
    #crossings_y = vals[1:,1]<.3
    #for i in range(len(vals[1:,0])):
    #    if crossings_x[i] == True:
    #        print vals[1:,1][i]
    #    #print vals[1:,0][crossings_x]
    #print crossings
    #print vals[1:,0][crossings], vals[:-1,0][crossings],vals[1:,1][crossings]

    #mp.plot(vals[0:,0],vals[0:,1])
    #mp.show()

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


def pmk_phase_reset(phi, dx=0., dy=0., iapp=.1,
        steps_per_cycle = 10000, num_cycles = 10, return_intermediates=False,
        y0 = None,x0=None, T = None):
    print phi
    
    if (y0 is None or x0 is None) or T is None:
        T, x0, y0, error = pmk_limit_cycle(iapp)

    steps_before = int(phi * steps_per_cycle) + 1

    # run up to the perturbation                                                                                                                                       
    t1 = np.linspace(0, phi * T, steps_before)
    vals1 = integrate.odeint(pmk_system,
            [x0, y0],
            t1, args=(iapp,))

    # run after the perturbation                                                                                                                                       
    t2 = np.linspace(phi * T, T * num_cycles,
            steps_per_cycle * num_cycles - steps_before)
    vals2 = integrate.odeint(pmk_system,
            list(vals1[-1,:] + np.array([dx, dy])),
            t2, args=(iapp,))

    # calculate the most recent time a new cycle was started                                                                                                           
    x_section = .5
    crossings = ((vals2[:-1,0] > x_section) * (vals2[1:,0] <= x_section)
            * (vals2[1:,1] > .4))
    if len(crossings) == 0:
        raise RuntimeError("No complete cycles after the perturbation")
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
    gamma, C, a, dt = pmk_params()
    
    # define applied current
    iapp = .5
    #print gamma1, gamma2, C, a, b, b_star, dt

    # solve for pwl system
    #x0 = np.array([-.2,.6]) # initiaon condition
    #maxiter = 1000 # total number of iterations
    #tfinal = 10 # final time
    
    #t = np.linspace(0,tfinal,maxiter) # create time array to evaluate odeint

    # LIMIT CYCLE
    T, x0, y0, err = pmk_limit_cycle(iapp=iapp)
    sol = integrate.odeint(pmk_system,[x0,y0],np.linspace(0,T,10000),args=(iapp,)) # solve the IVP

    #vrange = [0,.8] # information for nullclines
    # create nullclines
    #nullcline_w = nullclinew(vrange=vrange, b_star=b_star, gamma1=gamma1, gamma2=gamma2, b=b, dt=dt)
    #nullcline_v = nullclinev(vrange=vrange, iapp=iapp, a=a, C=C, dt=dt)
    # make sure nullclines are culled to equal length
    #if len(nullcline_w[0]) > len(nullcline_w[1]): 
    #    nullcline_w[0] = np.delete(nullcline_w[0], -1)

    mp.figure(1)

    mp.text(.51, .32, '1')
    mp.text(.94, .85, '2')
    mp.text(.25, .94, '3')
    mp.text(-.25, .43, '4')


    # plot nullclines
    vrange = np.linspace(0,.6,2)
    mp.plot(vrange,vrange/gamma,color='k',ls='--',lw=2)
    mp.plot(np.linspace(-1,a/2.,2),iapp-np.linspace(-1,a/2.,2),color='green',ls='--',lw=2)
    mp.plot(np.linspace(a/2.,(1+a)/2.,2),np.linspace(a/2.,(1+a)/2.,2)-a+iapp,color='green',ls='--',lw=2)
    mp.plot(np.linspace((1+a)/2.,2,2),1-np.linspace((1+a)/2.,2.,2)+iapp,color='green',ls='--',lw=2)
    #mp.plot(nullcline_v[0], nullcline_v[1],color='green',ls=':')
    #mp.plot(nullcline_w[0], nullcline_w[1],color='k',ls=':')

    mp.axvline(x=a/2, ymin=-.5, ymax=1, color='gray',lw=2)
    mp.axvline(x=(1+a)/2, ymin=-.5, ymax=1, color='gray',lw=2)

    # plot solution and show plot
    mp.plot(sol[:,0],sol[:,1],color='red',lw=3)
    mp.xlabel('v')
    mp.ylabel('w')

    mp.xlim(-.5, 1.2)
    mp.ylim(.3, 1)

    # PRC calculations
    """
    mp.figure(2)
    n_phis = np.linspace(0, 1, 50)
    #mp.figure(3)
    dx = 0.
    dy = 1e-4
    n_prc_o = np.array([
            pmk_phase_reset(phi, dx=dx, dy=dy, iapp=iapp,x0=None, y0=None, T=None,
                                  steps_per_cycle=100000)
            for phi in n_phis
            ])
    mp.plot(n_phis,n_prc_o/dy,'o',color='.5',markeredgecolor='0.8')
    mp.plot(n_phis,n_prc_o/dy,'-',color='.5')


    dx = 1e-4
    dy = 0.
    n_prc_o = np.array([
            pmk_phase_reset(phi, dx=dx, dy=dy, iapp=iapp,x0=None, y0=None, T=None,
                                  steps_per_cycle=100000)
            for phi in n_phis
            ])
    mp.plot(n_phis,n_prc_o/dx,'bo',markeredgecolor='b')
    mp.plot(n_phis,n_prc_o/dx,'b-')



    mp.xlabel('$\phi$')
    mp.ylabel('$\Delta\phi$ (normalized)')
    """
    mp.show()

if __name__ == '__main__':
    main()
    
