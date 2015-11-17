# Written by Youngmin Park (yxp30@case.edu) MS/BS CWRU 2013
# This file contains all the functions required for various
# piecewise Morris-Lecar (PML) scripts to run.

# Libraries for pml.py
import matplotlib.pylab as mp
import numpy as np
import scipy as sp
import scipy.linalg as linalg
from scipy import integrate
from scipy.optimize import brentq

# ANY CHAGES MADE HERE MUST ALSO BE MADE TO PML.PY

#gamma1 = 2.0
#gamma2 = 0.25
#C = 0.825
#a = 0.25
#b = 0.5
#b_star = 0.2

#vrange = [0, .9] # Voltage range over which to evaluate the program
#wrange = [0, .9] # w range.


# parameter functions - changes made here will be reflected in
# all programs that call this library file.
def pml_params():
	gamma1 = 2.0
	gamma2 = 0.25
	C = 0.825
	a = 0.25
	b = 0.5
	b_star = 0.2
	dt = .001
	return [gamma1, gamma2, C, a, b, b_star, dt]

def limit_cycle_solution(dt=None, init=None, iapp=None, sim_time=None, inside_limit_cycle=None, outside_limit_cycle=None):
        print "Calculating limit cycle"
        print "\tUpper bound:", inside_limit_cycle
        print "\tInitial condition:", init[1]
        print "\tLower bound:", outside_limit_cycle

        winit = brentq(return_map, outside_limit_cycle, inside_limit_cycle, args=(init, dt, iapp, sim_time))

        solutiona, time_of_flighta = periodic_orbit(i=1, sim_time=sim_time, initial_cond=np.array([pml_params()[4], winit]), dt=dt, iapp=iapp)
        solution_vectora = np.array(solutiona)
        wa = solution_vectora[-1][1]

        solutionb, time_of_flightb = periodic_orbit(i=2, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2.0, wa]), dt=dt, iapp=iapp)
        solution_vectorb = np.array(solutionb)
        wb = solution_vectorb[-1][1]

        solutionc, time_of_flightc = periodic_orbit(i=3, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2.0, wb]), dt=dt, iapp=iapp)
        solution_vectorc = np.array(solutionc)
        wc = solution_vectorc[-1][1]

        solutiond, time_of_flightd = periodic_orbit(i=4, sim_time=sim_time, initial_cond=np.array([pml_params()[4], wc]), dt=dt, iapp=iapp)
        solution_vectord = np.array(solutiond)
        wd = solution_vectord[-1][1]

        print "initial condition at iapp = "+str(iapp)+": ("+str(pml_params()[4])+","+str(winit)+")"
        print "Time of flight through region 1: ", time_of_flighta
        print "Time of flight through region 2: ", time_of_flightb
        print "Time of flight through region 3: ", time_of_flightc
        print "Time of flight through region 4: ", time_of_flightd
        print "Limit cycle period: ", time_of_flighta+time_of_flightb+time_of_flightc+time_of_flightd
	
	temp = np.append(solution_vectora, solution_vectorb, 0)
	temp = np.append(temp, solution_vectorc, 0)
	temp = np.append(temp, solution_vectord, 0)

	return [temp, time_of_flighta+time_of_flightb+time_of_flightc+time_of_flightd]


def separatrix_solution(dt=None, sim_time=None, eigenvectoru=None, fixed_point=None, iapp=None):
        print "Calculating separatrix..."
        dt_reverse = -1*dt
        fixed_point_perturbed = fixed_point + (1.0/10000.0)*eigenvectoru

        separatrix_sol4, separatrix_flight4 = periodic_orbit(i=4, sim_time=sim_time, initial_cond=fixed_point_perturbed, dt=dt_reverse, iapp=iapp)
        s4 = separatrix_sol4[-1][1]
        separatrix_sol4 = np.array(separatrix_sol4)

        separatrix_sol3, separatrix_flight3 = periodic_orbit(i=3, sim_time=sim_time, initial_cond=np.array([pml_params()[4],s4]), dt=dt_reverse, iapp=iapp)
        s3 = separatrix_sol3[-1][1]
        separatrix_sol3 = np.array(separatrix_sol3)

        separatrix_sol2, separatrix_flight2 = periodic_orbit(i=2, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2,s3]), dt=dt_reverse, iapp=iapp)
        s2 = separatrix_sol2[-1][1]
        separatrix_sol2 = np.array(separatrix_sol2)

        separatrix_sol1, separatrix_flight1 = periodic_orbit(i=1, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2,s2]), dt=dt_reverse, iapp=iapp)
        s1 = separatrix_sol1[-1][1]
        separatrix_sol1 = np.array(separatrix_sol1)
	initialization_coord = separatrix_sol1[-1][1]

        separatrix_sol5, separatrix_flight5 = periodic_orbit(i=4, sim_time=sim_time, initial_cond=np.array([pml_params()[4],s1]), dt=dt_reverse, iapp=iapp)
        s5 = separatrix_sol5[-1][1]
        separatrix_sol5 = np.array(separatrix_sol5)

	temp = np.append(separatrix_sol4, separatrix_sol3, 0)
	temp = np.append(temp, separatrix_sol2, 0)
	temp = np.append(temp, separatrix_sol1, 0)
	temp = np.append(temp, separatrix_sol5, 0)

	# temp returns total separatrix solution
	# initialization_coord returns last value in separatrix calulation
	return temp, initialization_coord

def coordinate_transform(old_coord_data, iapp):
	# Translate, then transform
	transformed = np.array([[1.245218544034379,-0.7978622303790213],[-0.3779055518316757,1.4298048589659018]])

	if len(old_coord_data)==2 and np.rank(old_coord_data)==1:
		new_coord = np.zeros([2,1])
		#print new_coord
		#print old_coord_data
		new_coord[0] = old_coord_data[0] - 0.4 + 2*iapp
		new_coord[1] = old_coord_data[1] - 0.15 + iapp
		new_coord_final = np.dot(transformed, np.array([new_coord[0], new_coord[1]]))
		return new_coord_final

	elif old_coord_data.shape[1] == 2 and np.rank(old_coord_data)==2:
		old_coord_data[:,0] = old_coord_data[:,0] - 0.4 + 2*iapp
		old_coord_data[:,1] = old_coord_data[:,1] - 0.15 + iapp
		for i in range(len(old_coord_data)):
			old_coord_data[i] = np.dot(transformed, old_coord_data[i])
		return old_coord_data

	elif old_coord_data.shape[0] == 2 and np.rank(old_coord_data)==2:
		old_coord_data[0] = old_coord_data[0] - 0.4 + 2*iapp
		old_coord_data[1] = old_coord_data[1] - 0.15 + iapp
		for i in range(len(old_coord_data[0])):
			old_coord_data[:,i] = np.dot(transformed, old_coord_data[:,i])
		return old_coord_data			

	else:
		raise Exception("couldn't find appropriate manipulation for", old_coord_data.shape)

def coordinate_reverse_transform(old_coord_data, iapp):
	# Translate, then transform
	transformed = np.array([[1.245218544034379,-0.7978622303790213],[-0.3779055518316757,1.4298048589659018]])
	transformed = linalg.inv(transformed)

	if len(old_coord_data)==2 and np.rank(old_coord_data)==1:
		old_coord_data = np.dot(transformed, old_coord_data)
		old_coord_data[0] = old_coord_data[0] + 0.4 - 2*iapp
		old_coord_data[1] = old_coord_data[1] + 0.15 - iapp
		return old_coord_data

	elif old_coord_data.shape[1] == 2 and np.rank(old_coord_data)==2:
		for i in range(len(old_coord_data)):
			old_coord_data[i] = np.dot(transformed, old_coord_data[i])
		old_coord_data[:,0] = old_coord_data[:,0] + 0.4 - 2*iapp
		old_coord_data[:,1] = old_coord_data[:,1] + 0.15 - iapp
		return old_coord_data

	elif old_coord_data.shape[0] == 2 and np.rank(old_coord_data)==2:
		for i in range(len(old_coord_data[0])):
			old_coord_data[:,i] = np.dot(transformed, old_coord_data[:,i])
		old_coord_data[0] = old_coord_data[0] + 0.4 - 2*iapp
		old_coord_data[1] = old_coord_data[1] + 0.15 - iapp
		return old_coord_data

	else:
		raise Exception("couldn't find appropriate manipulation for", old_coord_data.shape)


# Boundary calculations to keep everything as smooth and precise as possible

# Boundary between 1 and 2
def boundary1(x, initial_cond, iapp, dt, final_cond):
	solution1 = calculated_solution(i=1, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
	if dt > 0:
		if final_cond == None:
			solution = solution1[0]-(1+pml_params()[3])/2
			return solution
		else:
			solution = solution1[0]-final_cond[0]
			return solution
	if dt < 0:
		solution = solution1[0]-pml_params()[4]
		return solution

# Boundary between 2 and 3
def boundary2(x, initial_cond, iapp, dt, final_cond):
	solution1 = calculated_solution(i=2, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
	if dt > 0:
		if final_cond == None:
			solution = solution1[0]-(1+pml_params()[3])/2
			return solution
		else:
			solution = solution1[1]-final_cond[1]
			return solution
	if dt < 0:
		solution = solution1[0]-(1+pml_params()[3])/2
		return solution

# Boundary between 3 and 4
def boundary3(x, initial_cond, iapp, dt, final_cond):
	solution1 = calculated_solution(i=3, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
	if dt > 0:
		if final_cond == None:
			solution = solution1[0]-pml_params()[4]
			return solution
		else:
			solution = solution1[0]-final_cond[0]
			return solution

	if dt < 0:
		solution = solution1[0]-(1+pml_params()[3])/2
		return solution

# Boundary between 4 and 1
def boundary4(x, initial_cond, iapp, dt, final_cond):
	solution1 = calculated_solution(i=4, t=x, type='real', initial_cond=initial_cond, iapp=iapp)
	if dt > 0:
		if final_cond == None:
			solution = solution1[0]-pml_params()[4]
			return solution
		else:
			solution = solution1[0]-final_cond[0]
			return solution
	if dt < 0:
		solution = solution1[0]-pml_params()[4]
		return solution

# Function that takes voltage range and param a
# and returns an array of equation f(v) (2.3)

def f():
        # These equations are from (2.3) in Coombes 2008
        assert (type(vrange) is list) and (len(vrange) is 2)

        varray = np.arange(vrange[0], vrange[1], dt)
        if np.round(pml_params()[3]/2, decimals=int(1/dt))-pml_params()[3]/2 == 0:
                border1end = (pml_params()[3]/2)/dt - dt
                border2start = (pml_params()[3]/2)/dt
        else:
                border1end = (pml_params()[3]/2)/dt
                border2start = (pml_params()[3]/2)/dt

        if np.round(((1+pml_params()[3])/2)/dt, decimals=int(1/dt))-((1+pml_params()[3])/2)/dt == 0:
                border2end = ((1+pml_params()[3])/2)/dt
                border3start = ((1+pml_params()[3])/2)/dt + dt
        else:
                border2end = ((1+pml_params()[3])/2)/dt
                border3start = ((1+pml_params()[3])/2)/dt

        varray1 = varray[vrange[0]/dt:border1end]*(-1)
        varray2 = varray[border2start:border2end]-pml_params()[3]
        varray3 = varray[border3start:vrange[1]/dt]*(-1) + 1

        temp = np.append(varray1, varray2)
        temp = np.append(temp, varray3)

        return [varray, temp]

def g():
        varray = np.arange(vrange[0], vrange[1], dt)
        warray = np.arange(wrange[0], wrange[1], dt)
        if np.round(pml_params()[4]/dt, decimals=int(1/dt))-pml_params()[4]/dt == 0:
                border1end = pml_params()[4]/dt - dt
                border2start = pml_params()[4]/dt
        else:
                border1end = pml_params()[4]/dt
                border2start = pml_params()[4]/dt
        garray1 = (varray[vrange[0]/dt:border1end]-pml_params()[0]*warray[wrange[0]/dt:border1end]+pml_params()[5]*pml_params()[0]-pml_params()[4])/pml_params()[0]
        garray2 = (varray[border2start:vrange[1]/dt]-pml_params()[1]*warray[border2start:vrange[1]/dt]+pml_params()[5]*pml_params()[1]-pml_params()[4])/pml_params()[1]

        temp = np.append(garray1, garray2)

        # TEMPORARY MEASURE - PLEASE FIX
        # varray = np.delete(varray, -1)
        # i get array size mismatch for dt = 0.1

        #print np.size(temp), np.size(varray)
        return [varray, temp]


def nullclinew(vrange=None, b_star=None, gamma1=None, gamma2=None, b=None, dt=None):
        varray = np.arange(vrange[0], vrange[1], dt)

        if np.round(b/dt, decimals=int(1/dt))-b/dt == 0:
                border1end = b/dt - dt
                border2start = b/dt
        else:
                border1end = b/dt
                border2start = b/dt
        nullclinew1 = (varray[vrange[0]/dt:border1end]+b_star*gamma1-b)/gamma1
        nullclinew2 = (varray[border2start:vrange[1]/dt]+b_star*gamma2-b)/gamma2

        temp = np.append(nullclinew1, nullclinew2)

        # Varray is the voltage array initiated in this function
        # Temp is the combined piecewise nullcline solution
        return np.array([varray, temp])

def nullclinev(vrange=None, iapp=None, a=None, C=None, dt=None):
        varray = np.arange(vrange[0], vrange[1], dt)
        if np.round(pml_params()[3]/2, decimals=int(1/dt))-pml_params()[3]/2 == 0:
                border1end = (pml_params()[3]/2)/dt - dt
                border2start = (pml_params()[3]/2)/dt
        else:
                border1end = (pml_params()[3]/2)/dt
                border2start = (pml_params()[3]/2)/dt
        if np.round(((1+a)/2)/dt, decimals=int(1/dt))-((1+a)/2)/dt == 0:
                border2end = ((1+a)/2)/dt
                border3start = ((1+a)/2)/dt + dt
        else:
                border2end = ((1+a)/2)/dt
                border3start = ((1+a)/2)/dt

        varray1 = iapp-varray[vrange[0]/dt:border1end]
        varray2 = varray[border2start:border2end]-pml_params()[3]+iapp
        varray3 = 1+iapp-varray[border3start:vrange[1]/dt]
        temp = np.append(varray1, varray2)
        temp = np.append(temp, varray3)

        # Varray is the voltage array initiated in this function
        # Temp is the combined piecewise nullcline solution
        return np.array([varray, temp])


# ith A matrix
def Am(i):
        if i==1 or i == 3:
                return np.array([[1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[1], -1.0]])
        if i==2:
                return np.array([[-1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[1], -1.0]])
        if i==4:
                return np.array([[1.0/pml_params()[2], -1.0/pml_params()[2]],[1.0/pml_params()[0], -1.0]])


def b_vector(i, iapp):
        if i==1 or i == 3:
                return np.array([(iapp-pml_params()[3])/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[1]])
        if i==2:
                return np.array([(1.+iapp)/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[1]])
        if i==4:
                return np.array([(iapp-pml_params()[3])/pml_params()[2], pml_params()[5]-pml_params()[4]/pml_params()[0]])


# G matrix
def Gm(t=None, i=None, type=None): # The first term in equation (3.2) (G(t))
        lambda_plus = lambdam(i)[0][0].real
        lambda_minus = lambdam(i)[0][1].real
        exp_minus = np.exp(lambda_minus * t)
        exp_plus = np.exp(lambda_plus * t)
        if type == 'real':

                # Lambda matrix of ith A matrix
                # It's the diagonalized matrix of A in the paper
                G_matrix = np.zeros((2,2))
                G_matrix[0][0] = (1.0/(lambda_plus - lambda_minus)) * (lambda_plus * exp_plus - lambda_minus * exp_minus - Am(i)[1][1] * (exp_plus - exp_minus))
                G_matrix[0][1] = (-1.0) * ((lambda_plus - Am(i)[1][1])/(lambda_plus - lambda_minus)) * ((lambda_minus - Am(i)[1][1])/Am(i)[1][0]) * (exp_plus - exp_minus)
                G_matrix[1][0] = (Am(i)[1][0]/(lambda_plus - lambda_minus)) * (exp_plus - exp_minus)
                G_matrix[1][1] = (1.0/(lambda_plus - lambda_minus)) * (lambda_plus * exp_minus - lambda_minus * exp_plus + Am(i)[1][1] * (exp_plus - exp_minus))

                return G_matrix

        if type == 'imaginary' or 'im':
                G_matrix = np.zeros((2,2))
                # 'omega' and 'rho' variables are separately declared in Pm type='im'
                # So if I make changes to these expressions, do the same down there
                lambda_imag = lambdam(i)[0][0].imag

                rho = lambda_plus
                omega = lambda_imag
                exponential = np.exp(rho * t)
                omega_hat = omega/Am(i)[0][1]
                rho_hat = (rho-Am(i)[0][0])/Am(i)[0][1]

                coeff = exponential/omega_hat
                sin = np.sin(omega * t)
                cos = np.cos(omega * t)

                G_matrix[0][0] = coeff * (omega_hat * cos - rho_hat * sin)
                G_matrix[0][1] = coeff * sin
                G_matrix[1][0] = coeff * ((-1.0) * (rho_hat * rho_hat + omega_hat * omega_hat) * sin)
                G_matrix[1][1] = coeff * (omega_hat * cos + rho_hat * sin)

                return G_matrix


# K matrix
def Km(type=None, t=None,  i=None, v_vector=False): # K term in (3.2)
        lambda_plus = lambdam(i)[0][0].real
        lambda_minus = lambdam(i)[0][1].real

        exp_minus = np.exp(lambda_minus * t)
        exp_plus = np.exp(lambda_plus * t)
        if type == 'real':
                K_matrix = np.zeros((2,2))

                K_matrix[0][0] = (1.0/(lambda_plus-lambda_minus)) *\
                                (exp_plus - exp_minus -\
                                Am(i)[1][1] * ((exp_plus - 1.0)/lambda_plus-\
                                (exp_minus - 1.0)/lambda_minus))
                K_matrix[0][1] = (-1.0) * ((lambda_plus - Am(i)[1][1])/(lambda_plus - lambda_minus)) *\
                                ((lambda_minus - Am(i)[1][1])/Am(i)[1][0]) *\
                                ((exp_plus - 1.0)/lambda_plus -\
                                (exp_minus - 1.0)/lambda_minus)
                K_matrix[1][0] = (Am(i)[1][0]/(lambda_plus - lambda_minus)) *\
                                ((exp_plus - 1.0)/lambda_plus -\
                                (exp_minus - 1.0)/lambda_minus)
                K_matrix[1][1] = (1./(lambda_plus - lambda_minus)) *\
                                ((lambda_plus/lambda_minus) *\
                                (exp_minus - 1.0) -\
                                (lambda_minus/lambda_plus) *\
                                (exp_plus - 1.0)  +\
                                Am(i)[1][1] * ((exp_plus - 1.0)/lambda_plus -\
                                (exp_minus - 1.0)/lambda_minus))
                return K_matrix
        if type == 'imaginary' or 'im':
                lambda_imag = lambdam(i)[0][0].imag
                K_matrix = np.zeros((2,2))

                rho = lambda_plus
                omega = lambda_imag
                exponential = np.exp(rho * t)

                omega_hat = omega/Am(i)[0][1]
                rho_hat = (rho-Am(i)[0][0])/Am(i)[0][1]

                K_r = (1.0/(rho * rho + omega * omega)) * (rho * (exponential * np.cos(omega * t) - 1.0) + omega * exponential * np.sin(omega * t))
                K_i = (1.0/(rho * rho + omega * omega)) * (omega * (1.0 - exponential * np.cos(omega * t)) + rho * exponential * np.sin(omega * t))

                coeff = 1.0/omega_hat

                K_matrix[0][0] = coeff * (omega_hat * K_r - rho_hat * K_i)
                K_matrix[0][1] = coeff * K_i
                K_matrix[1][0] = coeff * ((-1.0) * (rho_hat * rho_hat + omega_hat * omega_hat) * K_i)
                K_matrix[1][1] = coeff * (omega_hat * K_r + rho_hat * K_i)

                return K_matrix

# returns eigenvalues/matrix for each region.
def lambdam(i):
        eigenvalues, diag_matrix = linalg.eig(Am(i))
        return [eigenvalues, diag_matrix]

# Limit cycle solution section
def calculated_solution(i=None, t=None, type=None, initial_cond=None, iapp=None):
	first_vector = np.dot(Gm(i=i, t=t, type=type), initial_cond)
	second_vector = np.dot(Km(i=i, t=t, type=type), b_vector(i=i, iapp=iapp))
	soltn = first_vector+second_vector
	return soltn


def periodic_orbit(sim_time=None, i=None, initial_cond=None, iteration=None, dt=None, iapp=None, two_regions=False, final_cond=None, egress=None):
		assert (initial_cond != None)
		solution = []
		egress_ingress = 0
		egress_ingress_point = 0
		i = i%4
		if i == 0:
			i = 4
		t = 0.0
		counter = 0
		period = 0.0
		eigenvalue, diag_matrix = lambdam(i)

		if eigenvalue[0].conj() == eigenvalue[0]: # "if the eigenvalue is real..."
			counter_ingress = 0
			type = 'real'
			while t < sim_time:
				soltn = calculated_solution(i=i, t=t, type='real', initial_cond=initial_cond, iapp=iapp)
				solution.append(soltn)
				if final_cond == None:
					if solution[counter][0] > pml_params()[4] and dt < 0:
						del solution[counter]
						x = brentq(boundary4, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='real', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break
					if solution[counter][0] > pml_params()[4] and dt > 0:
						del solution[counter]
						x = brentq(boundary4, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='real', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break

				if final_cond != None and egress:
					if coordinate_transform(solution[counter], iapp)[1] < 0.15 and counter_ingress == 0:
						solution_transformed = coordinate_transform(solution[counter], iapp)
						egress_ingress_point = solution_transformed[0]
						egress_ingress = t
						counter_ingress += 1
					temp = coordinate_transform(solution[counter], iapp)
					if temp[0] > 0.15:
						break #great precision isn't necessary so we break here
						#del solution[counter]
						#x =  brentq(boundary4, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						#soltn = calculated_solution(i=i, t=x, type='real', initial_cond=initial_cond, iapp=iapp)
						#solution.append(soltn)
						#break

				if final_cond != None and egress != True:
					temp = coordinate_transform(solution[counter], iapp)
					temp2 = coordinate_transform(final_cond, iapp)
					if temp[0] > coordinate_transform(final_cond[0], iapp):
						break #great precision isn't necessary so we break here
						#del solution[counter]
						#x =  brentq(boundary4, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						#soltn = calculated_solution(i=i, t=x, type='real', initial_cond=initial_cond, iapp=iapp)
						#solution.append(soltn)
						#break

				t += dt
				counter += 1
				if dt < 0 and t < (-1*sim_time):
					t += 1000.0
				if dt > 0 and t > sim_time:
					t += 1000.0

		if eigenvalue[0].conj() != eigenvalue[0]:
			while t < sim_time:
				soltn = calculated_solution(i=i, t=t, type='im', initial_cond=initial_cond, iapp=iapp)
				solution.append(soltn)
				if i == 1 and final_cond == None:
					if solution[counter][0] < pml_params()[4] and dt < 0:
						del solution[counter]
						x =  brentq(boundary1, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break
					if solution[counter][0] > (1+pml_params()[3])/2 and dt > 0:
						del solution[counter]
						x =  brentq(boundary1, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break
				if i == 1 and final_cond != None:
					if solution[counter][0] > final_cond[0] and dt > 0:
						del solution[counter]
						x =  brentq(boundary1, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break
				if i == 2 and final_cond == None:
					if solution[counter][0] < (1+pml_params()[3])/2 and dt < 0:
						del solution[counter]
						x =  brentq(boundary2, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						soltn = calculated_solution(i=i, t=x, type='im', initial_cond=initial_cond, iapp=iapp)
						solution.append(soltn)
						break
					if solution[counter][0] < (1+pml_params()[3])/2 and dt > 0:
						del solution[counter]
						x =  brentq(boundary2, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						first_vector = np.dot(Gm(i=2, t=x, type='im'), initial_cond)
						second_vector = np.dot(Km(i=2, t=x, type='im'), b_vector(i=2, iapp=iapp))
						solution.append(first_vector + second_vector)
						break
							
				if i == 2 and final_cond != None:
					if solution[counter][1] > final_cond[1] and dt > 0:
						del solution[counter]
						x =  brentq(boundary2, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						first_vector = np.dot(Gm(i=2, t=x, type='im'), initial_cond)
						second_vector = np.dot(Km(i=2, t=x, type='im'), b_vector(i=2, iapp=iapp))
						solution.append(first_vector + second_vector)
						break
				if i == 3 and final_cond == None:
					if solution[counter][0] >= (1+pml_params()[3])/2 and dt < 0:
						del solution[counter]
						x =  brentq(boundary3, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						first_vector = np.dot(Gm(i=3, t=x, type='im'), initial_cond)
						second_vector = np.dot(Km(i=3, t=x, type='im'), b_vector(i=3, iapp=iapp))
						solution.append(first_vector + second_vector)
						break
					if solution[counter][0] <= pml_params()[4] and dt > 0:
						del solution[counter]
						x =  brentq(boundary3, t-15*dt, t+15*dt, args=(initial_cond, iapp, dt, final_cond))
						first_vector = np.dot(Gm(i=3, t=x, type='im'), initial_cond)
						second_vector = np.dot(Km(i=3, t=x, type='im'), b_vector(i=3, iapp=iapp))
						solution.append(first_vector + second_vector)
						break
				if i == 3 and final_cond != None:
					if solution[counter][0] < final_cond[0] and dt > 0:
						del solution[counter]
						x =  brentq(boundary3, t-10*dt, t+10*dt, args=(initial_cond, iapp, dt, final_cond))
						first_vector = np.dot(Gm(i=3, t=x, type='im'), initial_cond)
						second_vector = np.dot(Km(i=3, t=x, type='im'), b_vector(i=3, iapp=iapp))
						solution.append(first_vector + second_vector)
						break
				t += dt
				counter += 1
		time_of_flight = t

		if egress == True:
			solution = [solution, time_of_flight, egress_ingress, egress_ingress_point]

		else:
			solution = [solution, time_of_flight]
		return solution



def return_map(x, init, dt, iapp, sim_time):
        init = np.array([pml_params()[4], x])
        x = init[1]
        solution_vectora = np.array(periodic_orbit(i=1, sim_time=sim_time, initial_cond=init, dt=dt, iapp=iapp)[0])
        va = solution_vectora[-1][1]
        solution_vectorb = np.array(periodic_orbit(i=2, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2.0, va]), dt=dt, iapp=iapp)[0])
       	vb = solution_vectorb[-1][1]
        solution_vectorc = np.array(periodic_orbit(i=3, sim_time=sim_time, initial_cond=np.array([(1+pml_params()[3])/2.0, vb]), dt=dt, iapp=iapp)[0])
       	vc = solution_vectorc[-1][1]
        solution_vectord = np.array(periodic_orbit(i=4, sim_time=sim_time, initial_cond=np.array([pml_params()[4], vc]), dt=dt, iapp=iapp)[0])
       	vd = solution_vectord[-1][1]
        f_x = vd
       	g_x = x - vd
        return g_x


def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return array[idx]
