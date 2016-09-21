import numpy, math, random
#%pylab inline
import matplotlib.pyplot as plt

# Parameters:
T0 = 0.728     # temperature
T = 0.5     #temperature for Andersen Thermostat
L = 4.0 # box length
N = 64        # number of particles
M = 48.0      # mass of each particle
h = 0.032      # time step size
steps = 1000  # number of time steps


L=4.0
rho=1.0
N=64
M=48.0
T=2.0
kB=1.0
numpasses_in=10
sigma_in=.075

# Molecular Dynamics Code
# Copy your LJMD code from HW 2 here
def InitPositionCubic(N,L):
  position = numpy.zeros((N,3)) + 0.0
  Ncube = 1
  while(N > (Ncube*Ncube*Ncube)):
    Ncube += 1
  if(Ncube**3 != N):
    print("CubicInit Warning: Your particle number",N, \
          "is not a perfect cube; this may result " \
          "in a lousy initialization")
  rs = float(L)/Ncube
  roffset = float(L)/2 - rs/2
  added = 0
  for x in range(0, Ncube):
    for y in range(0, Ncube):
      for z in range(0, Ncube):
        if(added < N):
          position[added, 0] = rs*x - roffset 
          position[added, 1] = rs*y - roffset 
          position[added, 2] = rs*z - roffset 
          added += 1
  return position

def InitVelocity(N,T0,mass=1.):
  initNDIM = 3
  velocity = numpy.zeros((N,3)) + 0.0
  random.seed(1)
  netP = numpy.zeros((3,)) + 0.
  netE = 0.0
  for n in range(0, N):
    for x in range(0, initNDIM):
      newP = random.random()-0.5
      netP[x] += newP
      netE += newP*newP
      velocity[n, x] = newP
  netP *= 1.0/N
  vscale = math.sqrt(3*N*T0/(mass*netE))
  for n in range(0, N):
    for x in range(0, initNDIM):
      velocity[n, x] = (velocity[n, x] - netP[x]) * vscale
  return velocity


# The simulation will require most of the functions you have already implemented above.
# If it helps you debug, feel free to copy and paste the code here.

# We have written the Verlet time-stepping functions for you below, 'h' is the time step.
# ------------------------------------------------------------------------


def PutInBox(Ri):
  dims = [0,1,2]
  for i in dims:
      if Ri[i] < -L/2:
         Ri[i] = Ri[i]%(L/2)
      if Ri[i] >  L/2:
         Ri[i] = Ri[i]%(-L/2)
  return Ri 

#def PutinBox(Ri):
#    Ri=Ri-L*round(Ri/L)
#    return Ri


def Displacement(Ri,Rj):
  dR = numpy.array([0.0,0.0,0.0])
  Ri=PutInBox(Ri)
  Rj=PutInBox(Rj) 
  dx = Ri[0]-Rj[0]
  dy = Ri[1]-Rj[1]
  dz = Ri[2]-Rj[2]
  x_sign=numpy.sign(dx)
  y_sign=numpy.sign(dy)
  z_sign=numpy.sign(dz)
    
  if abs(L-abs(dx)) < abs(dx):
      dx = -x_sign*(L-abs(dx))
  if abs(L-abs(dy)) < abs(dy):
      dy = -y_sign*(L-abs(dy))
  if abs(L-abs(dz)) < abs(dz):
      dz = -z_sign*(L-abs(dz))
  dR = [dx, dy, dz]      
  return dR


def Distance(Ri,Rj):
  d = 0.0
  # <-- find minimum image distance d here -->
  dR = Displacement(Ri,Rj)
  d = math.sqrt(dR[0]**2+dR[1]**2+dR[2]**2)
  return d

def InternalForce(i, R):
  F = numpy.array([0.0, 0.0, 0.0])
  for iter_1 in range(0,len(R)):
    if iter_1 != i:
        D = Displacement(R[i], R[iter_1])
        dist = Distance(R[i], R[iter_1])
        Fx = 4*(12/(dist**13)-6/(dist**7))*(1/dist)*D[0]
        Fy = 4*(12/(dist**13)-6/(dist**7))*(1/dist)*D[1]
        Fz = 4*(12/(dist**13)-6/(dist**7))*(1/dist)*D[2]
        F[0]=F[0]+Fx
        F[1]=F[1]+Fy
        F[2]=F[2]+Fz
  return F

def ComputeEnergy(R, V):
 
  totalKE = 0.0
  totalU  = 0.0
  for iter_1 in range (0,len(V)):
    velocity_sq = ((V[iter_1][0])**2+(V[iter_1][1])**2+(V[iter_1][2])**2)   
    totalKE = totalKE+.5*M*velocity_sq
    
  for iter_2 in range (0,len(R)):
    for iter_3 in range (iter_2+1,len(R)):
       r=Distance(R[iter_2],R[iter_3])
       totalU = totalU + 4.0*(1.0/(r**12)-1.0/(r**6))
        
  totalE = totalKE + totalU
  return totalU, totalKE, totalE
    

def VMDOut(coordList):
  numPtcls=len(coordList[0])
  outFile=open("myTrajectory.xyz","w")
  outFile.write(str(numPtcls)+"\n")
  for coord in coordList:
    for i in range(0,len(coord)):
       outFile.write(str(i)+" "+str(coord[i][0])+" "+str(coord[i][1])+" "+str(coord[i][2])+"\n")
  outFile.close()

def VerletNextR(r_t,v_t,a_t,h):
  # Note that these are vector quantities. Numpy loops over the coordinates for us.
  r_t_plus_h = r_t + v_t*h + 0.5*a_t*h*h
  return r_t_plus_h

def VerletNextV(v_t,a_t,a_t_plus_h,h):
  # Note that these are vector quantities. Numpy loops over the coordinates for us.
  v_t_plus_h = v_t + 0.5*(a_t + a_t_plus_h)*h
  return v_t_plus_h

def VerletNextV_new(r_t,v_t,a_t,h):
   v_t_plus_h = v_t + a_t*h
   return v_t_plus_h
    
def PairCorrelation(R,bins):
    del_g=L/(2*bins)
    g = numpy.zeros(bins) # replace this with the calculation
    for iter_1 in range (0,len(R)-1):
        for iter_2 in range (iter_1+1,len(R)):
            dR=R[iter_1]-R[iter_2]
            dR=PutInBox(dR)
            r_dist=math.sqrt(dR[0]**2+dR[1]**2+dR[2]**2)
            if r_dist < L/2:
              index_g=int(r_dist/del_g)
              g[index_g]=g[index_g]+2
    
    density=N/(L**3)
    for iter_3 in range(0,bins):
        Part_inshell=(4.0/3.0)*math.pi*(((iter_3+1)*del_g)**3-(iter_3*del_g)**3)
        g[iter_3]=g[iter_3]/(Part_inshell*N*density)
    return g

def LegalKVecs(maxK):
    # calculate a list of legal k vectors
    kList=numpy.zeros((maxK**3,3))
    count=0
    ratio=2*math.pi/L
    
    for i in range (0, maxK):
        for j in range (0, maxK):
            for k in range (0,maxK):   
                    kList[count]=[i*ratio,j*ratio,k*ratio]
                    count=count+1
    return kList

def rhoK(k,R):
    #computes \sum_j \exp(i * k \dot r_j) 
    rhok=0.0
    for i in range (0,len(R)):
        dot_prod = k[0]*R[i][0]+k[1]*R[i][1]+k[2]*R[i][2]
        rhok=rhok+numpy.exp(-1j*dot_prod) 
    return rhok

def Sk(kList,R):
    #computes structure factor for all k vectors in kList
    #and returns a list of them
    sKList=[]
    for i in range(0, len(kList)):
        k_current=kList[i]
        sK=(rhoK(k_current,R)*rhoK(numpy.dot(-1,k_current),R))/N
        sKList.append(sK)
    return sKList

def plotSk(kList,skList):
    kVecsSquared = np.sum(kList*kList, axis=1) # get a list of scalars kvec dot kvec
    kMagList = np.sqrt(kVecsSquared)           # these are the magnitudes

    # This longer code is equivalent, and may help you see what's going on.
    #kMagList = []
    #for k in kList:
    #    kMagList.append(math.sqrt(k[0]**2+k[1]**2+k[2]**2))
    #plt.axis([0,25,0,25])
    plt.xlabel('k')
    plt.ylabel('Intensity of S(k)')
    plt.title('S(k) Peak Intensity')
    plt.plot(kMagList,skList,'.') # you can use whatever marker you want here
    


def VACF_calculate(vel_0,V):
    #In this function, a specific timestep is chosen as the starting point. From here, 
    #the VACF is calculated by taking an average value of V(0)*V(t)
    vacf_t=0
    for i in range (0,len(V)):
        vacf_t=vacf_t+(numpy.dot(vel_0[i],V[i]))
    vacf_t=vacf_t/len(V)
    return vacf_t

def ComputeDiffusionC(vacf_t):
    # Measures observable Diffusion Coefficient
    diff=0.0
    totalTime = 0.0
    for i in range(0,len(vacf_t)):
        diff+=h*vacf_t[i]
        totalTime+=h
    diff=diff/(totalTime)
    return diff            


def AndersenThermostat(V,A,nA,h,T, prob):
    # modify the velocities appropriately
    newV = numpy.zeros(3)
    sigma=numpy.sqrt(T/M)
    if (numpy.random.uniform() < prob):
        newV=[numpy.random.normal(0,sigma),numpy.random.normal(0,sigma),numpy.random.normal(0,sigma)]
    else:
        newV = VerletNextV( V, A, nA, h )
    return newV

def ComputeMomentum(V):
    # Returns the total momentum of the system
    momentum = [0.0, 0.0, 0.0]
    ans = 0.0    
    for i in range(0, len(V)):
        momentum+=M*V[i];
    ans = numpy.sqrt(momentum[0]**2+momentum[1]**2+momentum[2]**2)
    return ans

def CalcPotential(R,i):
    #returns the potential on particle i
    myPotential = 0.0
    for part_count in range (0, len(R)):
        if part_count != i:
            r=Distance(R[i],R[part_count])
            myPotential +=4.0*(1.0/(r**12)-1.0/(r**6))
    return myPotential

def UpdatePosition(R,i,sigma):
    old_pos=R[i].copy()
    new_pos=[0.0,0.0,0.0]
    eta_x=rand.gauss(0.0, sigma)
    eta_y=rand.gauss(0.0, sigma)
    eta_z=rand.gauss(0.0, sigma)
    eta = [eta_x, eta_y, eta_z]
    new_pos=old_pos+eta
    return (old_pos, new_pos, eta)







# Main Loop.  
# ------------------------------------------------------------------------

# R, V, and A are the position, velocity, and acceleration of the atoms
# respectively. nR, nV, and nA are the _next_ positions, velocities, etc.
# You can adjust the total number of timesteps here. 
# You will need to 

def simulate(steps,h):
  R = InitPositionCubic(N, L)
  V = InitVelocity(N, T0, M)
  A = numpy.zeros((N,3))
  P = numpy.zeros((steps))
  
  P[0]=ComputeMomentum(V)
  nR = numpy.zeros((N,3))
  nV = numpy.zeros((N,3)) 
    
  E = numpy.zeros(steps)
  KinE = numpy.zeros(steps)
  T_instant = numpy.zeros(steps)
    
  maxK = 8
  gbins = 50
  equil_t= 100
  kB = 1.0

#  g_sum = numpy.zeros(gbins)
#  sK_sum = numpy.zeros(maxK**3)
#  kList = LegalKVecs(maxK)
#  vacf_out=numpy.zeros(steps-equil_t) 
#  vacf_normed=numpy.zeros(steps-equil_t)
#  vel_0=numpy.zeros((N,3))
  D=0
  prob = .01  # collision probability for Andersen
    
    
  for t in range(0,steps):
    ENergy = ComputeEnergy(R,V)
    E[t]=ENergy[2]
    KinE[t]=ENergy[1]
    T_instant[t]=KinE[t]*(2.0/(3.0*N*kB))
    #if t%1==0:
    #    print(t, ENergy,T_instant[t])
    
    
    for i in range(0,len(R)):
      F    = InternalForce(i, R)
      A[i] = F/M
      nR[i] = VerletNextR( R[i], V[i], A[i], h )
      PutInBox( nR[i] )
      
    for i in range(0,len(R)):
      nF = InternalForce(i, nR)
      nA = nF/M  
      nV[i] = VerletNextV( V[i], A[i], nA, h )
  #    nV[i] = AndersenThermostat(V[i],A[i],nA,h,T, prob)
    
   
 #   if t >= equil_t:
 #       g_sum=g_sum+PairCorrelation(R, gbins)
 #       sKList = Sk(kList,R)
 #       sK_sum=sK_sum+numpy.array(sKList)
        
 #   if t == equil_t:
 #       vel_0=V
 #       vel_norm=VACF_calculate(vel_0,vel_0)
 #       vacf_out[t-equil_t]=VACF_calculate(vel_0,V)
 #       vacf_normed[t-equil_t]=vacf_out[t-equil_t]/vel_norm
 #   if t > equil_t:
 #       vacf_out[t-equil_t]=VACF_calculate(vel_0,V)
 #       vacf_normed[t-equil_t]=vacf_out[t-equil_t]/vel_norm
    
    
 #   P[t]=ComputeMomentum(V)
    R = nR.copy()
    V = nV.copy()
  #g_sum=g_sum/(steps-equil_t)
  #sK_sum=sK_sum/(steps-equil_t)
  #print sK_sum
  #D=ComputeDiffusionC(vacf_out)
  #print "Diffusion constant is " + str(D)
  return R

def CalcPotential(R,i):
    #returns the potential on particle i
    myPotential = 0.0
    for part_count in range (0, len(R)):
        if part_count != i:
            r=Distance(R[i],R[part_count])
            myPotential +=4.0*(1.0/(r**12)-1.0/(r**6))
    return myPotential

def UpdatePosition(R,i,sigma):
    #should change the position of particle i
    # using the Gaussian distribution described above
    #You should return the old position
    # so you can reset if necessary
    old_pos=R[i].copy()
    new_pos=[0.0,0.0,0.0]
    eta_x=numpy.random.normal(0.0, sigma)
    eta_y=numpy.random.normal(0.0, sigma)
    eta_z=numpy.random.normal(0.0, sigma)
    eta = [eta_x, eta_y, eta_z]
    
    new_pos=old_pos+eta
    
    
    return (old_pos, new_pos, eta)


InitialPositions = InitPositionCubic(64, 4.0)
#InitialPositions are lattice positions
Vacancy_Initial = numpy.concatenate([InitialPositions[0:30],InitialPositions[31:64]])    
def MCloop(numpasses, sigma, R):
    # Arguments:
    # numpasses - number of times to loop over particles
    # sigma - standard deviation of next move distance
    Vac_Dist_Old=0.0
    total_steps=0
    accepted_steps=0
    acceptance_ratio=0.0
    random.seed(0)
    PE_per_step=numpy.zeros(numpasses*len(R))
    for pass_num in range(0, numpasses):
        for i in range(0, len(R)):
            V_orig=CalcPotential(R,i)
            (old_position, new_position, eta)=UpdatePosition(R,i,sigma)
            R[i]=new_position
            PutInBox(R[i])
            V_prime=CalcPotential(R,i)
            del_V=V_prime-V_orig
            accept_prob=numpy.exp(-del_V/(kB*T))
            #automatically accepting values that have negative del_V
            test_crit=random.random()
           # print test_crit, accept_prob
            total_steps+=1
            
            if test_crit < accept_prob:
                accepted_steps+=1
            else:
                R[i]=old_position
            
            #Store the necessary variables for creating PE plot
 
            totalU  = 0.0
            for iter_2 in range (0,len(R)):
                for iter_3 in range (iter_2+1,len(R)):
                   r=Distance(R[iter_2],R[iter_3])
                   totalU = totalU + 4.0*(1.0/(r**12)-1.0/(r**6))
            PE_per_step[pass_num*len(R)+i]+=totalU

            #Find Vacancy Position
            for run_var in range(0,len(R)):
                for run_var2 in range(0,len(InitialPositions)):
                    Vac_Dist_New = Distance(R[run_var],InitialPositions[run_var2])
                    if Vac_Dist_New > Vac_Dist_Old:
                        vac_index=run_var2
                        Vac_Dist_Old=Vac_Dist_New
           
        if pass_num%100==0:
            print pass_num
        print vac_index
        
    acceptance_ratio=float(accepted_steps)/float(total_steps)
    print "Acceptance Ratio is = " + str(acceptance_ratio)
    return R, PE_per_step

