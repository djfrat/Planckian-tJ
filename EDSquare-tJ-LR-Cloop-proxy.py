#!/opt/python-3.8.3/bin/python3.8 
####!/usr/bin/python3 

import time, numpy, sys , mpmath , scipy , random , pickle
from scipy.linalg import eigh_tridiagonal,eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.special import gamma
from itertools import product
from os.path import exists
##import tqdm

# calculate all quantities for crossover lines a la Simkovic
# do not use relE0 for compressibility


TypeSpin = "spinful"#"spinful"

np = -1111
LatticeType         = "Generic"
DeltaSztot			= 0 

t1x = (int)( sys.argv[1])
t1y = (int)( sys.argv[2])
t2x = (int)( sys.argv[3])
t2y = (int)( sys.argv[4])
npp  = (int)( sys.argv[5])  # if npp=0 do GC
#nwind = (int)( sys.argv[5])
PotV0 = (float)( sys.argv[6])
PotJa = (float)( sys.argv[7])  
Kinb = (float)( sys.argv[8]) 

#PotH = (float)( sys.argv[9]) # update filenames


PotJb = Kinb**2*PotJa
PotJb = 0.58*PotJa
#PotJb = 1.0
#PotJb = 0.0

#PotJb = (float)( sys.argv[8])  # PotJb is determined automatically by Kinb
PotV = PotV0 # V in units of t
 

# variables
BoundaryConditions = "TBC"
relE0 				= 0 # Poilb, use only for optcond not for muvsN

# --------------------- option flags ------------------------------------
FlagE0shift 		   = 0 # use for each np the minimum/avg E0 over fluxes; compressibility is much more stable / another option is using the average, but minimum should be correct according to Gros
FlagSecondNeighbors    = 1 # always active for frustrated square
FlagOnlyHF    		   = 0 #only store hamiltonians on disk
FlagOneHole			   = 0 #calculate only Ns=-1 and Ns sectors
FlagOnlyEnergies       = 0 #do not calculate optcond
FlagStaticCorrelations = 0
FlagSigmaPerp          = 0 # calculate sigma_xy
# -----------------------------------------------------------------------

#SigmaQuenchedFlag   = 1
RetainedLevels0     = 400#401    # Lanczos states 
nPhi1               = 4
nPhi2               = 4
alpha               = 1.0
Kina                = 1.0 
#Kinb                = 0
Kinb_1              = Kinb       # in units of Kina
Kinb_2              = Kinb       # in units of Kina
wmax                = 6.0		 # binning of sigma(w)



FlagSpinCoupling    = 1


# structural parameters
GC 					= False #but true, keep for basis construction
Zglobal             = 1 #True  # deprecated, now only on Drude weight
ED                  = 0
FTLM                = True      # if Lanczos, do FTLM or LTL
shufflefluxes       = 1         # shuffle flux list for faster partial convergence
cutoffdetbal        = 1.e-6     # 1.e-6 # changing this while keeping small has no effect on results, but 1.e-8 is too small!!!
densmin				= 1.e-3
deltadens           = 1.e-10     #density variation for compressibility  
deltadens0			= 0.01      # mu is piecewise constant if T=0 GC formulas are used      

if FlagOneHole : 
    Betacut			= 100.0      # below this temperature, use T=0 GC formulas
else :
    Betacut 		= 10.0       # T range is more limited if I evaluate all fillings
         
nBin                = 10001      # use odd number to include w=0 
nmu                 = 501
angleJu 			= 0.0       # direction angle in units of π
GramSchmidtflag     = 0         # should orthogonalize only few times
GramSchmidtinterval = 1         # orthogonalize only few times -> time gain is negligible, most effort is in matrix elements
VerboseLanczosSparse = False # debug
ContinueRunAfterCreateHam = 1

# deprecated
Delta  				= 0         # (float)( sys.argv[2])
thrbasis            = 1.45      # hierarchically cutoff basis like Mayr
LanczosTolerance    = 1.e-14 
cutoffdrude         = -1.e160   # -500000#8#12   #0 = pick only physically realistic samples (no circulation, will vanish in TL)
cutoffUV            = 100.      # exclude calculation of matrix elements if energies are too far away (only ED and LTL)

#disorder = Delta * (numpy.random.random( nx*ny ) -0.5 ) 
print ("bins",wmax*2/nBin) 



# ------------------------------------------------------------------------------------------------------
# Define temperatures
# ------------------------------------------------------------------------------------------------------
BetaLine = 1. / numpy.array( [ 0.05 , 0.1, 0.2, 0.5 , 0.8 ,  1.0 , 1.2 , 1.4, 1.6 , 1.8,  2.0 , 2.2 , 2.4 , 2.6 , 2.8  , 3.0, 3.5 , 4.0 , 5.0 , 6.0 , 10.0 , 100.0 ] , dtype = 'float' )
BetaLine = 1. / numpy.arange(0.2,4.2,.2)
#
BetaLine = 1. / numpy.arange(0.02,0.2,.02)
BetaLine = numpy.append( BetaLine , 1. / numpy.arange(0.2,4.2,.2) )
BetaLine = numpy.append( BetaLine , 1./numpy.array ( [5.0,6.0,7.0,8.0,16.0] ) )

#
#BetaLine = 1. / numpy.arange(0.04,0.2,.04)
BetaLine = 1./numpy.array ( [ 0.02 , 0.04 , 0.08 , 0.12 , 0.16 ] )
BetaLine = 1./numpy.array ( [ 0.01 , 0.02 , 0.04 , 0.08 , 0.16 ] )
BetaLine = 1./numpy.array ( [ 0.01 , 0.02 , 0.04 , 0.06 , 0.08 , 0.1 , 0.12 , 0.14 ,  0.16 ] )
BetaLine = 1./numpy.array ( [ 0.03 , 0.04 , 0.06 , 0.08 , 0.1 , 0.12 , 0.14 ,  0.16 ] )
BetaLine = numpy.append( BetaLine , 1. / numpy.arange(0.2,4.2,.2) )
#BetaLine = numpy.array ( [ beta0 ] )
BetaLine = numpy.append( BetaLine , 1./numpy.array ( [6.0,8.0,10.0] ) )
#BetaLine = numpy.append( BetaLine , 1./numpy.array ( [ 0.01 , 0.02 ] ) )

###BetaLine = 1./numpy.array ( [ 0.02 ] )

#Suffix = str(t1x) + "_" + str(t1y) + "_" + str(t2x) + "_" + str(t2y) + "_Ja" + str(PotJa) + "_Jb" + str(PotJb) + "_V" + str(PotV0) + "_M" + str(RetainedLevels0)  + "_" + BoundaryConditions + "_" + TypeSpin + ".dat";
Suffix = "_" + str(t1x) + "_" + str(t1y) + "_" + str(t2x) + "_" + str(t2y) + "_J" + str(PotJa) + "_tp" + str(Kinb) + "_V" + str(PotV0) + "_M" + str(RetainedLevels0)  + "_" + BoundaryConditions + "_" + TypeSpin + ".dat";


def CreateFluxes(  ) :
    global PhiLine
    # Define fluxes    
    PhiLine = numpy.zeros( ( nPhi1 * nPhi2 , 2 ) ) 
    for i1 in numpy.arange( nPhi1  ) :
        for i2 in numpy.arange( nPhi2  ) :
#            PhiLine[ i1 + i2 * nPhi1 , : ] = numpy.array( [ ( nPhi1 > 1 ) * ( i1 / ( nPhi1 - ( nPhi1 > 1 ) ) ) / 2 + 0.25 * ( nPhi1 == 1)  , (nPhi2 > 1 ) * ( i2 / ( nPhi2 - ( nPhi2 > 1 ) ) ) / 2 + 0.25 * ( nPhi2 == 1 ) ] ) # includes point at 0.5, removes asymmetry np even/odd
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ (i1 + 0.5) / nPhi1, (i2 + 0.5) / nPhi2 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ i1 / nPhi1 +0.01, i2 / nPhi2 +0.001] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            if BoundaryConditions == "TBC" : 
            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ (i1 + 0.5) / nPhi1 + 0*1.e-7, (i2 + 0.5) / nPhi2 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
            if BoundaryConditions == "PBC" : PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ 0. , 0.] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            if BoundaryConditions == "PBC" : PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ 0 , 0.5] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ ( i1 + 0.5 ) / nPhi1 / 2.0  , ( i2 + 0.5 )  / nPhi2 / 2.0 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ -numpy.pi/2 , -numpy.pi/2 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ 0.21, 0.23] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ i1 / nPhi1 , i2 / nPhi2 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ i1 / nPhi1 , i2 / nPhi2 ] ) #( i1  / (nPhi1 -1 ) )/2    # poor man's
#            PhiLine[ i1 + i2 * nPhi1  ] = numpy.array( [ 0.01, 0.001] ) 
    if shufflefluxes :
        numpy.random.shuffle (PhiLine)
def CreateLattice(  ) :
    global u1 , u2  , q1, q2, R1, R2, Connectivity #, InvM
    Connectivity = 4

    if LatticeType == "Regular" :
        T1 = numpy.array( [ nx , 0  ] )
        T2 = numpy.array( [ 0  , ny ] )
    elif LatticeType == "Tilted" :
        T1 = numpy.array( [ nx , ny ] )
        T2 = numpy.array( [-ny , nx ] )
#        T2 = numpy.array( [-nx , 2*nx ] )
    elif LatticeType == "Generic" :
        T1 = numpy.array( [ t1x , t1y ] )
        T2 = numpy.array( [ t2x , t2y ] )
        
#        T2 = numpy.array( [ nx // 2  , ny , 0 ] ) # shifted so that 1D stacks can order more isotropically 
    else : 
        print("Bad lattice type")
        exit()
    u0 = numpy.array( [ 0 , 0 ] ) 
    u1 = numpy.array( [ 1 , 0 ] ) 
    u2 = numpy.array( [ 0 , 1 ] ) 

    R1 = T1[0] * u1 + T1[1] * u2
    R2 = T2[0] * u1 + T2[1] * u2


    M = numpy.array( [ R1 , R2 ] )
    InvM = scipy.linalg.inv( M )
    q1 = 2 * numpy.pi * InvM[ : , 0 ]
    q2 = 2 * numpy.pi * InvM[ : , 1 ]

    # total number of sites
    global ns ; ns = (int)( scipy.linalg.det( numpy.array( [ T1 , T2 ]) ) )
    global lx
#    global lx ; lx = ns // (  (LatticeType == "Tilted") * nx  + ( LatticeType == "Regular" ) * ny )    
    c = 1
    if abs( T1[1] ) > abs( T2[1] ) :
        if T1[1] % T2[1] == 0 and T2[1] > 1 : c = T1[1] // T2[1]
        elif T2[1] == 0 : c = T1[0] #nx
    else : 
        if T1[1] > 1 and T2[1] % T1[1] == 0 : c = T2[1] // T1[1]
        elif T1[1] == 0 : c = T2[1] #ny
    lx = abs (ns // c)

    print ("lx = ", lx)

    if lx == 12 : lx = 6
    if lx == 18 : lx = 6
    
#    lx = 4 # WARNING 
    
    print ("lxnew = ", lx)
#    exit()

#    global lx ; lx = nwind
    s = numpy.arange( ns )

    # neighbor table based on Square lattice    

    # all clusters
    global To ; To = numpy.zeros( ( ns , 4 ) , dtype = "int" )
    vecdir = numpy.array( [ [1 , 0 ] , [ 0 , 1 ] , [ -1 , 0 ] , [ 0 , -1 ] ] )
    for di , dv in enumerate( vecdir ) : 
        for i in s :
            r = numpy.array( [ i % lx , i // lx ] )
            v = -1  
            while( ( s == v ).any() == False ) :
                for a in numpy.arange( -2 , 3 ) :
                    for b in numpy.arange( -2 , 3 ) :
                        v = ( r + dv + a * T1 + b * T2 )
                        if ( v[ 0 ] // lx == 0 ) * ( v[ 1 ] // ( ns // lx ) == 0 )  :
                            To[ i , di ] = v @ numpy.array( [ 1 , lx ] )
                break       
                
# print neighbors: this table is wrong for 4 2 -1 3


#    To [ 0 , 2 ] = 13
    
    for i in numpy.arange( ns ):
        print ( i , To [i , 0] , To [i , 1] , To [i , 2 ] , To [i , 3]  )
    

    
    # define r and q lists
    global phi ; phi = numpy.zeros( ( ns , ns ) , dtype = 'complex' )
    for q in numpy.arange( ns ) :
        for r in numpy.arange( ns ) :
            Listr = ( r % lx ) * u1 + ( r // lx ) * u2
            Listq = ( q % lx ) * q1 + ( q // lx ) * q2
            phi[ q , r ] = numpy.exp( 1.0j * numpy.dot( Listq , Listr ) ) / numpy.sqrt( ns )
            

    for q in numpy.arange( ns ) :
        Listq = ( q % lx ) * q1 + ( q // lx ) * q2
        print( q , Listq / (2*numpy.pi) )

    # Ewald summation for any dimension d and any alpha parameter !
    global InvDist , sumVEwald
    InvDist = numpy.zeros( ns , dtype = 'float' )
    d           = 2 # dimension of the system
    Replicas    = 5
    epsilon     = numpy.pi / ns
#    epsilon     = ( 3.0 / lx )**2.0 	# Simone's optimal value of epsilo 	# Simone's optimal value of epsilon 
    v_cell = scipy.linalg.det( M )

    const_A = numpy.pi**( d / 2 ) * epsilon**( ( alpha - d ) / 2) / gamma( alpha / 2 ) / v_cell
    const_B = epsilon**( alpha / 2 ) / gamma( alpha  /2 )
    const_C = 2 * epsilon**( alpha / 2 ) / alpha / gamma( alpha / 2 )

    for site in numpy.arange( ns ) :
        sum_A = 0.0
        sum_B = 0.0
        r = ( site % lx ) * u1 + ( site // lx ) * u2

        for shx in numpy.arange( - Replicas , Replicas + 1 ) :
            for shy in numpy.arange( - Replicas , Replicas + 1 ) :
                Gl = shx * q1 + shy * q2
                nGl = numpy.linalg.norm( Gl )
                if nGl > 1.e-14 :
                    sum_A += ( numpy.cos( numpy.dot( Gl , r ) ) - 1 ) * mpmath.expint( - ( ( d - alpha ) / 2 - 1 ) , nGl**2 / 4 / epsilon )

        for shx in numpy.arange( - Replicas , Replicas + 1 ) :
            for shy in numpy.arange( - Replicas , Replicas + 1 ) :
                xl = shx * R1 + shy * R2

                nxlr = numpy.linalg.norm( xl + r )
                if nxlr > 1.e-14 :
                    sum_B += mpmath.expint( - ( alpha / 2 - 1 ) , nxlr**2 * epsilon )
                nxl = numpy.linalg.norm( xl )
                if nxl > 1.e-14 :
                    sum_B -= mpmath.expint( - ( alpha / 2 - 1 ) , nxl**2 * epsilon  )

        InvDist[ site ] = const_A * sum_A + const_B * sum_B + const_C
    InvDist[ 0 ] = 0.0
    sumVEwald = numpy.average( InvDist )
    # this way there is a local attractive term, it is due to the repulsion by the self-images so it should be there    
    InvDist -= sumVEwald
def CreateBond() :
    global BondA, BondB
    BondA = numpy.zeros( ( ns * Connectivity // 2 , 2 ) , dtype = "int" )
    BondB = numpy.zeros( ( ns * Connectivity // 2 , 2 ) , dtype = "int" )

    for s in numpy.arange( ns ) :
        BondA[ 0 + ( Connectivity // 2 ) * s , : ] = numpy.array( [ s , To[ s , 0 ] ] ) 
        BondA[ 1 + ( Connectivity // 2 ) * s , : ] = numpy.array( [ s , To[ s , 1 ] ] ) 
        BondB[ 0 + ( Connectivity // 2 ) * s , : ] = numpy.array( [ s , To[ To[ s , 0 ] , 1 ] ] ) 
        BondB[ 1 + ( Connectivity // 2 ) * s , : ] = numpy.array( [ s , To[ To[ s , 1 ] , 2 ] ] ) 
def CreateBasis( nploc ) :
    nw = 0
    for x in numpy.arange( 2**ns ) : 
        bitset = bin(x).count("1")
        nw += ( bitset == nploc )

    Basis = numpy.zeros( nw , dtype = "int" )

    nw = 0
    for x in numpy.arange( 2**ns ) :
        bitset = bin(x).count("1")
        if bitset == nploc :
            Basis[ nw ] = x
            nw += 1
            
    BackBasis = { str( x ) : e  for e,x in enumerate( Basis ) } # no longer needed now
    return Basis , BackBasis
def DecimateBasis( nbnew ) :
    newBasis = numpy.zeros( nbnew , dtype = "int" )
    nw = 0
    for u in BasisUp :
        for d in BasisDw :
            if u & d == 0 :
                a = u + (d << ns)
                newBasis[ nw ] = a
                nw += 1

    newBackBasis = { str( a ) : e for e,a in enumerate( newBasis ) }
                
    return newBasis , newBackBasis
def CreateDecimatedStaticCorrelationMatrices( ) :
    global co , corrI, corrJ , corrV , corrP , alloc
    # ------------------------------------------------------------------------------------------------------------
    #               Hopping                     LR  LR_loc                          ( SzSz + S+S- + S-S+ ) * Z 
    # ------------------------------------------------------------------------------------------------------------
    alloc = nb * (  2 * ( Connectivity // 2 )   + 1 + 1     + FlagSpinCoupling * (  28 * ( Connectivity // 2 ) ) ) * ( ns - np ) + 1
    Pocc = np / 2 / ns 
    alloc = nb * (  2 * ( Connectivity // 2 )   + 1 + 1     + FlagSpinCoupling * (  28 * ( Connectivity // 2 ) ) ) * (int) ( np * np / 2 / ns ) + 1
    alloc=max(alloc,100000000)

    #---------------
    # list of processes : 
    #---------------
    #   if (i,j) n.n.
    #   111 ->  Six | 121 ->  Sjx | 211 ->  ni.Six | 221 ->  ni.Sjx
    #   112 ->  Siy | 122 ->  Sjy | 212 ->  ni.Siy | 222 ->  ni.Sjy
    #   113 ->  Siz | 123 ->  Sjz | 213 ->  ni.Siz | 223 ->  ni.Sjz
    #   300 ->       Si . Sj
    #   400 ->  ni . Si . Sj
    #---------------
    #   if (i,j) n.n.n.
    #   511 ->  Six | 521 ->  Sjx | 611 ->  ni.Six | 621 ->  ni.Sjx
    #   512 ->  Siy | 522 ->  Sjy | 612 ->  ni.Siy | 622 ->  ni.Sjy
    #   513 ->  Siz | 523 ->  Sjz | 613 ->  ni.Siz | 623 ->  ni.Sjz
    #   700 ->       Si . Sj
    #   800 ->  ni . Si . Sj
    #---------------


    def donS( state , bond , idx ) : # nS for all the nn of the hole
    #---------------
    # list of processes : 
    #---------------
    #    7 ->  ni.Six 
    #    8 ->  ni.Siy 
    #    9 ->  ni.Siz 


        global co , corrI, corrJ , corrV , corrP

        Diagi = 0.0
        for b in numpy.arange( len( bond ) ):
            niup = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 0 * ns ) ) > 0 )
            nidw = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 1 * ns ) ) > 0 )
            Diagi += 0.5 * ( niup - nidw ) 
        # z-component of spin Si and Sj
        corrI[ co ] = CurrentIndex ; corrJ[ co ] = CurrentIndex ; corrV[ co ] = Diagi ; corrP[ co ] = idx + 9 ; co += 1

        for b in numpy.arange( len( bond ) ):

            siup = bond[ b , 0 ] + 0 * ns
            sidw = bond[ b , 0 ] + 1 * ns

            wiup = 1 << siup
            widw = 1 << sidw

            newstate = state ^ wiup ^ widw
            indexvec = Dict.get (str(newstate))
            if ( state & wiup > 1 ) == 1 and ( state & widw > 1 ) == 0 and indexvec is not None :
                sign = + 1 
                if siup < sidw : 
                    for c in numpy.arange( siup + 1 , sidw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sidw < siup : 
                    for c in numpy.arange( sidw + 1 , siup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 7 ; co += 1
                # y-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = -1j * sign ; corrP[ co ] = idx + 8 ; co += 1

            if ( state & wiup > 1 ) == 0 and ( state & widw > 1 ) == 1 and indexvec is not None :
                sign = + 1 
                if siup < sidw : 
                    for c in numpy.arange( siup + 1 , sidw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sidw < siup : 
                    for c in numpy.arange( sidw + 1 , siup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 7 ; co += 1
                # y-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = +1j * sign ; corrP[ co ] = idx + 8 ; co += 1


   
    def doS( state , bond , idx ) :
        global co , corrI, corrJ , corrV , corrP

        Diagi = 0.0
        Diagj = 0.0
        for b in numpy.arange( len( bond ) ):
            niup = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 0 * ns ) ) > 0 )
            nidw = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 1 * ns ) ) > 0 )
            njup = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 0 * ns ) ) > 0 )
            njdw = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 1 * ns ) ) > 0 )
            Diagi += 0.5 * ( niup - nidw ) 
            Diagj += 0.5 * ( njup - njdw )
        # z-component of spin Si and Sj
        corrI[ co ] = CurrentIndex ; corrJ[ co ] = CurrentIndex ; corrV[ co ] = Diagi ; corrP[ co ] = idx + 13 ; co += 1
        corrI[ co ] = CurrentIndex ; corrJ[ co ] = CurrentIndex ; corrV[ co ] = Diagj ; corrP[ co ] = idx + 23 ; co += 1

        for b in numpy.arange( len( bond ) ):

            siup = bond[ b , 0 ] + 0 * ns
            sidw = bond[ b , 0 ] + 1 * ns

            wiup = 1 << siup
            widw = 1 << sidw

            newstate = state ^ wiup ^ widw   #possibly wrong?
            indexvec = Dict.get (str(newstate))
            if ( state & wiup > 1 ) == 1 and ( state & widw > 1 ) == 0 and indexvec is not None :
                sign = + 1 
                if siup < sidw : 
                    for c in numpy.arange( siup + 1 , sidw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sidw < siup : 
                    for c in numpy.arange( sidw + 1 , siup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 11 ; co += 1
                # y-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = -1j * sign ; corrP[ co ] = idx + 12 ; co += 1

            if ( state & wiup > 1 ) == 0 and ( state & widw > 1 ) == 1 and indexvec is not None :
                sign = + 1 
                if siup < sidw : 
                    for c in numpy.arange( siup + 1 , sidw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sidw < siup : 
                    for c in numpy.arange( sidw + 1 , siup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 11 ; co += 1
                # y-component of spin Si
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = +1j * sign ; corrP[ co ] = idx + 12 ; co += 1

            sjup = bond[ b , 1 ] + 0 * ns
            sjdw = bond[ b , 1 ] + 1 * ns

            wjup = 1 << sjup
            wjdw = 1 << sjdw

            newstate = state ^ wjup ^ wjdw
            indexvec = Dict.get (str(newstate))
            if ( state & wjup > 1 ) == 1 and ( state & wjdw > 1 ) == 0 and indexvec is not None :
                sign = + 1 
                if sjup < sjdw : 
                    for c in numpy.arange( sjup + 1 , sjdw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sjdw < sjup : 
                    for c in numpy.arange( sjdw + 1 , sjup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Sj
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 21 ; co += 1
                # y-component of spin Sj
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = -1j * sign ; corrP[ co ] = idx + 22 ; co += 1

            if ( state & wjup > 1 ) == 0 and ( state & wjdw > 1 ) == 1 and indexvec is not None :
                sign = + 1 
                if sjup < sjdw : 
                    for c in numpy.arange( sjup + 1 , sjdw ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                if sjdw < sjup : 
                    for c in numpy.arange( sjdw + 1 , sjup ):
                        if ( state & ( 1 << c ) > 0 ) : sign = -sign
                # x-component of spin Sj
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] = sign       ; corrP[ co ] = idx + 21 ; co += 1
                # y-component of spin Sj
                corrI[ co ] = CurrentIndex ; corrJ[ co ] = indexvec ; corrV[ co ] =  1j * sign ; corrP[ co ] = idx + 22 ; co += 1

    def doSiSj( state , bond , idx ) :
        global co , corrI, corrJ , corrV , corrP

        Diag = 0.0
        for b in numpy.arange( len( bond ) ):
            niup = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 0 * ns ) ) > 0 )
            nidw = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 1 * ns ) ) > 0 )
            njup = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 0 * ns ) ) > 0 )
            njdw = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 1 * ns ) ) > 0 )
            Diag += 0.5 * ( niup - nidw ) * 0.5 * ( njup - njdw )
        corrI[ co ] = CurrentIndex
        corrJ[ co ] = CurrentIndex
        corrV[ co ] = Diag
        corrP[ co ] = idx
        co += 1

        for b in numpy.arange( len( bond ) ):
            for sect in numpy.arange( 2 ) :
                s0 = bond[ b , 0 ] + sect * ns
                s2 = bond[ b , 1 ] + ( ( 1 + sect ) % 2 )  * ns
                w0 = 1 << s0 ;
                w2 = 1 << s2 ;
                s1 = bond[ b , 0 ] + ( ( 1 + sect ) % 2 )  * ns
                s3 = bond[ b , 1 ] + sect * ns
                w1 = 1 << s1 ;
                w3 = 1 << s3 ;
        
                newstate = state ^ w0 ^ w1 ^ w2 ^ w3
                indexvec = Dict.get (str(newstate))

                if ( state & w0 > 0 ) == 0 and ( state & w2 > 0 ) == 0 and indexvec is not None :
                    sign = + 1
                    if s2 < s3 :
                        for c in numpy.arange( s2 + 1 , s3 ):
                            if ( state & ( 1 << c ) > 0 ) : sign = -sign
                    if s3 < s2 :
                        for c in numpy.arange( s3 + 1 , s2 ):
                            if ( state & ( 1 << c ) > 0 ) : sign = -sign
                    if s0 < s1 :
                        for c in numpy.arange( s0 + 1 , s1 ):
                            if ( ( state ^ w2 ^ w3 ) & ( 1 << c ) > 0 ) : sign = -sign
                    if s1 < s0 :
                        for c in numpy.arange( s1 + 1 , s0 ):
                            if ( ( state ^ w2 ^ w3 ) & ( 1 << c ) > 0 ) : sign = -sign

                    corrI[ co ] = CurrentIndex
                    corrJ[ co ] = indexvec
                    corrV[ co ] = 0.5 * sign
                    corrP[ co ] = idx
                    co += 1
                    

    corrI = numpy.zeros( alloc  , dtype = 'int' )
    corrJ = numpy.zeros( alloc  , dtype = 'int' )
    corrV = numpy.zeros( alloc  , dtype = 'float' )
    corrP = numpy.zeros( alloc  , dtype = 'int' )
    
    direct = '../hamiltonians-tJ-nnn/'
    file = direct + 'corrns_' + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + '_np' + str(np) + "_" + TypeSpin 
    ok  = exists( file + ".npz" )

    if ok == 1 :
        print( "# Correlation matrices found, loading files... ")
        corrs = numpy.load( file + ".npz" )
        corrI = corrs['I']
        corrJ = corrs['J']
        corrV = corrs['V']
        corrP = corrs['P']
    else :
        co = 0
        siteholeref = 0
        sitespinref = To[ siteholeref   , 0  ]
        sitennref  =  To[ siteholeref   , 1  ]
        sitennnref =  To[ sitespinref   , 1  ]

        sitee      =  To[ siteholeref   , 0  ]
        siten      =  To[ siteholeref   , 1  ]
        sitew      =  To[ siteholeref   , 2  ]
        sites      =  To[ siteholeref   , 3  ]

        sitene     =  To[ sitee         , 1  ]
        sitenw     =  To[ siten         , 2  ]
        sitesw     =  To[ sitew         , 3  ]
        sitese     =  To[ sites         , 0  ]


        sitecenter =  To[ To[ To[ 0 , 0 ] , 0 ] , 0]            # for 33-33 cluster
#        sitecenter =  To[ To[ To[ To [ 0 , 1 ] , 1 ] , 0 ] , 0] # for 4004 cluster
#        sitecenter =  To[ To[ 0 , 1 ] , 0]            # for 31-13 cluster

#        sitecenter =  1 # for self-correlation, use column 6
        
        for i , v in enumerate( Basis ) :
                    vup =  v & (2**ns-1)  # pick only ns rightmost bits  
                    vdw =  v >> ns        # pick only ns  leftmost bits  
                    CurrentIndex = i

                    # calcule les matrices Six, Siy, Siz, Sjx , Sjy, Sjz  sans les correler avec n_0 pour n.n.
                    doS   ( v , numpy.array( [ [ sitespinref , sitennref   ] ] ) , 100 )
                    # calcule les matrices Six, Siy, Siz, Sjx , Sjy, Sjz  sans les correler avec n_0 pour n.n.n.
                    doS   ( v , numpy.array( [ [ sitespinref , sitennnref  ] ] ) , 500 )
                    # calcule la matrice Si.Sj sans la correler avec n_0 pour n.n.
                    doSiSj( v , numpy.array( [ [ sitespinref , sitennref   ] ] ) , 300 )
                    # calcule la matrice Si.Sj sans la correler avec n_0 pour n.n.n.
                    doSiSj( v , numpy.array( [ [ sitespinref , sitennnref  ] ] ) , 700 )
                    # calcule la matrice Si.Sj sans la correler avec n_0 pour la plus longue distance sur le sous-réseau opposé
                    doSiSj( v , numpy.array( [ [ 0 , sitecenter  ] ] )          , 1000 )
                    # calcule la matrice Si.Sj sans la correler avec n_0 pour la plus longue distance sur le même sous-réseau
                    doSiSj( v , numpy.array( [ [ 0 , To[ sitecenter , 1 ] ] ] ) , 1100 )
                    # calcule les matrices Six, Siy, Siz, Sjx , Sjy, Sjz  sans les correler avec n_0  sur le sous-réseau opposé
#                    doS   ( v , numpy.array( [ [ 0 , sitecenter  ] ] )          , 1200 )
                    # calcule les matrices Six, Siy, Siz, Sjx , Sjy, Sjz  sans les correler avec n_0  sur le même sous-réseau 
#                    doS   ( v , numpy.array( [ [ 0 , To[ sitecenter , 1 ] ] ] ) , 1400 )

                    # verifie si le trou est bien au site siteref
                    if ( not vup & ( 1 << siteholeref ) and not vdw & ( 1 << siteholeref ) ):
                        # PairIndex : on cree une matrice par paire de sites (si , sj). On les recuperera dans ComputeMatrix. 
                        # operateurs n.Si et n.Sj pour n.n.
                        PairIndex = 200 
                        doS   ( v , numpy.array( [ [ sitespinref , sitennref  ] ] ) , PairIndex )
                        # operateurs n.Si et n.Sj pour n.n.
                        PairIndex = 600 
                        doS   ( v , numpy.array( [ [ sitespinref , sitennnref  ] ] ), PairIndex )
                        # operateurs n.Si.Sj n.n.
                        PairIndex = 400 
                        doSiSj( v , numpy.array( [ [ sitespinref , sitennref  ] ] ) , PairIndex )
                        # operateurs n.Si.Sj n.n.n.
                        PairIndex = 800
                        doSiSj( v , numpy.array( [ [ sitespinref , sitennnref ] ] ) , PairIndex )
                        # operateurs n.Si pour les 8 sites voisins 
                        PairIndex = 900   
                        donS ( v , numpy.array( [ [ sitee  , sitene ] ] ), PairIndex )
                        donS ( v , numpy.array( [ [ sitene , siten  ] ] ), PairIndex + 10 )
                        donS ( v , numpy.array( [ [ siten  , sitenw ] ] ), PairIndex + 20 )
                        donS ( v , numpy.array( [ [ sitenw , sitew  ] ] ), PairIndex + 30 )
                        donS ( v , numpy.array( [ [ sitew  , sitesw ] ] ), PairIndex + 40 )
                        donS ( v , numpy.array( [ [ sitesw , sites  ] ] ), PairIndex + 50 )
                        donS ( v , numpy.array( [ [ sites  , sitese ] ] ), PairIndex + 60 )
                        donS ( v , numpy.array( [ [ sitese , sitee  ] ] ), PairIndex + 70 )
                          
        corrI = corrI [ : co ] #reset size to minimum
        corrJ = corrJ [ : co ]
        corrV = corrV [ : co ]
        corrP = corrP [ : co ]

        numpy.savez_compressed( file , I=corrI , J=corrJ, V=corrV, P=corrP )

        print("# allocated size per sparse matrix", alloc )
        print("# actual size per sparse matrix", co)

def CreateSparseStaticCorrelationMatrices( PairIndex ) :
    tmp = (numpy.copy( corrV )).astype(complex)
    tmp[ corrP != PairIndex ] = 0
    return scipy.sparse.coo_matrix( ( tmp , ( corrI , corrJ ) ) )

def CreateSparseStaticCorrelationMatricesShell ( PairIndex ) :
    tmp = (numpy.copy( corrV )).astype(complex)
    tmp[ corrP % 10 != PairIndex   ] = 0
    return scipy.sparse.coo_matrix( ( tmp , ( corrI , corrJ ) ) )


def CreateDecimatedHamiltonianMatrices( ) :
    global co , matI, matJ , matV , matP , alloc
    # ------------------------------------------------------------------------------------------------------------
    #               Hopping                     LR  LR_loc                          ( SzSz + S+S- + S-S+ ) * Z 
    # ------------------------------------------------------------------------------------------------------------
    alloc = nb * (  2 * ( Connectivity // 2 )   + 1 + 1     + FlagSpinCoupling * (  6 * ( Connectivity // 2 ) ) ) * ( ns - np ) + 1
    Pocc = np / 2 / ns 
    alloc = nb * (  2 * ( Connectivity // 2 )   + 1 + 1     + FlagSpinCoupling * (  6 * ( Connectivity // 2 ) ) ) * (int) ( np * np / 2 / ns ) + 1
    alloc=max(alloc,100000000)
    
    def doHopping( state , bond , idx ) :
        global co , matI, matJ , matV , matP
        for b in numpy.arange( len( bond ) ): 
            s0 = bond[ b , 0 ]
            s1 = bond[ b , 1 ]

            for sect in numpy.arange( 2 ) :
                w0 = 1 << (s0 + sect * ns); # either up or down hops
                w1 = 1 << (s1 + sect * ns); # either up or down hops
                if ( ( state & w0 > 0 ) != ( state & w1 > 0 ) ) : #and indexvec < nw and indexvec >= 0:
                    newstate = state ^ w0 ^ w1  # switches the two positions if one is empty and one is full 
                    indexvec = Dict.get (str(newstate)) 
                    if indexvec is not None :    
                            
                        sign = + 1 
                        if s0 < s1 :
                            for c in numpy.arange( s0 + 1 + sect * ns, s1 + sect * ns):
                                if ( state & ( 1 << c ) > 0 ) : sign = -sign
                        if s1 < s0 :
                            for c in numpy.arange( s1 + 1 + sect * ns, s0 + sect * ns):
                                if ( state & ( 1 << c ) > 0 ) : sign = -sign
                        if state & w0 > 0 : 
                            matI[ co ] = CurrentIndex
                            matJ[ co ] = indexvec
                            matV[ co ] =  sign
                            matP[ co ] = + ( 1 + ( b % ( Connectivity // 2 ) ) + idx * 2 )
                            co += 1
                        else :
                            matI[ co ] = CurrentIndex
                            matJ[ co ] = indexvec
                            matV[ co ] =  sign
                            matP[ co ] = - ( 1 + ( b % ( Connectivity // 2 ) ) + idx * 2 ) 
                            co += 1
    def doSiSj( state , bond , idx ) :
        global co , matI, matJ , matV , matP

        Diag = 0.0
        for b in numpy.arange( len( bond ) ):
            niup = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 0 * ns ) ) > 0 )
            nidw = (float)( state & ( 1 <<  ( bond[ b , 0 ] + 1 * ns ) ) > 0 )
            njup = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 0 * ns ) ) > 0 )
            njdw = (float)( state & ( 1 <<  ( bond[ b , 1 ] + 1 * ns ) ) > 0 )
#            Diag += 0.5 * ( niup - nidw ) * 0.5 * ( njup - njdw )
            Diag += 0.5 * ( niup - nidw ) * 0.5 * ( njup - njdw ) - 0.5 * ( niup + nidw ) * 0.5 * ( njup + njdw )
        matI[ co ] = CurrentIndex
        matJ[ co ] = CurrentIndex
        matV[ co ] = Diag
        matP[ co ] = +( Connectivity + 1 + idx )
        co += 1

        for b in numpy.arange( len( bond ) ):
            for sect in numpy.arange( 2 ) :
                s0 = bond[ b , 0 ] + sect * ns
                s2 = bond[ b , 1 ] + ( ( 1 + sect ) % 2 )  * ns
                w0 = 1 << s0 ;
                w2 = 1 << s2 ;
                s1 = bond[ b , 0 ] + ( ( 1 + sect ) % 2 )  * ns
                s3 = bond[ b , 1 ] + sect * ns
                w1 = 1 << s1 ;
                w3 = 1 << s3 ;
        
                newstate = state ^ w0 ^ w1 ^ w2 ^ w3
                indexvec = Dict.get (str(newstate))

                if ( state & w0 > 0 ) == 0 and ( state & w2 > 0 ) == 0 and indexvec is not None :
                    sign = + 1
                    if s2 < s3 :
                        for c in numpy.arange( s2 + 1 , s3 ):
                            if ( state & ( 1 << c ) > 0 ) : sign = -sign
                    if s3 < s2 :
                        for c in numpy.arange( s3 + 1 , s2 ):
                            if ( state & ( 1 << c ) > 0 ) : sign = -sign
                    if s0 < s1 :
                        for c in numpy.arange( s0 + 1 , s1 ):
                            if ( ( state ^ w2 ^ w3 ) & ( 1 << c ) > 0 ) : sign = -sign
                    if s1 < s0 :
                        for c in numpy.arange( s1 + 1 , s0 ):
                            if ( ( state ^ w2 ^ w3 ) & ( 1 << c ) > 0 ) : sign = -sign

                    matI[ co ] = CurrentIndex
                    matJ[ co ] = indexvec
                    matV[ co ] = 0.5 * sign
                    matP[ co ] = + ( Connectivity + 1 + idx)
                    co += 1

    matI = numpy.zeros( alloc  , dtype = 'int' )
    matJ = numpy.zeros( alloc  , dtype = 'int' )
    matV = numpy.zeros( alloc  , dtype = 'float' )
    matP = numpy.zeros( alloc  , dtype = 'int' )
    
    direct = '../hamiltonians-tJ-nnn/'
    file = direct + 'ham_' + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + '_np' + str(np) + "_" + TypeSpin 
    ok  = exists( file + ".npz" )
     
    if ok == 1 :
        print( "# Hamiltonian found, loading files... ")
        mats = numpy.load( file + ".npz" )
        matI = mats['I']
        matJ = mats['J']
        matV = mats['V']
        matP = mats['P']
    else :
        co = 0
        for i , v in enumerate( Basis ) :
                    vup =  v & (2**ns-1)  # pick only ns rightmost bits  
                    vdw =  v >> ns        # pick only ns  leftmost bits  
                    CurrentIndex = i
                    doHopping( v , BondA , 0 )
                    if FlagSecondNeighbors :    doHopping( v , BondB , 1 )
                    if FlagSpinCoupling :
                        doSiSj( v , BondA , 0 )
                        if FlagSecondNeighbors : doSiSj( v , BondB , 1 )

                    Diag = 0
                    for s0 in numpy.arange( ns ):
                        for s1 in numpy.arange( s0 + 1 , ns ):
                            w0 = 1 << s0 
                            w1 = 1 << s1 
                            newsite = s1
                            for b in numpy.arange( 0 , s0 %  lx ) : newsite = To[ newsite , 2 ] ;
                            for b in numpy.arange( 0 , s0 // lx ) : newsite = To[ newsite , 3 ] ;
                            Diag += ( (float)( vup & w0 > 0) + (float)( vdw & w0 > 0) - densshift) * ( (float)( vup & w1 > 0 ) + (float)( vdw & w1 > 0 ) - densshift) * InvDist[ newsite ]
        
                    matI[ co ] = CurrentIndex
                    matJ[ co ] = CurrentIndex
                    matV[ co ] = Diag 
                    matP[ co ] = Connectivity + 3
                    co += 1
        
        # extra local negative term from replicas # CHECK IF IT IS LEGIT, check if divide by 2: yes see Bagchi!!!
        
                    Diag = 0
                    if TypeSpin == "spinful" : 
                        for s0 in numpy.arange( ns ):
                            w0 = 1 << s0 
                            Diag += 0.5 * ( (float)( vup & w0 > 0) + (float)( vdw & w0 > 0) - densshift) * ( (float)( vup & w0 > 0 ) + (float)( vdw & w0 > 0 ) - densshift) * InvDist[ 0 ]
        
                    matI[ co ] = CurrentIndex
                    matJ[ co ] = CurrentIndex
                    matV[ co ] = Diag
                    matP[ co ] = Connectivity + 3
                    co += 1



#       # perpendicular magnetic field (might be incompatible with Sz=0)
#                    Diag = 0
#                    for s0 in numpy.arange( ns ):
#                        w0 = 1 << s0 
#                        Diag += 0.5 * ( (float)( vup & w0 > 0) - (float)( vdw & w0 > 0) )
#                            
#                    matI[ co ] = CurrentIndex
#                    matJ[ co ] = CurrentIndex
#                    matV[ co ] = Diag
#                    matP[ co ] = Connectivity + 4
#                    co += 1




    
    # constrains full size (bottom right corner)
        matI[ co ] = nb - 1
        matJ[ co ] = nb - 1
        matV[ co ] = 0
        matP[ co ] = Connectivity + 3
        co += 1

        matI = matI [ : co ] #reset size to minimum
        matJ = matJ [ : co ]
        matV = matV [ : co ]
        matP = matP [ : co ]

        numpy.savez_compressed( file , I=matI , J=matJ, V=matV, P=matP )

#        numpy.savez_compressed( fileI , arr=matI )
#        numpy.savez_compressed( fileJ , arr=matJ )
#        numpy.savez_compressed( fileV , arr=matV )
#        numpy.savez_compressed( fileP , arr=matP )
    
        print("# allocated size per sparse matrix", alloc )
        print("# actual size per sparse matrix", co)
def CreateSparseHamiltonian( ) :

    FluxVec = Phi1 * q1 + Phi2 * q2



    ab_nn_1   = numpy.exp(  1.0j * FluxVec @ u1 )
    ab_nn_2   = numpy.exp(  1.0j * FluxVec @ u2 )
    ab_nnn_1  = numpy.exp( 1.0j * FluxVec @ (  u1 + u2 ) )
    ab_nnn_2  = numpy.exp( 1.0j * FluxVec @ ( -u1 + u2 ) )   

#    print (FluxVec, "fluxvec H")
#    print ("chez moi", ab_nnn_2 )
#    print (-Kinb_1*ab_nnn_1 , -Kinb_2*ab_nnn_2, "second shell H")

    tmp = (numpy.copy( matV )).astype(complex)

    # index convention :
    #   +/-   1 : n.n.   fermion hopping along  ux      direction
    #   +/-   2 : n.n.   fermion hopping along  uy      direction
    #   +/-   3 : n.n.n. fermion hopping along  ux + uy direction
    #   +/-   4 : n.n.n. fermion hopping along -ux + uy direction
    #   Z+1 = 5 : Hubbard interection term. Also fixes Ja and Jb as ( -kin^2 ) / U as in the perturbation theory.
    #   Z+2 = 6 : n.n.   spin coupling Ja
    #   Z+3 = 7 : n.n.n. spin coupling Jb
    
    tmp[ matP == 1 ] *= - Kina * ab_nn_1
    tmp[ matP ==-1 ] *= - Kina * ab_nn_1.conj()
    tmp[ matP == 2 ] *= - Kina * ab_nn_2
    tmp[ matP ==-2 ] *= - Kina * ab_nn_2.conj()
    if FlagSecondNeighbors == True : 
        tmp[ matP == 3 ] *= - Kinb_1  * ab_nnn_1     
        tmp[ matP ==-3 ] *= - Kinb_1  * ab_nnn_1.conj()
        tmp[ matP == 4 ] *= - Kinb_2  * ab_nnn_2     
        tmp[ matP ==-4 ] *= - Kinb_2  * ab_nnn_2.conj()
    tmp[ matP == Connectivity + 1 ] *= PotJa
    tmp[ matP == Connectivity + 2 ] *= PotJb
    tmp[ matP == Connectivity + 3 ] *= PotV 
#    tmp[ matP == Connectivity + 4 ] *= PotH 
    
    
#    h= open ( "matrix"+ Suffix +".txt" , 'w' ) 
    
    
    return scipy.sparse.coo_matrix( ( tmp , ( matI , matJ ) ) )
def CreateSparseKineticMatrix(  ) :
    FluxVec = Phi1 * q1 + Phi2 * q2

    ab_nn_1   = numpy.exp(  1.0j * FluxVec @ u1 )
    ab_nn_2   = numpy.exp(  1.0j * FluxVec @ u2 )
    ab_nnn_1  = numpy.exp( 1.0j * FluxVec @ (  u1 + u2 ) )
    ab_nnn_2  = numpy.exp( 1.0j * FluxVec @ ( -u1 + u2 ) )   #check!!!!

#    tmp = numpy.copy( matV ) + 0.0j  
    tmp = (numpy.copy( matV )).astype(complex)
#    print ("sizetmpK - Mb " , (tmp.nbytes ) / 1024**2 )
    
    tmp[ matP == 1 ] *=  ab_nn_1
    tmp[ matP ==-1 ] *=  ab_nn_1.conj()
    tmp[ matP == 2 ] *=  ab_nn_2
    tmp[ matP ==-2 ] *=  ab_nn_2.conj()
    if FlagSecondNeighbors == True : 
        tmp[ matP == 3 ] *=  Kinb_1 * ab_nnn_1     
        tmp[ matP ==-3 ] *=  Kinb_1 * ab_nnn_1.conj()
        tmp[ matP == 4 ] *=  Kinb_2 * ab_nnn_2     
        tmp[ matP ==-4 ] *=  Kinb_2 * ab_nnn_2.conj()
#    tmp[ P == 5 ] = 0.0
    tmp[ matP == Connectivity + 1 ] = 0.0 
    tmp[ matP == Connectivity + 2 ] = 0.0 
    tmp[ matP == Connectivity + 3 ] = 0.0 
#    tmp[ P == (  Connectivity // 2 ) * ( 1 + ( FlagSecondNeighbors == True )  )  + 1 ]  = 0.0 

    
    return scipy.sparse.coo_matrix( ( tmp , ( matI , matJ ) ) )
def CreateSparseDrudeMatrixJu(  ) :

 
    FluxVec = Phi1 * q1 + Phi2 * q2
    
    ab_nn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ u1 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u1 ) + R2 [0] * ( q2 @ u1 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u1 ) + R2 [1] * ( q2 @ u1 ) ) )
    ab_nn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ u2 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u2 ) + R2 [0] * ( q2 @ u2 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u2 ) + R2 [1] * ( q2 @ u2 ) ) )
 
    ab_nnn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u1 ) + R2 [0] * ( q2 @ u1 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u1 ) + R2 [1] * ( q2 @ u1 ) ) )
    ab_nnn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u2 ) + R2 [0] * ( q2 @ u2 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u2 ) + R2 [1] * ( q2 @ u2 ) ) )
#    ab_nnn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ ( u1 + u2 ) ) + R2 [0] * ( q2 @ ( u1 + u2 ) ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ ( u1 + u2 ) ) + R2 [1] * ( q2 @ ( u1 + u2 ) ) ) )
#    ab_nnn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ ( u2 - u1 ) ) + R2 [0] * ( q2 @ ( u2 - u1 ) ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ ( u2 - u1 ) ) + R2 [1] * ( q2 @ ( u2 - u1 ) ) ) )
    ab_nnn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) )  * ( numpy.cos( numpy.pi * angleJu ) * u1 + numpy.sin( numpy.pi * angleJu ) * u2 ) @ ( u1 + u2  )  
    ab_nnn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) )  * ( numpy.cos( numpy.pi * angleJu ) * u1 + numpy.sin( numpy.pi * angleJu ) * u2 ) @ ( -u1 + u2  )  


    tmp = (numpy.copy( matV )).astype(complex)
    
    tmp[ matP == 1 ] *= Kina * ab_nn_1
    tmp[ matP ==-1 ] *= Kina * ab_nn_1.conj()
    tmp[ matP == 2 ] *= Kina * ab_nn_2
    tmp[ matP ==-2 ] *= Kina * ab_nn_2.conj()
    if FlagSecondNeighbors == True :
        tmp[ matP == 3 ] *= Kinb_1 * ab_nnn_1
        tmp[ matP ==-3 ] *= Kinb_1 * ab_nnn_1.conj()
        tmp[ matP == 4 ] *= Kinb_2 * ab_nnn_2
        tmp[ matP ==-4 ] *= Kinb_2 * ab_nnn_2.conj()
#    tmp[ P == 5 ] = 0.0
    tmp[ matP == Connectivity + 1 ] = 0.0 
    tmp[ matP == Connectivity + 2 ] = 0.0 
    tmp[ matP == Connectivity + 3 ] = 0.0 
#    tmp[ P == (  Connectivity // 2 ) * ( 1 + ( FlagSecondNeighbors == True )  )  + 1 ]  = 0.0 
     
    return scipy.sparse.coo_matrix( ( tmp , ( matI , matJ ) ) )
    
    
    

def CreateSparseDrudeMatrixJuperp(  ) :

    FluxVec = Phi1 * q1 + Phi2 * q2

    angleJuperp = angleJu + 0.5

    ab_nn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ u1 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJuperp ) * (  R1 [0] * (  q1 @ u1 ) + R2 [0] * ( q2 @ u1 ) )  +  numpy.sin( numpy.pi * angleJuperp ) * (  R1 [1] * (  q1 @ u1 ) + R2 [1] * ( q2 @ u1 ) ) )
    ab_nn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ u2 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJuperp ) * (  R1 [0] * (  q1 @ u2 ) + R2 [0] * ( q2 @ u2 ) )  +  numpy.sin( numpy.pi * angleJuperp ) * (  R1 [1] * (  q1 @ u2 ) + R2 [1] * ( q2 @ u2 ) ) )
 
#    ab_nnn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJuperp ) * (  R1 [0] * (  q1 @ u1 ) + R2 [0] * ( q2 @ u1 ) )  +  numpy.sin( numpy.pi * angleJuperp ) * (  R1 [1] * (  q1 @ u1 ) + R2 [1] * ( q2 @ u1 ) ) )
#    ab_nnn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJuperp ) * (  R1 [0] * (  q1 @ u2 ) + R2 [0] * ( q2 @ u2 ) )  +  numpy.sin( numpy.pi * angleJuperp ) * (  R1 [1] * (  q1 @ u2 ) + R2 [1] * ( q2 @ u2 ) ) )
    ab_nnn_1 = 1.0j * numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ ( u1 + u2 ) ) + R2 [0] * ( q2 @ ( u1 + u2 ) ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ ( u1 + u2 ) ) + R2 [1] * ( q2 @ ( u1 + u2 ) ) ) )
    ab_nnn_2 = 1.0j * numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ ( u2 - u1 ) ) + R2 [0] * ( q2 @ ( u2 - u1 ) ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ ( u2 - u1 ) ) + R2 [1] * ( q2 @ ( u2 - u1 ) ) ) )

    tmp = (numpy.copy( matV )).astype(complex)
    
    tmp[ matP == 1 ] *= ab_nn_1
    tmp[ matP ==-1 ] *= ab_nn_1.conj()
    tmp[ matP == 2 ] *= ab_nn_2
    tmp[ matP ==-2 ] *= ab_nn_2.conj()
    if FlagSecondNeighbors == True :
        tmp[ matP == 3 ] *= Kinb_1 * ab_nnn_1
        tmp[ matP ==-3 ] *= Kinb_1 * ab_nnn_1.conj()
        tmp[ matP == 4 ] *= Kinb_2 * ab_nnn_2
        tmp[ matP ==-4 ] *= Kinb_2 * ab_nnn_2.conj()
#    tmp[ P == 5 ] = 0.0
    tmp[ matP == Connectivity + 1 ] = 0.0 
    tmp[ matP == Connectivity + 2 ] = 0.0 
    tmp[ matP == Connectivity + 3 ] = 0.0 
#    tmp[ P == (  Connectivity // 2 ) * ( 1 + ( FlagSecondNeighbors == True )  )  + 1 ]  = 0.0 
     
    return scipy.sparse.coo_matrix( ( tmp , ( matI , matJ ) ) )
    
   
    
def CreateSparseDrudeMatrixKu(  ) :
    FluxVec = Phi1 * q1 + Phi2 * q2

    ab_nn_1 =  numpy.exp(  1.0j * FluxVec @ u1 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u1 ) + R2 [0] * ( q2 @ u1 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u1 ) + R2 [1] * ( q2 @ u1 ) ) )
    ab_nn_2 =  numpy.exp(  1.0j * FluxVec @ u2 ) / 2 / numpy.pi  * ( numpy.cos( numpy.pi * angleJu ) * (  R1 [0] * (  q1 @ u2 ) + R2 [0] * ( q2 @ u2 ) )  +  numpy.sin( numpy.pi * angleJu ) * (  R1 [1] * (  q1 @ u2 ) + R2 [1] * ( q2 @ u2 ) ) )
 
    ab_nnn_1 =  numpy.exp(  1.0j * FluxVec @ (  u1 + u2 ) )  * ( numpy.cos( numpy.pi * angleJu ) * u1 + numpy.sin( numpy.pi * angleJu ) * u2 ) @ ( u1 + u2  )  
    ab_nnn_2 =  numpy.exp(  1.0j * FluxVec @ ( -u1 + u2 ) )  * ( numpy.cos( numpy.pi * angleJu ) * u1 + numpy.sin( numpy.pi * angleJu ) * u2 ) @ ( -u1 + u2  )  
 


#    tmp = numpy.copy( matV ) + 0.0j  
    tmp = (numpy.copy( matV )).astype(complex)
#    print ("sizetmpKu - Mb " , (tmp.nbytes ) / 1024**2 )
    
    tmp[ matP == 1 ] *= Kina * ab_nn_1
    tmp[ matP ==-1 ] *= Kina * ab_nn_1.conj()
    tmp[ matP == 2 ] *= Kina * ab_nn_2
    tmp[ matP ==-2 ] *= Kina * ab_nn_2.conj()
    if FlagSecondNeighbors == True : 
        tmp[ matP == 3 ] *= Kinb_1 * ab_nnn_1
        tmp[ matP ==-3 ] *= Kinb_1 * ab_nnn_1.conj()
        tmp[ matP == 4 ] *= Kinb_2 * ab_nnn_2
        tmp[ matP ==-4 ] *= Kinb_2 * ab_nnn_2.conj()
#    tmp[ P == 5 ] = 0.0
    tmp[ matP == Connectivity + 1 ] = 0.0 
    tmp[ matP == Connectivity + 2 ] = 0.0 
    tmp[ matP == Connectivity + 3 ] = 0.0 
#    tmp[ P == (  Connectivity // 2 ) * ( 1 + ( FlagSecondNeighbors == True )  )  + 1 ]  = 0.0 
     
    return scipy.sparse.coo_matrix( ( tmp , ( matI , matJ ) ) )







def CreateSparseDisorderHamiltonian (arraydis)  :
    Idis = numpy.zeros( nb  , dtype = 'int' )
    Jdis = numpy.zeros( nb  , dtype = 'int' )
    #    nw = len( Basis )
    Vdis = numpy.zeros( nb  , dtype = 'complex' )
    
    co = 0 
    for w in numpy.arange( nb ):
        state = Basis[ w ]
        Diag = 0    

        for s0 in numpy.arange( ns ): 
            for t0 in numpy.arange( 2 ) :
                w0 = 1 << s0 + t0 * ns
                Diag += (float)( state & w0 > 0) * arraydis [ s0 ]

        Idis[ co ] = w
        Jdis[ co ] = w
        Vdis[ co ] = Diag

        co += 1
        
    return scipy.sparse.coo_matrix( ( Vdis , ( Idis , Jdis ) ) )
def CreateBinningDOS( mhl , mybinnumber , nPhidone , name ) :
    mhl = mhl.reshape( ( mhl.shape[0] * mhl.shape[1] , mhl.shape[2] ) )
#    global hist , bin_edges
#    ran = PotJ + 8.0 #optimal range in 1D
#    ran = PotJ + 16.0 #optimal range in 2D
    ran1 = numpy.amin( mhl [ :, 0] ) #automatic
    ran2 = numpy.amax( mhl [ :, 0] ) #automatic
    histdos , bin_edgesdos =  numpy.histogram( mhl[ : , 0 ] , mybinnumber , weights = mhl[ : ,  1 ] / nPhidone , range = (ran1,ran2)) # Symmetric
#    hist , bin_edges = numpy.asarray( numpy.histogram( mhl[ : , 0 ] , mybinnumber , weights = mhl[ : ,  1 ] /  nPhidone ) )   # Automatic
    deltabin = bin_edgesdos [1] - bin_edgesdos[0]
    bin_edgesdos += deltabin/2  
#    hist = hist/deltabin #if it's done here don't do it in Mathematica
    
    orig_stdout = sys.stdout
    f = open( name , 'w' )
    sys.stdout = f
    for ele in numpy.arange( len( histdos ) ) :
        print( bin_edgesdos[ ele ] , histdos[ ele ] ) #prints left of interval
    sys.stdout = orig_stdout
    f.close()
def JustCreateBinning( mhl , mybinnumber   ) :
    ran = PotJa + PotJb + PotV + 16.0 #optimal range in 2D
# this sums all up to nPhi!
##    mhl = mhl.reshape( ( mhl.shape[0] * mhl.shape[1] , mhl.shape[2] ) )
#    ran = PotJ + 8.0 #optimal range in 1D
    ran = 6.0
    ran = wmax
#    ran = numpy.amax( mhl [ :, 0] ) #automatic (but too wide)
    justhist , justbin_edges =  numpy.histogram( mhl[ : , 0 ] , mybinnumber , weights = mhl[ : ,  1 ] , range = (-ran,ran)) # Symmetric
 
    return  justhist , justbin_edges
def JustWrite( jhist , jbin , name ) :
    orig_stdout = sys.stdout
    f = open( name , 'w' )
    sys.stdout = f
    for ele in numpy.arange( len( jhist ) ) :
        print( jbin[ ele ] , jhist[ ele ] ) #prints left of interval
    sys.stdout = orig_stdout
    f.close()
def CreateBinning( mhl , mybinnumber , nPhidone , name ) :
    global hist , bin_edges
    mhl = mhl.reshape( ( mhl.shape[0] * mhl.shape[1] , mhl.shape[2] ) )
#    ran = PotJ + 8.0 #optimal range in 1D
    ran = PotJa + PotJb + PotV + 16.0 #optimal range in 2D
    ran = 6.0
    ran = wmax
#    ran = numpy.amax( mhl [ :, 0] ) #automatic (but too wide)
    hist , bin_edges =  numpy.histogram( mhl[ : , 0 ] , mybinnumber , weights = mhl[ : ,  1 ] / nPhidone , range = (-ran,ran)) # Symmetric
#    hist , bin_edges = numpy.asarray( numpy.histogram( mhl[ : , 0 ] , mybinnumber , weights = mhl[ : ,  1 ] /  nPhidone ) )   # Automatic
    
    orig_stdout = sys.stdout
    f = open( name , 'w' )
    sys.stdout = f
    for ele in numpy.arange( len( hist ) ) :
        print( bin_edges[ ele ] , hist[ ele ] ) #prints left of interval
#        print( bin_edges[ ele ] , hist[ ele ] ,  hist[ ele ]**2 ) # this should calculate errorbars

#    arrsave = numpy.column_stack((bin_edges [ 0 : mybinnumber], hist )) 
#    numpy.savetxt(name, arrsave, delimiter='	')
    sys.stdout = orig_stdout
    f.close()
def WriteResults( betas ,dru,dru1, dru2, proxy, zeta,  divide, name ) :
    orig_stdout = sys.stdout
    g = open( name , 'w' )
    sys.stdout = g
    print ("# BC = ", BoundaryConditions)
    print ("# relE0 = ", relE0)
    print ("# E0shift = ", FlagE0shift)   
    print ("# alpha = ", alpha)
    print ("# sumVEwald = ", sumVEwald)
    print ("# lx = ", lx)
    print ("# SecondNeighbors = ", FlagSecondNeighbors)
    print ("# Kina Kinb = ", Kina , Kinb)
    
    
    if not ED  :
        print("# basis size = ",nb)                  
        print("# Nlevels = ",RetainedLevels)
        print("# Gram-Schmidt = ", GramSchmidtflag)
    else :
        print("# basis size = ",nb)                  
#    if ED :
    print("# Zglobal = ", Zglobal)    
    print("# UV optcond cutoff = ", cutoffUV)    
    print("# Nflux = ",elemflux+1,"/",nPhi1*nPhi2)
#    print("# fluxes_x = ", min(PhiLine),"..", max (PhiLine) )
    print("# fluxes_x = ", min(PhiLine [ : , 0 ]),"..", max (PhiLine[ : , 0 ]) )
    print("# fluxes_y = ", min(PhiLine [ : , 1 ]),"..", max (PhiLine[ : , 1 ]) )
#    if shufflefluxes :
#        print("# shuffled fluxes:")
#        print("# ",PhiLine)
    print("# time =", time.time() - start)
    print()          
    print("# T          Drude")
    for i in numpy.arange (len (betas) ) :
#        if divide[i] != 0 :
        if divide != 0 :
#            print(1./betas [i] , dru [i]/(divide[i]), dru1 [i]/(divide[i]), dru2 [i]/(divide[i]), zeta [i] /(divide[i]))  #count only nonzero drudes
#            print(1./betas [i] , dru [i]/(divide[i]), dru1 [i]/(divide[i]), dru2 [i]/(divide[i]), zeta [i] /(divide[i]) )  #count only nonzero drudes
            print(1./betas [i] , dru [i]/(divide), dru1 [i]/(divide), dru2 [i]/(divide), proxy [i]/(divide), zeta [i] /(divide) )  #count only nonzero drudes
    sys.stdout = orig_stdout
    g.close()
    

def DetailedBalance( ene ) :
#    a = numpy.ones (  (RetainedLevels , RetainedLevels) )
    ene += 1.e-140 #regularize
    a =  (1.0 - numpy.exp( - ene ) )/ene
    return a
    
def DetailedBalancesign( ene ) :
#    a = numpy.ones (  (RetainedLevels , RetainedLevels) )
    ene += 1.e-20 #regularize
    a =  ((1+numpy.sign(ene))/2)*(1.0 - numpy.exp( - ene ) )/ene
    return a
  
  
    
def DetailedBalancelow( ene ) :
    thr = 140.
    thr=600.
#    if  ene  <  cutoffdetbal :  #I only look at positive energies
#        a = 1.0 # 0.0 nearby levels: don't count or count as Beta
#    elif ( ene  <  thr  and  ene  >  cutoffdetbal ) :  #I only look at positive energies
#        a =  (1.0 - numpy.exp( - ene ) )/ene
#    else :
#        a=1/ene 
    a=0
    if numpy.abs( ene ) <  cutoffdetbal :
        a = 1.0 - ene / 2.0# 0.0 nearby levels: don't count or count as Beta
    elif ( numpy.abs( ene ) <=  thr  and numpy.abs( ene ) >=  cutoffdetbal ) :  
        a =  (1.0 - numpy.exp( - ene ) )/ene
    elif ene >thr  :
        a= 1/ene
    elif ene < -thr :
        a=   numpy.exp(thr)
#    return a * ene
    return a
def CreateCenteredThermalProbabilities( ene , ene0 , beta ) :
    l = numpy.copy( ene )
    # returns separately Z and prob 
    thr = 160.
    thr = 700.
    thr = 500.
    l = l - ene0  #absolute shift
    a = numpy.zeros (len (ene))
    for i in numpy.arange(len (ene)) :
        if numpy.abs( beta * l [i] ) <  thr :
            a [i] =  numpy.exp( - beta * l[i] )
        else :
#            a [i] =  numpy.exp( - numpy.sign( l[i] ) * thr) 
            a [i] =  numpy.exp( - numpy.sign( l[i] ) * thr) * ( 1 - ( l[i] > thr ) ) #is 0 if l[i]>thr
    return a , numpy.sum( a )
def CreateCenteredThermalProbabilitiesplain( ene , ene0 , beta ) :
    a = numpy.exp( - beta * ( ene - ene0 ) )
    return a , numpy.sum (a)
def ComputeLegendreMinimum( dens_target ):
    global steps_taken
    mu_prec = 1.e-14
    

    if Beta > 0 :#< 0.5 :
#        mu_min =  - PotV  - 5.0*Kina - 5.0/Beta #takes the last one
        mu_min = -5.0*Kina -5.0/Beta 
#       mu has a log singularity in 2D at the edge in the classical limit
#        mu_min = -5.0*Kina -min(numpy.abs(numpy.log(dens_target*(1-dens_target)) ) /Beta , 10 )
#        if dens_target*(1-dens_target) > 0 :
#            mu_min = -5.0*Kina -min(numpy.abs(numpy.log(dens_target*(1-dens_target)) ) /Beta , 20 )
#        else :
#            mu_min = -5.0*Kina 
#        mu_min = -5.0*Kina -min(numpy.abs(numpy.log(numpy.abs(dens_target*(1-dens_target))) ) /Beta , 20 )
        mu_min = -1.0*Kina
        mu_min = -5.0*Kina
        if Beta < 2.0 :
            mu_min = -1.0*Kina -min(numpy.abs(numpy.log(numpy.abs(dens_target*(1-dens_target))) ) /Beta , 20 )

#        mumin = -10 -5/Beta
    else :
        mu_min =  - PotV  - 5.0*Kina            
#    mu_max = 0
#    mu_max= 20
    mu_max = -mu_min   # activate if need to go above half filling, like in spinless
    mu_min += 8*(dens_target-0.5)  # if mu is centered around 0, restore centering if all is shifted
    mu_max += 8*(dens_target-0.5)

    mua = mu_min
    mub = mu_max 
    
    
    mut = (mua+mub)/2 
#    mut=mub
    steps_taken = -1



#    n = numpy.arange( npmax + 1 )
    n = numpy.arange( npmax  )
    
    while numpy.abs( mut - mua ) > mu_prec:
#        expvec , _ = CreateCenteredThermalProbabilities( -  mut  * n   , E0ref , Beta ) #should put the shifts here too? avoid overflows
        expvec , _ = CreateCenteredThermalProbabilities( -  mut  * n  + corrshift , E0ref , Beta ) #should put the shifts here too? avoid overflows
###        expvec = numpy.exp( mut * n * Beta - Beta * corrshift)
        #do not use relE0 for the GC probabilities when calculating the compressibility (cf. Gros, must use the absolute energies)


        
#        ztmp = numpy.sum(             Zeta_N[ : , eb ] * expvec[ : ] * numpy.exp( -Beta * allE0shifts) )
#        cost = numpy.sum( n[ : ] *    Zeta_N[ : , eb ] * expvec[ : ] * numpy.exp( -Beta * allE0shifts) ) / ztmp / ns  - dens_target


        ztmp_elem = numpy.zeros( PhiLine.shape[0] )
        ntmp_elem = numpy.zeros( PhiLine.shape[0] )

        tmpv = (expvec[ : ] * numpy.exp( -Beta * allE0shifts) )
        if Beta > Betacut : #take the maximum of E-mu*N, the actual Zeta has a log (but adding it won't work...)
            for iel in numpy.arange( len(PhiLine) ) : 
                ntmp_elem [ iel ] = numpy.argmin (allE0[ : , iel ] - mut * n ) 
#                ntmp_elem [ iel ] = numpy.argmin (allE0[ : , iel ] - numpy.log( allnb * allE0Krylovs [ : , iel ] ) / Beta  - mut * n ) 
            ntmp =   numpy.average( ntmp_elem ) / ns
##       low T explicit expression of ZN
#            ntmp = 0.0
#            for iel in numpy.arange( len(PhiLine) ) :            
#                ztmp_elem   = numpy.exp( Beta * ( mut * n - allE0[ : , iel ] ) ) @ ( allnb * allE0Krylovs [ : , iel ]  )     
#                ntmp_elem   = numpy.exp( Beta * ( mut * n - allE0[ : , iel ] ) ) @ ( n[ : ] * allnb * allE0Krylovs [ : , iel ]  )  /ns / ztmp_elem
#                ntmp +=  ntmp_elem / len(PhiLine)     
        else : 
#            ztmp_elem[ : ]   = Zeta_N_elem[ : , eb , : ].T @ tmpv[ : ]
#            ntmp_elem[ : ]   = Zeta_N_elem[ : , eb , : ].T @ ( n[ : ] * tmpv ) /ns / ztmp_elem
            ztmp_elem[ : ]   = Zeta_N_elem[ : , eb , :  ].T @ ( allnb * tmpv[ : ] )
            ntmp_elem[ : ]   = Zeta_N_elem[ : , eb , :  ].T @ ( allnb * n  * tmpv ) / ns / ztmp_elem
            ntmp =  numpy.average( ntmp_elem )

        cost = ntmp - dens_target





        if cost < 0 :   mua = mut
        else        :   mub = mut
        
        mut = ( mub + mua ) / 2
        steps_taken += 1

    return mua 
def LanczosSparse( v0 , A ) :
    a =numpy.zeros( RetainedLevels , dtype = 'float' ) # max size M
    b =numpy.zeros( RetainedLevels , dtype = 'float' ) # max size M

#    Q = numpy.zeros ( ( nb , RetainedLevels ) , dtype = 'complex' ) 
    Q = numpy.zeros ( ( nb , RetainedLevels ) , dtype = 'complex' )  #nb x RetainedLevels and Krilov_v is same
    
    x = v0
    q = x / numpy.linalg.norm( x )
    Q [   : , 0      ] =  q # max size ( nb , M )
    r = A @ q
    a[0] = numpy.vdot( q , r ).real
    r = r - a[ 0 ] * q
    b[0] = numpy.linalg.norm( r ) 

    
    eneold = 1
    enenew = 0
    for iM in numpy.arange( 1 , RetainedLevels   ) :
        eneold = enenew
        v = q
        q = r / b[ iM - 1  ]
        Q [ : , iM   ] = q
#        if GramSchmidtflag :
#        if GramSchmidtflag and iM % GramSchmidtinterval == 0 :
        if GramSchmidtflag and ( iM % GramSchmidtinterval == 0 or iM ==  RetainedLevels - 1 ):
            Q [ : , 0 : iM   ] = GramSchmidt( Q [ : , 0 : iM   ] )
#            q = Q[ : , Q.shape[ 1 ] - 1 ]
#            v = Q[ : , Q.shape[ 1 ] - 2 ]
            q = Q[ : , iM   ]
            v = Q[ : , iM - 1 ]

        r = A @ q - b[ iM - 1  ] * v
        a[iM] = numpy.vdot( q , r ).real
        r = r - a[  iM  ] * q
        b[iM] = numpy.linalg.norm( r )

        if VerboseLanczosSparse and  iM % 10 == 0 :
            krylov_e , krylov_v = eigh_tridiagonal( a.real , b[ 0 : b.shape[ 0 ] - 1 ].real , tol = 1.e-14 , check_finite=True , lapack_driver = 'stebz')
            enenew = krylov_e[ 0 ]
            print( a.shape[ 0 ] , enenew , numpy.abs( eneold  - enenew ) )

    krylov_e , krylov_v = eigh_tridiagonal( a , b [ 0 : b.shape[ 0 ] - 1 ], tol = 1.e-14 , check_finite = True, lapack_driver = 'stebz')


#    psi =  ncon( [ krylov_v , Q ] , [ [1,-1] , [-2,1]  ] ).T
    psi = Q @ krylov_v  # psi and Q are two big matrices of size (nb x RetainedLevels) -> at the return, only one (psi) is stored
    
    if VerboseLanczosSparse :
        print( "# < psi_0   |   psi_0 > = " , psi[ : , 0 ].conj() @ psi[ : , 0 ] )
        print( "# < psi_0 | H | psi_0 > = " , psi[ : , 0 ].conj() @ A @ psi[ : , 0 ] )
    return krylov_e[ 0 : RetainedLevels ] , psi[ : , 0 : RetainedLevels ] , krylov_v[ : , 0 : RetainedLevels ] 
def GramSchmidt(A):
    dim = A.shape[1]
    Q = numpy.zeros(A.shape, dtype=A.dtype)
    for j in numpy.arange( dim ):
        q = A[:,j]
        for i in numpy.arange( j ):
            rij = numpy.vdot( Q[ : , i ] , q )
            q = q - rij * Q[ : , i ]
        rjj = numpy.linalg.norm( q, ord = 2 )
        if numpy.isclose( rjj , 0.0  , atol=1e-16):
            raise ValueError("invalid input matrix")
        else:
            Q[ : , j ] = q / rjj
    return Q





def ComputeLanczosFiniteTConductivity( el   ) :

    global hist_N , justbin_edges , justhist # used for annealed average
    global Zeta_N_elem, Kinsym_N_elem,  Drude_N_elem  , Drude1_N_elem , Drude2_N_elem 


    global Zetacumul,Kincumul,Kinsymcumul,Zetaelem,Zetaelem_ForN
    global ContribFiniteTConductivity,ContribFiniteTConductivityperp, ContribDOS
    global E0first, E0absolute, E0avg , Kin0first, Kin0absolute , Kin0avg , Krylov0 , allE0, allE0Krylovs
    global allE0first, allE0absolute

#    print ("size Ju - Mb ",  Ju.nbytes/1024**2)


#    vec_r = 1.0 - 2.0*numpy.random.random( ham.shape[ 0 ] ) #+1.0j*numpy.random.random( ham.shape[ 0 ] ) 
#    vec_r = numpy.random.random( ham.shape[ 0 ] ) + 1.0j*numpy.random.random( ham.shape[ 0 ] ) 

    # vectors must be real in LanczosSparse? no, just the a and b are
    # but J|r> is not real, check if routine is ok with TBC
    vec_r = 1.0 - 2.0*numpy.random.random( nb ) + 1.0j*( 1.0 - 2.0*numpy.random.random( nb ) )
    Norm_r=numpy.linalg.norm( vec_r ) 
    vec_r = vec_r  /  numpy.linalg.norm( vec_r )

    ham= CreateSparseHamiltonian(  ) 

    

#    hamdense=ham.todense()
#    hamdense.tofile("matrix.txt", " ", "")

#    x= CreateSparseHamiltonian(  ) 
#    y= hamZ
#    d = numpy.concatenate((x.data, 0.1*y.data))
#    r = numpy.concatenate((x.row, y.row))
#    c = numpy.concatenate((x.col, y.col))
#    ham = scipy.sparse.coo_matrix((d,(r,c)))

    EigenEnergies , EigenVectors , KrylovVectors = LanczosSparse( vec_r  , ham )

#    EigenEnergies , EigenVectors , KrylovVectors = LanczosSparse( vec_r  , CreateSparseHamiltonian(  ) ) # -> 1st big matrix in memory (nb x RetainedLevels) + ham

    
    Ju = CreateSparseDrudeMatrixJu()
    vec_jr= Ju.T.conj() @ vec_r
#    print((numpy.exp(- beta * EigenEnergies / 2)).shape)
#    print((KrylovVectors[ 0 , : ].conj()).shape)
#    print((EigenVectors.T).shape)
#    print()

#    Prob2,Zeta2 = CreateCenteredThermalProbabilities(      EigenEnergies , E0ref , 0.5*beta0 ) 
#    Prob2 = numpy.exp(- beta * EigenEnergies/2 )
#    vec_jr = Ju @ ( (Prob2 * KrylovVectors[ 0 , : ].conj() @  EigenVectors.T  ).T   ) #why the conj? because

    Norm_jr=numpy.linalg.norm( vec_jr )  
    print("Norm_jr",Norm_jr)
#    vec_jr = vec_jr /  numpy.linalg.norm( vec_jr ) #DO NOT NORMALIZE JR (it's the same, it is normalized in the routine)


    Jijpart = EigenVectors.T.conj() @ Ju # 2nd big matrix in memory   
    del Ju




    Ki = - Kina * CreateSparseDrudeMatrixKu( )  
    Kij = EigenVectors.T.conj() @ Ki @ EigenVectors  

    
#   elements for non symmetric form of Kin        
    ContributionKnonsym = numpy.zeros( RetainedLevels )
    for i in numpy.arange( RetainedLevels ) :
        PartKinA =  KrylovVectors[ 0 , i ] 
        PartKinB =  EigenVectors[ : , i ].conj() @ Ki @ vec_r 
        tmpcontrib = PartKinA * PartKinB
        ContributionKnonsym[ i ] = tmpcontrib.real  #imag is nonzero !

#    print ("size Eig - Mb ",  EigenVectors.nbytes/1024**2)
    del EigenVectors # only 1 big matrix 
    del Ki




    TildeEigenEnergies , TildeEigenVectors , TildeKrylovVectors = LanczosSparse( vec_jr , ham  ) # -> 2nd big matrix
#    TildeEigenEnergies , TildeEigenVectors , TildeKrylovVectors = LanczosSparse( vec_jr , CreateSparseHamiltonian(  )  ) # -> 2nd big    

    Jij = Jijpart @ TildeEigenVectors #small matrix
    del Jijpart

    del TildeEigenVectors

    # storing instead 
#    print ("size TildeEig - Mb ",  TildeEigenVectors.nbytes/1024**2)
#    print ("size Jijpart - Mb ",  Jijpart.nbytes/1024**2)

    
    #-> 0 big matrices in memory


    K00 = Kij [ 0 , 0 ]
    K00 = K00.real
    if el == 0 :
        E0first = EigenEnergies[ 0 ]
        allE0first [ np ] = E0first
        Kin0first = K00

    if  EigenEnergies[ 0 ] < E0absolute :
        E0absolute = EigenEnergies[ 0 ]
        allE0absolute [ np ] = E0absolute
        Kin0absolute = K00

    E0avg += EigenEnergies[ 0 ]  #calculate average Energy in canonical ensemble à la Poilblanc (close to Gros)
    Kin0avg += K00

    allE0 [ np , el ] = EigenEnergies[ 0 ] 
    allE0Krylovs [ np , el ] = (KrylovVectors[ 0 , 0 ]**2).real




#    Jij = EigenVectors.T.conj() @ Ju @ TildeEigenVectors#    Jij = ncon( [ EigenVectors.conj() , (Ju.toarray()).T.conj() , TildeEigenVectors ] , [ [1,-1] , [1,2] , [2,-2]  ] )  #important if TBC



    #-------------------FINITE T CONDUCTIVITY --------------
    ContributionC = numpy.zeros( ( RetainedLevels , RetainedLevels, 2 ) ) 
    ContributionCperp = numpy.zeros( ( RetainedLevels , RetainedLevels, 2 ) ) 
    ContributionCii = numpy.zeros( ( RetainedLevels, 2 ) ) 
    ContributionKsym = numpy.zeros( ( RetainedLevels , RetainedLevels ) , dtype='complex') 
    ContributionProxy = numpy.zeros( ( RetainedLevels - 2 , 2 ) ) 


    indel = 0
    
#### USE VECTORIZED LOOP


    ContributionProxy [ : , 1 ]     =  numpy.abs(  KrylovVectors[ 0 , 1 : RetainedLevels -1  ] *  KrylovVectors[ 0 , 1 : RetainedLevels -1 ].conj() )
    ContributionProxy [ : , 0 ]     =  (EigenEnergies[ 2 : RetainedLevels   ] - EigenEnergies[ 0 : RetainedLevels -2 ] )/2


    freqs                            = TildeEigenEnergies - EigenEnergies[ : , numpy.newaxis ] # matrix of freqs i=numpy.newaxis, j=plain
    ContributionKsym                 = KrylovVectors[ 0 , : , numpy.newaxis] * Kij *  KrylovVectors[ 0 , numpy.newaxis, : ].conj()
    ContributionC [ : , : , 0 ]      = freqs
    ContributionC [ : , : , 1 ]      = Norm_jr*(  KrylovVectors[ 0 , : , numpy.newaxis] * Jij *  TildeKrylovVectors[ 0 , numpy.newaxis, : ].conj() ).real * numpy.pi / ns

 

    ContributionCTherm = numpy.copy( ContributionC )
    ContributionCThermperp = numpy.copy( ContributionCperp )
    ContribDOS [ el , : , 0 ] =  EigenEnergies   - E0ref # - PotV * ns * InvDist[0] / 2 # -  2 * (ns/2)**2 * InvDist[0]
    ContribDOS [ el , : , 1 ] =  1/RetainedLevels

    CreateBinningDOS( ContribDOS, 50 , 1 , "DOSFTLM" + "_np" + str(np) + Suffix )
    
    
#    + "_" + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + "_np" + str(np) + "_J" + str(PotJa) + "_V" + str(PotV0) + "_M" + str(RetainedLevels0) + "_" + BoundaryConditions + "_" + TypeSpin + ".dat" )


    Krylov0      = (KrylovVectors[ 0 , 0 ]**2).real

#    print("K0",Krylov0) 


    ContribFiniteTKinAverage   = numpy.zeros( ( RetainedLevels , RetainedLevels ) )
    ContribFiniteTConductivitylocal = numpy.zeros( ( RetainedLevels * RetainedLevels  , 2 ) ) 

    #-------- BETA DEPENDENT QUANTITIES --------------------
    for eb , Beta in enumerate( BetaLine ) : # only one temperature for LTL
        if not relE0 :
            Prob , Zeta           = CreateCenteredThermalProbabilities(      EigenEnergies , E0ref , Beta ) #use probab with absolute energy shift (not TBC-dependent shift, Imada-Prelovsek)
            Prob_ForN , Zeta_ForN = Prob , Zeta
            TildeProb , TildeZeta = CreateCenteredThermalProbabilities( TildeEigenEnergies , E0ref, Beta )
        else :
            Prob , Zeta           = CreateCenteredThermalProbabilities(      EigenEnergies , E0ref + (EigenEnergies[0] - E0first)  , Beta ) # for sigma use probab with TBC-dependent shift (~Poilblanc)
            Prob_ForN , Zeta_ForN = CreateCenteredThermalProbabilities(      EigenEnergies , E0ref  , Beta )                        # for N use probab without TBC-dependent shift (cf. Gros)
            TildeProb , TildeZeta = CreateCenteredThermalProbabilities( TildeEigenEnergies , E0ref + (EigenEnergies[0] - E0first)  , Beta )
        
        Zetanew = 0.0
        ContribKnonsymBuff = 0.0


#       partition function and nonsymmetric Kin at given beta        
        Zetanew      = numpy.vdot( KrylovVectors[ 0 , :  ] , KrylovVectors[ 0 , :  ] * Prob[ : ] ).real    #with shift à la Poilb
        Zetanew_ForN = numpy.vdot( KrylovVectors[ 0 , :  ] , KrylovVectors[ 0 , :  ] * Prob_ForN[ : ] ).real #without shift for Gros
        
        proxynew = numpy.vdot (  ContributionProxy [ : , 1 ] , ContributionProxy [ : , 0 ] * Prob [ 1 : RetainedLevels - 1] ).real



 
        ContribKnonsymBuff = ContributionKnonsym @ Prob

 
        
 

#       optical conductivity and symmetric Kin at given beta         

    #Kij = Kij *  KrylovVectors[ 0 ,: ].conj()
    #Kij = ( KrylovVectors[ 0 , : ] * Kij.T ).T

        tmpK = numpy.copy(ContributionKsym)
        tmpK = tmpK * numpy.sqrt( Prob )
        ContribFiniteTKinAverage = (( numpy.sqrt( Prob ) * tmpK.T ).T).real 
#        tmpK = tmpK * Prob2
#        ContribFiniteTKinAverage = (( Prob2  * tmpK.T ).T).real 



#        DetBal = numpy.zeros ( (RetainedLevels , RetainedLevels) )  # (1-exp(-beta*dE))/(beta*dE)
#        for i in  numpy.arange( RetainedLevels ) : 
#            for j in numpy.arange( RetainedLevels ) :
#                DetBal [ i  , j ]= DetailedBalancelow ( Beta* ( TildeEigenEnergies[  j ] - EigenEnergies[ i  ] ) )
        DetBal =  DetailedBalance ( Beta* ( TildeEigenEnergies - EigenEnergies[ : , numpy.newaxis ] ) ) 

#        Prob2bis,Zeta2bis = CreateCenteredThermalProbabilities(      EigenEnergies , E0ref , (1-frac)*beta0 ) 

        tmp = ContributionC[ : , : , 1 ] * Beta * DetBal 
#        ContributionCTherm[ : , : , 1 ] = ( Prob2  * tmp.T ).T       
        ContributionCTherm[ : , : , 1 ] = ( Prob  * tmp.T ).T   # for sigma local (quenched) this should have Prob_ForN (unshifted) and so should zetanew
                                                                # I can also use the shifted values for both sigma and Prob, as is done here, will not change the result as the factor simplifies in the ratio

#        if not Zglobal :
#            ContributionCTherm[ : , : , 1 ] /= Zetanew

        Sumruleofel = numpy.sum( ContributionCTherm[ : , : , 1 ] ) / numpy.pi  #/2 # divide if both sides! 
#        if eb == 0 :
#            print("sumrule(lowT)/ Zetanew  sumrule/exp(-betaE0)/Krylov0   sumrule/exp(-betaE0)   log(zetanew)  log(exp(-betaE0)*Krylov0)  Krylov0")
#            print(Sumruleofel / Zetanew , Sumruleofel * numpy.exp(BetaLine[0]*EigenEnergies[0]) / Krylov0 , Sumruleofel * numpy.exp(BetaLine[0]*EigenEnergies[0]) , numpy.log(Zetanew) , numpy.log(numpy.exp(-BetaLine[0]*EigenEnergies[0]) * Krylov0) , Krylov0)




            
        
#        TotalSumFiniteTKinAverage = ContribKnonsymBuff
        TotalSumFiniteTKinAverage = numpy.sum( ContribFiniteTKinAverage )  



        
        Dovertwopi1 =   -TotalSumFiniteTKinAverage   / Zetanew /  2 / ns
        Dovertwopi2 =   Sumruleofel/Zetanew 
        Dovertwopi  =   (-Dovertwopi1 + Dovertwopi2)
#        print(Dovertwopi1,Dovertwopi2,Dovertwopi)

#   calculate local thermodynamic static quantities
        
        Kinsymcumul [ eb ] += TotalSumFiniteTKinAverage #this will sum all Z 
        Kincumul [eb]    += ContribKnonsymBuff
#        Proxycumul[eb]   += proxynew / Zetanew * RetainedLevels / (EigenEnergies[ RetainedLevels -1 ]-EigenEnergies[0]) #local (quenched) proxy scaled
        Proxycumul[eb]   += proxynew / Zetanew # local (quenched) proxy
        Drudecumul [eb]  += Dovertwopi   
        Drudecumul1 [eb] += Dovertwopi1
        Drudecumul2 [eb] += Dovertwopi2
        Sumrulecumul [ eb ] +=  Sumruleofel

        Zetacumul [ eb ]         += Zetanew      #this will average the free energy instead of the Z
        elDru [eb] +=1
        Zetaelem_ForN [ eb , el ] = Zetanew_ForN #* nb  # this one is not shifted as in Poilblanc, use for Gros
        Zetaelem [ eb , el ]      = Zetanew      #* nb  # this one is  shifted as in Poilblanc
#        Zetacumul [ eb ] += numpy.log(Zetanew) #this will average the free energy instead of the Z



#   calculate global thermodynamic static quantities and their local reductions for output
        Zeta_N_elem[ np , eb , el ] = Zetanew 
        Kinsym_N_elem [ np , eb , el ] = TotalSumFiniteTKinAverage 
        Drude_N_elem  [ np , eb , el ] = Dovertwopi  
        Drude1_N_elem [ np , eb , el ] = Dovertwopi1 
        Drude2_N_elem [ np , eb , el ] = Dovertwopi2 
        Zetacumul [ eb ]   = numpy.sum ( Zeta_N_elem  [ np , eb , : ] ) 
#        Drudecumul [ eb ]  = numpy.sum ( Drude_N_elem [ np , eb , : ] ) 
#        Drudecumul1 [ eb ] = numpy.sum ( Drude1_N_elem [ np , eb , : ] )
#        Drudecumul2 [ eb ] = numpy.sum ( Drude2_N_elem [ np , eb , : ] )




     
#        namesuffix = "_" + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + "_np" + str(np) + "_J" + str(PotJa) + "_V" + str(PotV0) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + "_M" + str(RetainedLevels0)+ "_" + BoundaryConditions + "_" + TypeSpin + ".dat" 
        namesuffix = "_np" + str(np) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix 





#        ContribFiniteTConductivity[ eb , el ,  : RetainedLevels**2 , : ] = ContributionCTherm.reshape( RetainedLevels**2 , 2 )
        ContribFiniteTConductivitylocal = ContributionCTherm.reshape( RetainedLevels**2 , 2 )
        justhist , justbin_edges = JustCreateBinning( ContribFiniteTConductivitylocal , nBin  )            
        allhist_N        [ np , eb , el , : ] = justhist / Zetanew
        allhist_N_global [ np , eb , el , : ] = justhist 
        allhist_cumul_local  = numpy.sum ( allhist_N        [ np , eb , : , : ] , axis = 0) / ( el + 1 )
        allhist_cumul_global = numpy.sum ( allhist_N_global [ np , eb , : , : ] , axis = 0) / Zetacumul [ eb ]
        hist_N[ np , eb , : ] = allhist_cumul_global  #used for annealed average, obsolete,  can be directly summed there
        JustWrite ( allhist_cumul_local  , justbin_edges , "sigwlocFTLM" + namesuffix)  # quench
#        JustWrite ( allhist_cumul_global , justbin_edges , "sigwFTLM" + namesuffix)     # anneal

        if FlagSigmaPerp : 
            justhistperp , justbin_edgesperp = JustCreateBinning( ContribFiniteTConductivitylocalperp , nBin  )            
            allhistperp_N        [ np , eb , el , : ] = justhistperp / Zetanew
            allhistperp_N_global [ np , eb , el , : ] = justhistperp 
            allhistperp_cumul_local  = numpy.sum ( allhistperp_N        [ np , eb , : , : ] , axis = 0) / ( el + 1 )
            allhistperp_cumul_global = numpy.sum ( allhistperp_N_global [ np , eb , : , : ] , axis = 0) / Zetacumul [ eb ]
            histperp_N[ np , eb , : ] = allhistperp_cumul_global  #used for annealed average
            JustWrite ( allhistperp_cumul_local  , justbin_edges , "sigwperplocFTLM" + namesuffix)
            JustWrite ( allhistperp_cumul_global , justbin_edges , "sigwperpFTLM" + namesuffix)
        
#        allhist_N [ np , eb , el , : ] = justhist * nb # this is sigma_tilde (only corr function without its normalization)

#    print(Drudecumul)
#    print(Drudecumul1)
#    print(Drudecumul2)

    print("final Drude",Drudecumul1[0]/(el+1),Drudecumul2[0]/(el+1),Drudecumul[0]/(el+1))

    WriteResults (BetaLine, Drudecumul, Drudecumul1, Drudecumul2 , Proxycumul , Zetacumul  , ( el + 1 )  , "drudeFTLM"  + "_np" + str(np)  + Suffix)      
# }}}






def ReadCreateBasis() :
    global nb, Basis , Dict
    global BasisUp,DictUp,BasisDw,DictDw
    
    t1=time.time()

    direct = '../bases-tJ/'
    file = direct + 'basis_' + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + '_np' + str(np) + "_" + TypeSpin 
    filedict = direct + 'dict_' + str(t1x) + "_" + str(t1y)  + "_" + str(t2x) + "_" + str(t2y) + '_np' + str(np) + "_" + TypeSpin 
    ok  = exists( file + ".npz" )
    ok  *= exists( filedict + ".pkl" )


    if ok :
        bases = numpy.load( file + ".npz" )
        Basis = bases['bas']
        with open(filedict + ".pkl", 'rb') as fdict:
            Dict = pickle.load(fdict)

        nb = len(Basis)
        
        t2=time.time()
        print("import basis",t2-t1)    
        
    else :

        if TypeSpin == "spinless" :
            BasisUp , DictUp = CreateBasis(  np  )
            BasisDw , DictDw = CreateBasis(   0  )
        else :
            BasisUp , DictUp = CreateBasis(  np - np // 2  )
            BasisDw , DictDw = CreateBasis( np // 2 )    
        t2=time.time()
        print("create basis",t2-t1)    
    
    
        
        nb=len(BasisUp)*len(BasisDw)
        print( "# basis size (original) : ", nb)
    
        nww = 0
        for u in BasisUp :
            for d in BasisDw :
                if u & d == 0 :
    #                a = u + d << ns
                    nww += 1
    
    
        t1=time.time()
        Basis,Dict = DecimateBasis( nww )
        t2=time.time()
        print("decimate basis",t2-t1)    

        nb=nww

        numpy.savez_compressed( file , bas=Basis , dic=Dict )
        with open(filedict + ".pkl", 'wb') as fdict:
            pickle.dump(Dict, fdict)
        

    print( "# basis size (decimated) : ", nb)
    print( "# density : ", np ,  ns ) 






def DefineTables_N() :
	# tables needed for each np
#    global RetainedLevels
#    if ED :
#        RetainedLevels = nb            
#        print ("# ----------- ED --------------")
#    else :
#        RetainedLevels = min ( [ nb , RetainedLevels0 ])
#        if FTLM :
#            print ("# --------- FTLM -----------")
#        else :
#            print ("# --------- LTL -----------")
#    ContribDOS = numpy.zeros( (len( PhiLine ) , RetainedLevels , 2 ) )
              
    global elDru                 ;    elDru        = numpy.zeros ( len ( BetaLine ) )
    global Drudecumul1           ;    Drudecumul1  = numpy.zeros ( len ( BetaLine ) )
    global Drudecumul2           ;    Drudecumul2  = numpy.zeros ( len ( BetaLine ) )
    global Drudecumul            ;    Drudecumul   = numpy.zeros ( len ( BetaLine ) )
    global Zetacumul             ;    Zetacumul    = numpy.zeros ( len ( BetaLine ) )
    global Zetaelem_ForN         ;    Zetaelem_ForN = numpy.zeros ( (len ( BetaLine ) , len( PhiLine )) )
    global Zetaelem              ;    Zetaelem    = numpy.zeros ( (len ( BetaLine ) , len( PhiLine )) )
    global Kincumul              ;    Kincumul     = numpy.zeros ( len ( BetaLine ) )
    global Kinsymcumul           ;    Kinsymcumul     = numpy.zeros ( len ( BetaLine ) )
    global Corrnnsymcumul        ;    Corrnnsymcumul     = numpy.zeros ( len ( BetaLine ) )
    global Corrnnnsymcumul       ;    Corrnnnsymcumul     = numpy.zeros ( len ( BetaLine ) )
    global CorrnSiSjnncumul      ;    CorrnSiSjnncumul   = numpy.zeros ( len ( BetaLine ) )
    global CorrnSiSjnnncumul     ;    CorrnSiSjnnncumul   = numpy.zeros ( len ( BetaLine ) )
    global Corrspinbagsymcumul   ;    Corrspinbagsymcumul     = numpy.zeros ( len ( BetaLine ) )
    global Corrlongdiffsymcumul  ;    Corrlongdiffsymcumul     = numpy.zeros ( len ( BetaLine ) )
    global Corrlongsamesymcumul  ;    Corrlongsamesymcumul     = numpy.zeros ( len ( BetaLine ) )
    global SSsymcumul            ;    SSsymcumul               = numpy.zeros ( len ( BetaLine ) )
    global Sumrulecumul          ;    Sumrulecumul = numpy.zeros ( len ( BetaLine ) ) 
    global Proxycumul              ;    Proxycumul = numpy.zeros ( len ( BetaLine ) ) 

#    global ContribDOS            ;    ContribDOS = numpy.zeros( (len( PhiLine ) , RetainedLevels , 2 ) ) #+1 


def DefineTables_GC() :   
     
    # global tables for GC loop over np 
    global allnb                 ;    allnb            = numpy.ones ( npmax )
    global Zeta_N                ;    Zeta_N           = numpy.zeros ( ( npmax , len ( BetaLine ) ) )
    global Kinsym_N              ;    Kinsym_N         = numpy.zeros ( ( npmax , len ( BetaLine ) ) )
    global Corrnnsym_N           ;    Corrnnsym_N      = numpy.zeros ( ( npmax , len ( BetaLine ) ) )
    global Corrnnnsym_N          ;    Corrnnnsym_N     = numpy.zeros ( ( npmax , len ( BetaLine ) ) )
    global Corrspinbagsym_N      ;    Corrspinbagsym_N = numpy.zeros ( ( npmax , len ( BetaLine ) ) )
    global hist_N                ;    hist_N           = numpy.zeros ( ( npmax , len(BetaLine) , nBin ) , dtype='float')
    global allhist_N             ;    allhist_N        = numpy.zeros ( ( npmax , len(BetaLine) , len( PhiLine ) , nBin ) , dtype='float')
    global allhist_N_global      ;    allhist_N_global = numpy.zeros ( ( npmax , len(BetaLine) , len( PhiLine ) , nBin ) , dtype='float')
    global histperp_N            ;    histperp_N       = numpy.zeros ( ( npmax , len(BetaLine) , nBin ) , dtype='float')
    global allhistperp_N         ;    allhistperp_N    = numpy.zeros ( ( npmax , len(BetaLine) , len( PhiLine ) , nBin ) , dtype='float')
    global allhistperp_N_global  ;    allhistperp_N_global = numpy.zeros( ( npmax , len(BetaLine) , len( PhiLine ) , nBin ) , dtype='float')
    global allE0shifts           ;    allE0shifts      = numpy.zeros ( npmax   , dtype='float')
    global allE0                 ;    allE0            = numpy.zeros ( ( npmax ,  len( PhiLine ) )   , dtype='float')
    global allE0Krylovs          ;    allE0Krylovs     = numpy.ones  ( ( npmax ,  len( PhiLine ) ) , dtype='float')
    global allE0absolute         ;    allE0absolute    = numpy.zeros ( npmax , dtype='float')
    global allE0first            ;    allE0first       = numpy.zeros ( npmax , dtype='float')
   
    global Zeta_N_elem           ;    Zeta_N_elem      = numpy.zeros ( ( npmax , len ( BetaLine ) , len( PhiLine ) ) )
    global Kinsym_N_elem         ;    Kinsym_N_elem    = numpy.zeros ( ( npmax , len ( BetaLine ) , len( PhiLine ) ) )
    global Drude_N_elem          ;    Drude_N_elem     = numpy.zeros ( ( npmax , len ( BetaLine ) , len( PhiLine ) ) )
    global Drude1_N_elem         ;    Drude1_N_elem    = numpy.zeros ( ( npmax , len ( BetaLine ) , len( PhiLine ) ) )
    global Drude2_N_elem         ;    Drude2_N_elem    = numpy.zeros ( ( npmax , len ( BetaLine ) , len( PhiLine ) ) )

    

##########################################################################################
#
#                                      main preamble
#
##########################################################################################





start = time.time()
CreateFluxes( )
CreateLattice( )

CreateBond()
end = time.time()
print("# Time elapsed in CreateLattice: ", end-start )


start = time.time()

# definitions for GC loop over np 
npmin= 1
npmax = ns + 1  

if npp == 0 : 
    nploopmin = npmin
    nploopmax = npmax -1
#    nploopmin = 2*ns//3  # fill up to 30% doping 
    
else :
    nploopmin = npp ## -1
    nploopmax = npp 


n = numpy.arange( npmax ) # table of all np once and for all

DefineTables_GC()  
  
  
# shifts to avoid overflows
#the average energy scales like -3.2 per hole, i.e. -3.2*(ns-np)
E0ref = 0 #3.2 # bottom of single hole band from RetraceablePathApproximation Brinkman-Rice (square lattice)
mushift =  3.2  # this avoids overflows in GC sums over N

if FlagOneHole :
    corrshift = mushift * ( ns + 1 ) #this focuses on half filling
else :
    corrshift = mushift * ( ns - 1 ) # this focuses somewhere around two holes (compromise)
    corrshift = mushift * ( 3 * ns // 4 ) # this focuses somewhere around two holes (compromise)





# partition function for 0 and ns particles
if TypeSpin == "spinless" :
    densshift = 0.5
    densshift = 0.5
    Zeta_N[ 0 , : ]   =  [ CreateCenteredThermalProbabilities( numpy.array([ 0 ]) , E0ref , beta ) [ 1 ]  for beta in BetaLine ] 
    Zeta_N[ ns  , : ] =  [ CreateCenteredThermalProbabilities( numpy.array([ 0 ]) , E0ref , beta ) [ 1 ]  for beta in BetaLine ] 
    for iel in numpy.arange( len(PhiLine) ) :            
        Zeta_N_elem[ 0 , : , iel ] =  Zeta_N [ 0 , : ]
        Zeta_N_elem[ ns , : , iel ] =  Zeta_N [ ns , : ]
else :
    #shifted Hint
    densshift = 1.0
    densshift = 0.0
    Zeta_N[ 0 , : ]      =  [ CreateCenteredThermalProbabilities( numpy.array([ 0 ]) , E0ref , beta ) [ 1 ]  for beta in BetaLine ] 
    for iel in numpy.arange( len(PhiLine) ) :            
        Zeta_N_elem[ 0 , : , iel ] =  Zeta_N [ 0 , : ]
#        Zeta_N_elem_relE0[ 0 , : , iel ] =  Zeta_N [ 0 , : ]
#check the V part coming from Vii









#density table

if TypeSpin == "spinless" :
    densLine = numpy.array( [ 0.0 , 0.05 , 0.1 , 0.2 , 0.3 , 0.35 , 0.4 , 0.45 , 0.5 ] , dtype = 'float' )
    densLine = numpy.array( [ 0.05 ,  0.1 , 0.15 , 0.2 , 0.25 , 0.3 ,  0.4 , 0.45 , 0.5 , 0.55 , 0.6 , 0.65 , 0.7 , 0.75 , 0.8 , 0.85 , 0.9 , 0.95 ] , dtype = 'float' )
    densLineChi = densLine
else :
    densLineChi = numpy.arange ( 0. , 1.01, 0.01) 
    densLine = numpy.arange ( 0.5 , 1.05, 0.05) 
    densLine = numpy.append ( densLine , numpy.array( [ 0.99 , 0.999 ]) )
    densLine = numpy.append( densLine , numpy.array ( [ ( ns - 1.0 ) / ns ] ) )

    densLineChi = numpy.array( [ densmin ]) 
    densLineChi = numpy.append ( densLineChi , numpy.arange ( 0.01 , 1.0, 0.01) )
    densLineChi = numpy.append ( densLineChi , numpy.array( [ 1.0 - densmin ]) )
    densLine = numpy.array( [ 0.5 , 0.6, 0.7, 0.75 , 0.8 , 0.85 , 0.9, 0.95 , 0.99 , 0.999 ])
    densLine = numpy.append( densLine , numpy.array ( [ ( ns - 1.0 ) / ns ] ) )


    if FlagOneHole :    
        densLineChi = numpy.arange ( 0.9 , 1.001, 0.001) 
        densLine = numpy.array( [ 0.9, 0.95 , 0.99 , 0.999 ])
        densLine = numpy.append( densLine , numpy.array ( [ ( ns - 1.0 ) / ns ] ) )



#initialize of Energy file 
orig_stdout = sys.stdout
g = open ( "energies-FTLM" + Suffix , 'w' )
sys.stdout = g
print( "### n   Ja   Jb    V    E0avg   E0min   E0minprime   E0PBC   K0avg   K0min   K0PBC" ) 
sys.stdout = orig_stdout
g.close()


        
        
#initialize mu file 
orig_stdout = sys.stdout
g = open ( "mu-FTLM"  + Suffix , 'w' )
sys.stdout = g
#print( "### Temp               n        deltan       mu      1/chi_c   Zgrand    K   steps" ) 
print( "# Temp  n        n(check)       mu      1/chi_c   Omegagrand  Zgrand    K   steps" ) 
sys.stdout = orig_stdout
g.close()        



        

##########################################################################################
#
#                                      main loop over np 
#
##########################################################################################

# swap np loop and el loop

# check E0first, E0absolute and define per elemflux

DefineTables_N()  # they are not reset at each flux




np=ns-1
ReadCreateBasis()
#hamZ=CreateSparseZeemanGlobalSz()

for elemflux , myphi in enumerate( PhiLine ) :
    print (elemflux + 1 , "/" , len(PhiLine))
    Phi1 = myphi[ 0 ]
    Phi2 = myphi[ 1 ]


#    for np in numpy.arange( npmax-1 ,  npmin-1, -1 ) :
#    for np in numpy.arange( nploopmax-1 ,  nploopmin-1, -1 ) :
    for np in numpy.arange( nploopmax ,  nploopmin - 1 , -1 ) :
        print("np",np,"ns",ns)
    
    
        ReadCreateBasis()
#        DefineTables_N()  # they are reset at each flux!

        if ED :
            RetainedLevels = nb            
            print ("# ----------- ED --------------")
        else :
            RetainedLevels = min ( [ nb , RetainedLevels0 ])
            if FTLM :
                print ("# --------- FTLM -----------")
            else :
                print ("# --------- LTL -----------")
        ContribDOS = numpy.zeros( (len( PhiLine ) , RetainedLevels , 2 ) )
        
        t1=time.time()
        CreateDecimatedHamiltonianMatrices( )
        if FlagStaticCorrelations and FlagOnlyEnergies : 
            CreateDecimatedStaticCorrelationMatrices()
        t2=time.time()
        print("create H and correlation matrices",t2-t1)
    
    
        # ------------------------------------------------------------------------------------------------------
        E0absolute=1.e10
        E0absoluteprime=1.e10
        Kin0absolute=1.e10
        E0avg=0.0 
        Kin0avg=0.0 
    
        
        if ED :
            ComputeEDFiniteTConductivity( elemflux  )   
        else :
            if FTLM :
#                if not FlagOnlyEnergies and np != ns : ComputeLanczosFiniteTSzCorr( elemflux  )
#                if not FlagOnlyEnergies and np != ns : ComputeLanczosFiniteTDensCorr( elemflux  )
                if not FlagOnlyEnergies and np != ns : ComputeLanczosFiniteTConductivity( elemflux  )
                if     FlagOnlyEnergies or np == ns: ComputeLanczosEnergies (elemflux) 
            else :
                ComputeLanczosLowTConductivity( elemflux  )




        allnb [ np ] = nb
        if FlagE0shift :  allE0shifts [ np ] =   ( allE0first [ np ] - allE0absolute [ np ] )    # recheck sign, should shift wrt average E0, see below
#    if FlagE0shift :  allE0shifts [ np ] =   - (E0first - E0avg / len(PhiLine))     # check sign
    
    # write energies after flux loop at each np
        orig_stdout = sys.stdout
        g = open ( "energies-FTLM" + Suffix , 'a' )
        sys.stdout = g
        print( np ,  PotJa , PotJb , PotV , E0avg /( elemflux + 1 ) , allE0absolute[ np ] , E0absoluteprime , allE0first [ np ] , Kin0avg / ( elemflux + 1 ) , Kin0absolute , Kin0first)
        sys.stdout = orig_stdout
        g.close()
    
    
    #   store thermal averages for a particular flux; Zetacumul and similar are obsolete, they can be resummed directly here; to be cleaned up
        
        for eb , Beta in enumerate( BetaLine ) :
            if Zglobal :
                if FTLM :
###                    Zeta_N_elem[ np , eb , : ] = Zetaelem[ eb , : ] * nb  #/ len(PhiLine) #already defined
#                    Zeta_N[ np , eb ] = Zetacumul[ eb ] * nb  / len(PhiLine)
#                    Zeta_N[ np , eb ] = numpy.sum(Zeta_N_elem[ np , eb , : ])  / len(PhiLine)
                    Zeta_N[ np , eb ] = numpy.sum(Zeta_N_elem[ np , eb , : ])  * nb / ( elemflux + 1 )
#                    Zeta_N[ np , eb ] = numpy.exp(Zetacumul[ eb ]/ len(PhiLine)) * nb  

                    Kinsym_N[ np , eb ] = Kinsymcumul[ eb ]  / Zetacumul[ eb ]            
                    Corrnnsym_N[ np , eb ] = Corrnnsymcumul[ eb ]  / Zetacumul[ eb ]            
                    Corrnnnsym_N[ np , eb ] = Corrnnnsymcumul[ eb ]  / Zetacumul[ eb ]            
                    Corrspinbagsym_N[ np , eb ] = Corrspinbagsymcumul[ eb ]  / Zetacumul[ eb ]            
                else :
                    print("# T=", 1./Beta, "Zbar=", Zetacumul [ eb ]   / ( elemflux + 1 )  ,"Drude" , Drudecumul [eb]  / Zetacumul [eb]) 
                    Zeta_N[ np , eb ] = Zetacumul[ eb ] / ( elemflux + 1 )
            else :
                if FTLM :
                    print("# T=", 1./Beta, "Zbar=", Zetacumul [ eb ] * nb  / ( elemflux + 1 ) ,"Drude" , Drudecumul [eb]  /  elDru [eb] )#len(PhiLine)  ) # Lanczos  own Z  
                    Zeta_N[ np , eb ] = Zetacumul[ eb ] * nb  / ( elemflux + 1 )
                    Kinsym_N[ np , eb ] = Kinsymcumul[ eb ]  / Zetacumul[ eb ]               
                    Corrnnsym_N[ np , eb ] = Corrnnsymcumul[ eb ]  / Zetacumul[ eb ]            
                    Corrnnnsym_N[ np , eb ] = Corrnnnsymcumul[ eb ]  / Zetacumul[ eb ]            
                    Corrspinbagsym_N[ np , eb ] = Corrspinbagsymcumul[ eb ]  / Zetacumul[ eb ]            
                else :
                    print("# T=", 1./Beta, "Zbar=", Zetacumul [ eb ]   / ( elemflux + 1 )  ,"Drude" , Drudecumul [eb]  / elDru [eb] ) # Lanczos low T own Z
                    Zeta_N[ np , eb ] = Zetacumul[ eb ] / ( elemflux + 1 )
    
end = time.time()
print("# Time elapsed in loop over np: ", end-start )












##########################################################################################
#
#                thermal loop - Grand Canonical -> do it for all elemflux
#
##########################################################################################

start = time.time()
print("# Entering GC loop")




if not FlagOnlyEnergies and not FlagOnlyHF and npp == 0 :
    print(densLine)
    for eb , Beta in enumerate( BetaLine ) :
        print ("sigma ", (eb+1),'/',len(BetaLine)) 

        for dens in densLine :

            muofn = ComputeLegendreMinimum( dens )
            expvec , _ = CreateCenteredThermalProbabilities( -  muofn  * n + corrshift  , E0ref , Beta )  #this is exp (beta* mu * n)
###            expvec = numpy.exp( muofn * n * Beta - Beta * corrshift)

            sig_avgLoc = numpy.zeros ( nBin )
            sig_avgLoc_quenched = numpy.zeros ( nBin )
            sig_avgLoc_quenchedperp = numpy.zeros ( nBin )

#  quenched 

            pippo = numpy.zeros( nBin ) 
            pippoperp = numpy.zeros( nBin ) 
            if Beta > Betacut : #take the maximum of E-mu*N plus log correction
                for iel in numpy.arange( ( elemflux + 1 ) ) : 
                    n_avgLoc_elem = numpy.argmin (allE0[ : , iel ] - numpy.log( allnb * allE0Krylovs [ : , iel ] ) / Beta  - muofn * n ) 
                    pippo +=  allhist_N[ n_avgLoc_elem , eb , iel , : ]  #sum all spectra that give the proper N(mu) at a given flux 
                    pippoperp +=  allhistperp_N[ n_avgLoc_elem , eb , iel , : ]  #sum all spectra that give the proper N(mu) at a given flux 
                    
            else : 
                for iel in numpy.arange( ( elemflux + 1 ) ) : 
                    tmpv = expvec[ : ] * numpy.exp( -Beta * allE0shifts)
                    Zgrand_elem     = Zeta_N_elem[ : , eb , iel ] @ tmpv
                    pippo += ( allhist_N[ : , eb , iel , : ].T  @  (  Zeta_N_elem[ : , eb , iel ] * tmpv ) ) / Zgrand_elem #already divided by ns
    #                pippo += ( allhist_N[ : , eb , iel , : ].T  @  tmpv  ) / Zgrand_elem #this if I don't normalize sigma
                    pippoperp += ( allhistperp_N[ : , eb , iel , : ].T  @  (  Zeta_N_elem[ : , eb , iel ] * tmpv ) ) / Zgrand_elem #already divided by ns

            sig_avgLoc_quenched [ : ] = pippo / ( elemflux + 1 )
            sig_avgLoc_quenchedperp [ : ] = pippoperp / ( elemflux + 1 )

            if ED == False :
                f = open( "sigwGCq-FTLM" + "_dens" + str('{0:.3g}'.format(dens)) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix , 'w' )
            else :
                f = open( "sigwGCq-ED" + "_dens" + str('{0:.3g}'.format(dens)) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix , 'w' )
            
            sys.stdout = f
            for ele in numpy.arange( len( justhist ) ) :
                print( justbin_edges[ ele ] , sig_avgLoc_quenched[ ele ] ) #prints left of interval
            sys.stdout = orig_stdout
            f.close()

            f = open( "sigwperpGCq-FTLM" + "_dens" + str('{0:.3g}'.format(dens)) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix , 'w' )
            sys.stdout = f
            for ele in numpy.arange( len( justhist ) ) :
                print( justbin_edges[ ele ] , sig_avgLoc_quenchedperp[ ele ] ) #prints left of interval
            sys.stdout = orig_stdout
            f.close()

#  quenched but pre-averaged 


            pippo = numpy.zeros( nBin ) 
            for iel in numpy.arange( ( elemflux + 1 ) ) : 
                tmpv = expvec[ : ] * numpy.exp( -Beta * allE0shifts)
                Zgrand_elem     = Zeta_N_elem[ : , eb , iel ] @ tmpv
                pippo += ( hist_N[ : , eb  , : ].T  @  (  Zeta_N_elem[ : , eb , iel ] * tmpv ) ) / Zgrand_elem #already divided by ns
#                pippo += ( allhist_N[ : , eb , iel , : ].T  @  tmpv  ) / Zgrand_elem #this if I don't normalize sigma
            sig_avgLoc_quenched [ : ] = pippo / ( elemflux + 1 )

            if ED == False :
                f = open( "sigwGC-FTLM" + "_dens" + str('{0:.3g}'.format(dens)) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix , 'w' )
            else :
                f = open( "sigwGC-ED" + "_dens" + str('{0:.3g}'.format(dens)) + "_T" + str( numpy.round( 1./Beta , decimals = 4) ) + Suffix , 'w' )
            
            sys.stdout = f
            for ele in numpy.arange( len( justhist ) ) :
                print( justbin_edges[ ele ] , sig_avgLoc_quenched[ ele ] ) #prints left of interval
            sys.stdout = orig_stdout
            f.close()

if not FlagOnlyHF and npp == 0 : 
    for eb , Beta in enumerate( BetaLine ) :
        print ("chi ", (eb+1),'/',len(BetaLine)) 
        for dens in densLineChi :

            muofn = ComputeLegendreMinimum( dens  )
            muofn0 = ComputeLegendreMinimum( dens - deltadens )
            invchiofn = (muofn - muofn0) / deltadens 


            


#            muofn = ComputeLegendreMinimum( dens )            
#            expvec , _ = CreateCenteredThermalProbabilities( -  muofn  * n  , E0ref , Beta )  #this is exp (beta* mu * n)
            expvec , _ = CreateCenteredThermalProbabilities( -  muofn  * n + corrshift  , E0ref , Beta )  #this is exp (beta* mu * n)
###            expvec = numpy.exp( muofn * n * Beta - Beta * corrshift)



# N quenched, average of ratios
# check that n = n_avgLoc
            n_avgLoc = 0.0
            n2_avgLoc = 0.0
            ZgrandLoc = 0.0
            Omegagrand = 0.0
            chi = 0.0
            if Beta > Betacut : #take the maximum of E-mu*N plus log correction
                for iel in numpy.arange( ( elemflux + 1 ) ) : 
    #                ntmp_elem [ iel ] = numpy.argmin (allE0[ : , iel ] - muofn * n ) 
                    n_avgLoc_elem = numpy.argmin (allE0[ : , iel ] - numpy.log( allnb * allE0Krylovs [ : , iel ] ) / Beta  - muofn * n ) 
                    n2_avgLoc_elem = n_avgLoc_elem**2
                    logZgrand_elem = - Beta * numpy.min (allE0[ : , iel ] - numpy.log( allnb * allE0Krylovs [ : , iel ] ) / Beta  - muofn * n )
                    Zgrand_elem   = numpy.exp( logZgrand_elem ) 
                    n_avgLoc +=  n_avgLoc_elem  / ( elemflux + 1 )
                    n2_avgLoc +=  n2_avgLoc_elem  / ( elemflux + 1 )
                    Omegagrand += - logZgrand_elem/Beta  / ( elemflux + 1 )
                    ZgrandLoc +=    Zgrand_elem  / ( elemflux + 1 )        
                    chielem =      Beta*(n2_avgLoc_elem - n_avgLoc_elem**2) # identically 0
                    chi    += chielem / ns / ( elemflux + 1 )   
            else :    
                for iel in numpy.arange( ( elemflux + 1 ) ) :            
                    Zgrand_elem     = Zeta_N_elem[ : , eb , iel ] @ ( expvec[ : ] * numpy.exp( -Beta * allE0shifts) )
                    logZgrand_elem     = numpy.log( Zeta_N_elem[ : , eb , iel ] @ ( expvec[ : ] * numpy.exp( -Beta * allE0shifts) ) )
                    n_avgLoc_elem   = n[ : ] @     (Zeta_N_elem[ : , eb , iel ] * expvec[ : ] * numpy.exp( -Beta * allE0shifts) ) /Zgrand_elem
                    n2_avgLoc_elem   = n[ : ]**2 @     (Zeta_N_elem[ : , eb , iel ] * expvec[ : ] * numpy.exp( -Beta * allE0shifts) )  /Zgrand_elem
                    n_avgLoc +=  n_avgLoc_elem  / ( elemflux + 1 )
                    n2_avgLoc +=  n2_avgLoc_elem  / ( elemflux + 1 )
                    Omegagrand += - logZgrand_elem / Beta / ( elemflux + 1 )
                    ZgrandLoc +=    Zgrand_elem  / ( elemflux + 1 )  
                    chielem =      Beta*(n2_avgLoc_elem - n_avgLoc_elem**2)
                    chi    += chielem / ( elemflux + 1 )/ns   
#                    chi    += 1/(chielem / ns) / len(PhiLine)    
#                chi = 1/chi
                    
            deltan2_avgLoc = (n2_avgLoc - n_avgLoc**2)
    
            orig_stdout = sys.stdout
            if ED == False :
                g = open ( "mu-FTLM" + Suffix , 'a' )
            else :
                g = open ( "mu-ED" + Suffix , 'a' )
            
            sys.stdout = g
#            print( 1./Beta , dens , "\t" , numpy.sqrt( n2_avgLoc - n_avgLoc**2 ) , muofn ,  invchiofn , ZgrandLoc , Kofmu , steps_taken )
#            print( 1./Beta , dens , "\t" ,deltan_avgLoc , muofn ,  invchiofn , Omegagrand , ZgrandLoc * numpy.exp( Beta* corrshift) , 0 , steps_taken )
#            print( 1./Beta , dens , "\t" ,numpy.sqrt(deltan2_avgLoc)/ns , muofn ,  1/invchiofn , Beta*deltan2_avgLoc/ns , Omegagrand , ZgrandLoc  , steps_taken )
            print( 1./Beta , dens , "\t" ,numpy.sqrt(deltan2_avgLoc)/ns , muofn ,  1/invchiofn , chi , Omegagrand , ZgrandLoc  , steps_taken )
            sys.stdout = orig_stdout
            g.close()

        if ED == False :
            g = open ( "mu-FTLM" + Suffix , 'a' )
        else :
            g = open ( "mu-ED" + Suffix , 'a' )
        
        sys.stdout = g
        print(  )
        print(  )
        sys.stdout = orig_stdout
        g.close()




    

    
    
    end = time.time()
    
    print("# Time elapsed in post-prod : ", end-start )
