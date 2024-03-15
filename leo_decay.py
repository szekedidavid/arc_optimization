### ======================================= ###
### ======= REQUIREMENTS CALCULATOR ======= ### 
### ======================================= ###
"""

Mission Need    : Low-Earth Orbit Observation Satellite
Propulsion Need : Station Keeping
Requirements    : Weight, Power, dV, Thrust, ISP

============================= ELABORATION ON IDENTIFIED REQUIREMENTS =============================

1. THRUST   : Based on the worst case scenario drag that can be experienced (maximum solar flux parameter) 
2. ISP      : Follows from the total dV requirement, assuming average solar flux (10+yrs) and note that the dV
              is provided as a range due to the assumption that the sattelites mass does not change
3. POWER    : The power requirement follows from estimations provided by ESA in "Engineering Guidelines for CubeSat Propulsion"

==================================================================================================

DaVinci Satellite:
    Size : 2U
    Mass : 2 kg 
    Alt (desired) : 400-500 [km]
    Alt (likely) : ESA Flyer Satellite [km]

Alternatives:
    DELFI-C3
    
Notes:
    Size requirement -> CubeSat too Small?
"""
### ======================================= ###

import numpy as np

# CONSTANTS
MU = 3.986004418e14
G0 = 9.80665

# USER INPUTS
A = 0.1*0.1                     # Spacecraft surface area [m^2]
cd = 2                          # Spacecraft coefficient of drag [-]
spacecraftmass = 5              # Spacraft mass [kg]
propmass = 0.10                 # Propellant mass [kg]
orbitalt = 500e3                # Orbital altitude [m] (180km-500km)
lifetime = 3                    # Total lifetime [yrs]
impulse_bit = 100e-6             # impulse per shot [uNs]

# space weather
f10 = 150                        # average solar flux parameter for 10+yrs
f10max = 175                    # maximum expected solar flux parameter
ap = 0                          # geomagnetic index

# margins
dragmargin = 0                 # [%]

def est_density(height, f10, ap):
    """
    Estimates the atmospheric density between 180km and 500km
    """
    
    virtualtemp = 900+2.5*(f10-70)+1.5*ap
    virtualmolarmass = 27-0.012*(height/1000-200)
    H = virtualtemp/virtualmolarmass
    
    density = 6*10**(-10)*np.exp(-(height/1000-175)/H)
    
    return density

# compute densities
density = est_density(orbitalt, f10, ap)
densitymax = est_density(orbitalt, f10max, ap)

# compute orbital velocity
v = np.sqrt(MU/(orbitalt+6371e3))

# compute drag
dragmax = 1/2*densitymax*v**2*A*cd*(1+dragmargin/100)
dragavg = 1/2*density*v**2*A*cd*(1+dragmargin/100)

# compute required total impulse
impulse = lifetime*365*24*3600*dragavg

# compute total number of shots
shots = impulse/impulse_bit

# compute the required frequency
frequency = shots/(lifetime*365*24*3600)

# compute time between shots
period = 1/frequency

# compute available power (ESA estimate 1-2W per kg)
powermax = 2*spacecraftmass
powermin = 1*spacecraftmass

# output
print("------------------------")
print("----| REQUIREMENTS |----")
print("------------------------")
print("Peak Thrust: " + str(np.round(dragmax*10**6,3)) + "[uN]")
print("Nominal Thrust: " + str(np.round(dragavg*10**6,3)) + "[uN]")
print("Total Impulse: " + str(np.round(impulse,1)) + " [Ns]")
print("Power: " + str(powermin) + "-" + str(powermax) + " [W]")
print("-----------------------")
print("")
print("------------------------")
print("-----| DESIGN IG |------")
print("------------------------")
print("Total Shots: " + str(np.round(shots)) + " [-]")
print("Frequency: " + str(np.round(frequency,5)) + " [Hz]")
print("Period: " + str(np.round(period,2)) + " [s]")
print("----------------------")