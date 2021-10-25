#!/usr/bin/env python3
"""
Model for the thermal evolution of the Earth

Core and mantle treated together
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import ode

# To create pdf figures vith vector fonts
mpl.rc('font', family='sans-serif', size=7, **{'sans-serif': ['Arial']})
mpl.rcParams['pdf.fonttype'] = 42

# physical parameters
year = 365 * 24 * 3600 # year in seconds
ME = 5.972e24 # mass of the Earth in kg
Cap = 1200 # effective heat capacity (potentially taking account of IC crystallisation etc.)
DT0 = 1330 # present potential temperature, in C
Qs0 = 46e12 # present surface heat flow in W.
Ta = 4e4 # activation temperature for viscosity, in C
amax = 4.5e9 # age of the Earth, in year

# radioactivity parameters. All times in year
tU235 = 7.04e8
lU235 = np.log(2) / tU235
CU235 = 5.69e-4
  
tU238 = 4.47e9
lU238 = np.log(2) / tU238
CU238 = 9.37e-5

tTh = 1.4e10
lTh = np.log(2) / tTh
CTh = 2.69e-5
  
tK = 1.25e9
lK = np.log(2) / tK
CK = 2.92e-5

def radio(concU, age):
    """Radiogenic heating as function of age
    
    concU: ratio of U concentration to CI model. 1 gives 21 ppb of U.
    age: in yr
    """
    hU235 = concU * 1e-9 * 0.0071 * CU235 * np.exp(lU235 * age)
    hU238 = concU * 1e-9 * 0.9928 * CU238 * np.exp(lU238 * age)
    hTh = concU / 21 * 75e-9 * CTh * np.exp(lTh * age)
    hK = concU / 21 * 1.19e-4 * 2.7e-4 * CK * np.exp(lK * age)
    htot = hU235 + hU238  + hTh + hK
    return htot, hU235, hU238, hTh, hK 

def Qs(Tpot, beta):
    """Surface heat flow as function of potiential temperature"""
    return Qs0 * (Tpot / DT0) ** (1 + beta ) * np.exp(beta * Ta * (1/DT0 - 1/Tpot) )

def Hrad(age, Ur):
    """Radiogenic heating depending on age and Urey number"""
    return Ur * Qs0 * radio(1, age)[0] / radio(1, 0)[0]

def dTdt(age, Tpot, beta, Ur):
    """Time derivative of the potential temperature

    age: backward positive time, zero being the present
    Tpot: mantle potential temperature
    beta: exponent of the heat flow scaling
    Ur: present day Urey number
    """
    return (Qs(Tpot, beta) -Hrad(age, Ur)) / (ME * Cap) * year

def plot_evol(tt, Tm, Qt, Hr, text, annot=None):
    """Plot evolution with time of temperature and energy budget
    """
    # temperature as function of time
    fig, axe = plt.subplots(2, 1, sharex=True, figsize=(3, 5))
    fig.tight_layout()
    axe[0].plot(tt*1e-9, Tm)
    axe[0].set_ylabel('Potential temperature, C')

    # surface heat flow and radioactive heating as function of time
    ln1 = axe[1].plot(tt*1e-9, Qt * 1e-12, label='Surface heat flow')
    ln2 = axe[1].plot(tt*1e-9, Hr * 1e-12, label='Radiogenic heating')
    axe[1].set_ylim([15, 200])
    axe[1].set_xlabel('Age, Gyr')
    axe[1].set_ylabel('Power, TW')
    lns = ln1 + ln2

    if text=='fw':
        # Urey number on the same plot, different y axis
        ax2 = axe[1].twinx()
        lastU = Hr[-1]/Qt[-1]
        print(text, " final Urey number =", lastU)
        ln3 = ax2.plot([tt[0]*1e-9, tt[-1]*1e-9], [lastU, lastU],
                    '--', c='k', label='Last Urey #={:04.2f}'.format(lastU))
        ax2.set_ylabel('Urey number')
        ln4 = ax2.plot(tt*1e-9, Hr/Qt, label='Urey #', c='g')
        # ax2.legend(loc='center right')
        lns += ln4 + ln3 
    labs = [l.get_label() for l in lns]
    axe[1].legend(lns, labs, loc='center right')

    if annot is not None:
        for ax, ann in zip(axe, annot):
            plt.text(0.01, 0.99, ann, ha='left', va='top',
                 transform=ax.transAxes, fontsize=8, weight='bold')

    plt.savefig(text+'_ThEvol.pdf', bbox_inches='tight')
    plt.close()

# timestep
dt = 1e6

# maximum potential T to stop calculation
Tmax = 2000

# now integrate
r = ode(dTdt).set_integrator('vode', method='adams')

# integrating backward from present
# initial (present) condition
Urey = 18e12 / Qs0
print('Urey number :', Urey)
beta = 0.3
r.set_initial_value(DT0, 0).set_f_params(beta, Urey)

# initialize arrays
time = np.array([0])
Tm = DT0
Tman = np.array([DT0])
Qsur = np.array([Qs0])
Hra = np.array([Hrad(0, Urey)])

while r.successful() and r.t < amax and Tm < Tmax:
    Tm = r.integrate(r.t+dt)
    age = r.t + dt
    time = np.append(time, -age)
    Tman = np.append(Tman, Tm)
    Qsur = np.append(Qsur, Qs(Tm, beta))
    Hra = np.append(Hra, Hrad(age, Urey))

# now plot
plot_evol(time, Tman, Qsur, Hra, text='bw', annot=['c', 'd'])

# Forward in time
r.set_initial_value(DT0+200, amax).set_f_params(beta, Urey)

# initialize arrays
time = np.array([-amax])
Tm = DT0 + 200
Tman = np.array([Tm])
Qsur = np.array([Qs(Tm, beta)])
Hra = np.array([Hrad(amax, Urey)])

while r.successful() and r.t > dt and Tm < Tmax:
    Tm = r.integrate(r.t - dt)
    age = r.t - dt
    time = np.append(time, -age)
    Tman = np.append(Tman, Tm)
    Qsur = np.append(Qsur, Qs(Tm, beta))
    Hra = np.append(Hra, Hrad(age, Urey))

# now plot
plot_evol(time, Tman, Qsur, Hra, text='fw', annot=['a', 'b'])

