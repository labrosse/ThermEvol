#!/usr/bin/env python3
"""
Model for the thermal evolution of the Earth

Core and mantle treated together
Based on the model of Labrosse and Jaupart (2007). See that paper for full
explanations.
Labrosse, S., & Jaupart, C. (2007). Thermal evolution of the Earth: Secular 
changes and fluctuations of plate characteristics. 
Earth and Planetary Science Letters, 260(3–4). 
https://doi.org/10.1016/j.epsl.2007.05.046
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import ode
import sys

# To create pdf figures vith vector fonts
mpl.rc('font', family='sans-serif', size=7, **{'sans-serif': ['Arial']})
mpl.rcParams['pdf.fonttype'] = 42

# Set the annotations here, with classic french and english choices
# as examples.
# root name for files, may depend on the language choice
# language = 'French'
language = 'French'
if language == 'French':
    contlabel = r'Surface continental, \%'
    templabel = r'Température potentielle, C'
    agelabel = r'Âge, Ga'
    powlabel = r'Puissance, TW'
    Qtlabel = r'Flux de chaleur total en surface'
    radlabel = r'Production radioactive totale'
    oceanlabel = r'Flux de chaleur océanique'
    mant2cont = r'Flux manteau $\rightarrow$ continents'
    contprod = r'Production de chaleur continentale'
    mantprod = r'Production de chaleur mantellique'
    root = 'fr_'
else: # default language = 'English':
    contlabel = r'Continental area, \%'
    templabel = r'Potential temperature, C'
    agelabel = r'Age, Gyr'
    powlabel = r'Power, TW'
    Qtlabel = r'Total surface heat flow'
    radlabel = r'Total radiogenic heating'
    oceanlabel = r'Oceanic heat flow'
    mant2cont = r'Mantle to continents heat flow'
    contprod = r'Continental heat production'
    mantprod = r'Mantle heat production'
    root = 'en_'

# physical parameters
year = 365 * 24 * 3600 # year in seconds
ME = 5.972e24 # mass of the Earth in kg
Cap = 1200 # effective heat capacity (potentially taking account of IC crystallisation etc.)
DT0 = 1330 # present potential temperature, C.
Qs0 = 46e12 # present surface heat flow, W.
Qcont0 = 14e12 # present continental heat flow, W.
Hcont0 = 8e12 # present continental heat production, W.
Ta = 4e4 # activation temperature for viscosity, in C
amax = 4.5e9 # age of the Earth, in year
k = 3.15 # thermal conductivity W/m/K
kappa = 8e-7 # thermal diffusivity m^2/s
tmax = 180e6 * year # max seafloor age
RE = 6.371e6 # radius of the Earth in m
Atot = 4 * np.pi * RE * RE # Surface of the Earth
Aoc =3.09e14 # total oceanic surface at present
max_cont = 1 - Aoc / Atot # maximum continental fraction
qcont = (Qcont0 - Hcont0) / (Atot - Aoc) # heat flux density at the base of continents.
coef_oc = Atot * k / np.sqrt(np.pi * kappa * tmax)

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

def sigmoid(x):
    """Sigmoid function to use as continental area function"""
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def cont_surf(age, t1, t2):
    """Continental area as function of age

    assumes a sigmoidal growth curve with growth mostly between t1 and t2
    all times are ages, i.e. positive backward
    """
    x = 4 * (t1 - age) / (t1 - t2)
    xmax = 4 * t1 / (t1 - t2) # value at present to ensure we get there
    return max_cont * sigmoid(x) / sigmoid(xmax)

def lam(S):
    """Shape factor for the heat flow

    Varies linearly with the continental surface from 2 (no continent)
    to 8/3 with present continental area
    """
    return 2 *(1 + S / 3 / max_cont)

def Qoc(Tm, S):
    """Oceanic heat flow as function of mantle temperature and continental frac
    
    Proportional to actual oceanic surface (fraction 1 - S)
    """
    return coef_oc * (1 - S) * lam(S) * Tm


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
    """Surface heat flow as function of potential temperature"""
    return Qs0 * (Tpot / DT0) ** (1 + beta ) * np.exp(beta * Ta * (1/DT0 - 1/Tpot) )

def Hrad(age, Ur):
    """Radiogenic heating depending on age and Urey number

    Total value for the BSE
    """
    return Ur * Qs0 * radio(1, age)[0] / radio(1, 0)[0]

def Hcont(age, Ur, S):
    """Continental heat production as function

    input: age, Urey number, continental fraction (S)
    Assumes a constant enrichment of continents over the BSE
    """
    rH = Hcont0 / Hrad(0, Ur)
    return S / max_cont * Hrad(age, Ur) * rH

def Qcont(S):
    """Heat flow at the base of continents

    assumes the same basal heat flux density as at present
    """
    return S * Atot * qcont

def dTdt_LJ(age, Tpot, Ur, ti, tf):
    """Time derivative of the potential temperature

    age: backward positive time, zero being the present
    Tpot: mantle potential temperature
    ti: intiation of continent formation
    tf: end of continent formation
    Ur: present day Urey number
    """
    # time derivative of Tpot
    S = cont_surf(age, ti, tf)
    return (Qoc(Tpot, S) + Qcont(S) - (Hrad(age, Ur) - Hcont(age, Ur, S))) / (ME * Cap) * year

def plot_evol_LJ(tt, Tm, QQoc, QQco, HHco, rSco, Hr, text, annot=None):
    """Plot evolution with time of various diagnotics

    tt: time, positive forward
    Tm: mantle potential temperature
    QQoc: total oceanic heat flow
    QQco: Heat flow to continent bottom
    HHco: heat production in continents
    rSco: Continental surface fraction
    Hr: total heat production 
    """
    fig, axe = plt.subplots(3, 1, sharex=True)#, figsize=(3, 5))
    fig.tight_layout()
    # continental fraction as function of time
    axe[0].plot(tt*1e-9, rSco)
    axe[0].set_ylabel(contlabel)
    # temperature as function of time
    axe[1].plot(tt*1e-9, Tm)
    axe[1].set_ylabel(templabel)

    # surface heat flow and radioactive heating as function of time
    Qt = QQoc + QQco +HHco
    ln1 = axe[2].plot(tt*1e-9, Qt * 1e-12, label=Qtlabel)
    ln2 = axe[2].plot(tt*1e-9, Hr * 1e-12, label=radlabel)
    # various contributions
    ln3 = axe[2].plot(tt*1e-9, QQoc * 1e-12, '--', label=oceanlabel)
    ln4 = axe[2].plot(tt*1e-9, QQco * 1e-12, '--', label=mant2cont)
    ln5 = axe[2].plot(tt*1e-9, HHco * 1e-12, '--', label=contprod)
    ln6 = axe[2].plot(tt*1e-9, (Hr - HHco) * 1e-12, '--', label=mantprod)
    axe[1].set_xlim([-amax * 1e-9, 0])
    axe[2].set_xlabel(agelabel)
    axe[2].set_ylabel(powlabel)
    lns = ln1 + ln2 +  ln3 + ln4 + ln5 + ln6
    labs = [l.get_label() for l in lns]
    axe[2].legend(lns, labs, ncol=2, loc='upper right')

    if annot is not None:
        for ax, ann in zip(axe, annot):
            plt.text(0.01, 0.99, ann, ha='left', va='top',
                 transform=ax.transAxes, fontsize=8, weight='bold')

    plt.savefig(root+text+'_ThEvol_LJ.pdf', bbox_inches='tight')
    plt.close()

# timestep - in years
dt = 1e6

# maximum potential T
Tmax = 2000

# now integrate
r = ode(dTdt_LJ).set_integrator('vode', method='adams')

# integrating backward from present
# initial (present) condition
#Urey = 0.86848
Urey = 18e12 / Qs0
print('Urey number :', Urey)
ti = 3e9
tf = 1.5e9
print('present cooling rate = ', dTdt_LJ(0, DT0, Urey, ti, tf) * 1e9)
# sys.exit()
# beta = 0.3
r.set_initial_value(DT0, 0).set_f_params(Urey, ti, tf)

# initialize arrays
time = np.array([0])
Tman = np.array([DT0])
Tm = DT0
QOC = np.array([Qoc(Tm, max_cont)])
QCC = np.array([Qcont(max_cont)])
HCC = np.array([Hcont(0, Urey, max_cont)])
Hra = np.array([Hrad(0, Urey)])
ContS = np.array([max_cont])

while r.successful() and r.t < amax and Tm < Tmax:
    Tm = r.integrate(r.t+dt)
    age = r.t + dt
    time = np.append(time, -age)
    Tman = np.append(Tman, Tm)
    S = cont_surf(age, ti, tf)
    ContS = np.append(ContS, S)
    QOC = np.append(QOC, Qoc(Tm, S))
    QCC = np.append(QCC, Qcont(S))
    HCC = np.append(HCC, Hcont(age, Urey, S))
    Hra = np.append(Hra, Hrad(age, Urey))

# now plot
plot_evol_LJ(time, Tman, QOC, QCC, HCC, ContS, Hra, text='bw', annot=['a', 'b', 'c'])

