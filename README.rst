ThermEvol
======

Simple scripts to compute thermal evolution models for the Earth.

Two alternative models are provided here:
* ThermEvol.py
The simplest possible model, using a standard scaling law for the mantle heat
flow as function of the Rayleigh number.
The Earth is assumed to evolve uniformly, i.e. the core and mantle evolve at
proportional rates. Only one temperature is used to parameterize the whole Earth.
Values of the various parameters are reasonable from the present knowledge
but can easily be varied in the preamble.
The calculation is done both forward in time, from a temperature
200K larger than the current one and backward from the present situation.
The foward in time calculation gives a present surface heat flow lower than
the one observed. The backward calculation leads to the classical thermal
catastrophe at about 1Gyr age.
* ThermEvol_LJ.py
The model proposed by Labrosse and Jaupart (2007)
(https://doi.org/10.1016/j.epsl.2007.05.046).
Compared to the other model, the heat flow scales linearly with temperature
assuming that subduction is controlled by continents rather than the intrinsic
stability of oceanic plates.

* Requirements
These script need the following packages to run, all installable using pip or panda:
``numpy``, ``scipy``, ``matplotlib``.

* Running
Simply run
``python ThermEvoll.py``
or 
``python ThermEvoll_LJ.py``



