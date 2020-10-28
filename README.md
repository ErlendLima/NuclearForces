# NuclearForces

This is the main page for projects involving computing the phase shift
<sup>1</sup>*S*<sub>0</sub> partial wave of neutron-proton scattering. Project 1 sets up the basic scattering theory,
using K-matrix theory and the variable phase approach to 
compute the phase shift of the Reid potential and comparing it to data of the Nijmegen group. Project 2 uses effective field
theory to improve on the potential. Both consist of a Jupyter notebook for reproducing all results, and a `latex` directory  
containing the textual analysis.

The projects share the common Julia code, available [here](/src/). 

## Installing the Scattering Package

The Julia package exists as a Git submodule. To use it, clone this repository with

```shell
git clone --recurse-submodules git@github.com:ErlendLima/NuclearForces.git
```
If you have an older git version than 2.8, try
```shell
git clone --recursive  git@github.com:ErlendLima/NuclearForces.git
```
Once cloned, the `Scattering` directory needs to be added to Julia's path. 
For Linux, the directory needs to be linked to `$HOME/.julia/dev`:
```shell
ln -s <path to Scattering> $HOME/.julia/dev/
```
e.g.
```shell
ln -s ~/gits/NuclearForces/Scattering/ $HOME/.julia/dev/
```
Finally the package must be added. Go to the `$HOME/.julia/dev/`, open Julia with the `julia` command, 
type `]` to enter package mode, and add the package with `add .Scattering`. E.g.
```shell
> cd ~/.julia/dev/
> julia
(@v1.5) pkg> add Scattering
```
The package can now be imported as any other package as `using Scattering`. Both
notebooks will work correctly and reproduce all results.
