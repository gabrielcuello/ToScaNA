! Synthetic D4 example: Ni powder in a V can, in a V furnace; vanadium in the furnace.
! CORRECT cannot treat the absorber measurement as a configuration: it is used here
! as the background of every measurement (the closest CORRECT can do).
! Cross sections and densities from ToScaNA's tables (same as build_state).
! Run from this folder:  python -m toscana.corrections.pycorrect Ni_powder.com
instrument d4
wavelength 0.5
beam 5.0000 0.8000
sample I_SCE.adat 0.2500 /density=0.045724 /temperature=300.0 /fullness=1.0
component 1.0 58.6934 18.5000 4.4900
container I_CE.adat 0.2600 /density=0.072230 /scattering=5.1047 /absorption=5.0800
furnace I_E.adat 2.0000 2.0100 /density=0.072230 /scattering=5.1047 /absorption=5.0800
vanadium I_VE.adat 0.3040 /furnace /density=0.072230 /scattering=5.1047 /absorption=5.0800
background I_ACE.adat
placzek series
spectrum 1
output Ni_powder_pycorrect.dat
execute /nolist
quit
