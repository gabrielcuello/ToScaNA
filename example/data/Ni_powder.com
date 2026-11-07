! Synthetic D4 example: Ni powder in a V can, in a furnace (used as background).
! CORRECT cannot use the absorber measurement (I_ACE.adat).
! Cross sections and densities from ToScaNA's tables (same as build_state).
! Run from this folder:  python -m toscana.corrections.pycorrect Ni_powder.com
instrument d4
wavelength 0.5
beam 5.0000 0.8000
sample I_SCE.adat 0.2500 /density=0.045724 /temperature=300.0 /fullness=1.0
component 1.0 58.6934 18.5000 4.4900
container I_CE.adat 0.2600 /density=0.072230 /scattering=5.1047 /absorption=5.0800
vanadium I_VE.adat 0.3040 /density=0.072230 /scattering=5.1047 /absorption=5.0800
background I_E.adat
placzek series
spectrum 1
output Ni_powder_pycorrect.dat
execute /nolist
quit
