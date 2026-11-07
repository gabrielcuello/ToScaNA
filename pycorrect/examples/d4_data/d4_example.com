! Synthetic D4 run: Ni sample in a vanadium can
instrument d4
wavelength 0.4985
beam 5.0 1.5
sample sample.dat 0.3 /density=0.0913 /temperature=300
component 1 ni
container can.dat 0.32 /type=v
vanadium vanadium.dat 0.3 /smoothing=-5
background background.dat
placzek series
spectrum 1
output corrected.dat
execute /nolist
