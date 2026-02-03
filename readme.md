# Waves Equalizer Project

## Overview
This project analyses wave data from square-wave inputs and computes an equalization filter to correct dispersion effects.

## Folder structure

## Commands to run script
### Run script to find equalizer function from one of the square waves at a given frequency and save it to a file
python analysis.py "Waves project\EXAMPLE.csv" --save-eq "Waves project\equalizer.npz"
### Use this equalizer function to try and recover a message sent and its output later down the wire
python analysis.py "Waves project\EXAMPLE.csv" --apply-eq "Waves project\equalizer.npz"