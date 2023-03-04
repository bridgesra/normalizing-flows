#!/bin/sh

# Run a fixed set of hyperparameters over all potential functions

for potential_function in "POT_5" "POT_6" #"POT_1"  "POT_2"  "POT_3"  "POT_4" "POT_5" "POT_6"
do                          
echo "Starting potential function "$potential_function
python3 ../src/fit_flow.py \
       --OUT_DIR ../out/ \
       --N_ITERS 50000 \
       --LR 1e-4 \
       --POTENTIAL $potential_function \
       --N_FLOWS 10\
       --BATCH_SIZE 100\
       --MOMENTUM .9\
       --N_PLOT_SAMPLES 25000

python3 ../src/fit_flow.py \
       --OUT_DIR ../out/ \
       --N_ITERS 15000 \
       --LR 1e-2 \
       --POTENTIAL $potential_function \
       --N_FLOWS 16\
       --BATCH_SIZE 100\
       --MOMENTUM .1\
       --N_PLOT_SAMPLES 10000

python3 ../src/fit_flow.py \
       --OUT_DIR ../out/ \
       --N_ITERS 15000 \
       --LR 1e-3 \
       --POTENTIAL $potential_function \
       --N_FLOWS 16\
       --BATCH_SIZE 400\
       --MOMENTUM .5\
       --N_PLOT_SAMPLES 10000



done
