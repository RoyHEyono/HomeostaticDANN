#!/usr/bin/env bash

# Grid parameters
brightness_factors=(0 0.5 0.75 1)
normtypes=(0 1)
normtype_detach=(0 1)
excitatory_only=(0 1)

# Get the lengths of each grid
num_brightness_factors=${#brightness_factors[@]}
num_normtypes=${#normtypes[@]}
num_normtype_detach=${#normtype_detach[@]}
num_excitatory_only=${#excitatory_only[@]}

# Generate all combinations
for grid_index in $(seq 0 $((num_brightness_factors * num_normtypes * num_normtype_detach * num_excitatory_only - 1))); do
    brightness_factor_idx=$((grid_index % num_brightness_factors))
    normtype_idx=$(((grid_index / num_brightness_factors) % num_normtypes))
    detach_normtype_idx=$(((grid_index / (num_brightness_factors * num_normtypes)) % num_normtype_detach))
    excitatory_only_idx=$(((grid_index / (num_brightness_factors * num_normtypes * num_normtype_detach)) % num_excitatory_only))

    brightness_factor=${brightness_factors[$brightness_factor_idx]}
    normtype=${normtypes[$normtype_idx]}
    detach_normtype=${normtype_detach[$detach_normtype_idx]}
    excitatory_only=${excitatory_only[$excitatory_only_idx]}

    # Print the combination
    echo "Combination $grid_index:"
    echo "Brightness Factor: $brightness_factor"
    echo "Normtype: $normtype"
    echo "Detach Normtype: $detach_normtype"
    echo "Excitatory Only: $excitatory_only"
    echo "------------------------------------"
done
