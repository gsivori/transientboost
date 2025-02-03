"""
Copyright 2023 Gaston Sivori

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

using Random

function inject_bursts_replace(rng::StableRNG,pattern::AbstractMatrix{Bool}, burst_fraction::Float64 = 0.1, 
                               burst_duration::Float64 = 5.0, inter_spike_range::Tuple{Float64, Float64} = (2.5, 5.0), sim_δt::Float64=0.1)
    num_units, num_timesteps = size(pattern)
    spiking_units = findall((sum(pattern,dims=2).>0)[:])
    num_burst_units = round(Int, burst_fraction * length(spiking_units))

    # Convert time parameters from ms to indices
    burst_duration_steps = round(Int, burst_duration / sim_δt)
    inter_spike_min = round(Int, inter_spike_range[1] / sim_δt)
    inter_spike_max = round(Int, inter_spike_range[2] / sim_δt)

    burst_units = [spiking_units[unit] for unit in randperm(rng,length(spiking_units))[1:num_burst_units]]  # Randomly select units to burst

    for unit in burst_units
        # Find the time indices of existing spikes for this unit
        spike_times = findall(pattern[unit, :])

        # Create bursts for the unit's spike times
        for spike_time in spike_times
            # Determine the time window for the burst
            burst_window = spike_time:(spike_time + burst_duration_steps - 1)
            if maximum(burst_window) > num_timesteps
                continue  # Skip if burst would go out of bounds
            end

            # Create inter-spike intervals for the burst
            burst_times = []
            current_time = spike_time
            while current_time <= spike_time + burst_duration_steps - 1
                push!(burst_times, current_time)
                current_time += rand(rng,inter_spike_min:inter_spike_max)
                if current_time > spike_time + burst_duration_steps - 1
                    break
                end
            end

            # Replace the original spike with the burst
            pattern[unit, burst_window] .= false  # Clear the original window
            pattern[unit, burst_times] .= true  # Insert burst spikes
        end
    end

    return pattern
end


function generate_pattern_variations(input_pattern::AbstractMatrix{Bool}, num_variations::Int = 20, failure_rate::Float64 = 0.3)
    variations = []

    for _ in 1:num_variations
        variation = zeros(Bool, size(input_pattern))
        for i in 1:size(input_pattern, 1)  # Iterate over rows (units)
            for j in 1:size(input_pattern, 2)  # Iterate over columns (time steps)
                # Retain spike with a probability of (1 - failure_rate)
                if input_pattern[i, j] && rand() > failure_rate
                    variation[i, j] = true
                end
            end
        end
        push!(variations, variation)
    end

    return variations
end

# Define a function to calculate the instantaneous rate (example using a window)
function instantaneous_rate(matrix::AbstractMatrix{Bool}, window_size::Int = 11)
    padding_count = div(window_size - 1, 2)
    # Initialize the rate matrix
    rate_matrix = zeros(Float64, size(matrix))

    # Apply moving sum with padding for each row
    for i in 1:size(matrix, 1)
        padded_signal = vcat(zeros(padding_count), matrix[i, :], zeros(padding_count))
        convolved = conv(padded_signal, ones(window_size))
        
        # Trim to match the original size
        rate_matrix[i, :] = convolved[(padding_count*2+1):(end-padding_count*2)]
    end

    return rate_matrix
end

# Function to calculate center of mass (mean time) for each unit
function center_of_mass(spike_activity::AbstractMatrix{Float64})
    time_steps = collect(1:size(spike_activity, 2))  # Vector of time indices
    # Compute the weighted sum of time steps (center of mass)
    weighted_sum = sum(spike_activity .* time_steps', dims=2)
    total_spikes = sum(spike_activity, dims=2)
    
    # Avoid division by zero by using `NaN` for rows with zero spikes
    com = weighted_sum ./ total_spikes
    com[total_spikes .== 0] .= NaN  # Set `NaN` where there are no spikes

    return vec(com)  # Return as a vector of length `n`
end

# Function to normalize ranks within each burst
function normalized_ranks(center_of_masses)
    # Sort units by their center of mass
    sorted_indices = sortperm(center_of_masses, rev=false)
    num_units = length(center_of_masses)
    
    if num_units == 0
        return Float64[]  # Return an empty vector for no units
    elseif num_units == 1
        return [0.0]  # Return 0 for a single unit
    end
    
    # Normalize ranks from 0 to 1
    ranks = collect(0:(num_units - 1))
    normalized_ranks = ranks / (num_units - 1)
    
    # Apply sorting to get the ranks in the original order
    return normalized_ranks[sorted_indices]
end

# Function to compute the mean instantaneous rate across multiple pattern variations
function mean_instantaneous_rate(matrices::Vector{Any}, window_size::Int = 11)
    n = size(matrices[1], 1)  # Number of units (rows)
    T = size(matrices[1], 2)  # Number of time steps (columns)
    sum_rate_matrix = zeros(Float64, n, T)

    # Iterate over each matrix variation and accumulate the instantaneous rates
    for matrix in matrices
        rate_matrix = instantaneous_rate(matrix, window_size)
        sum_rate_matrix .+= rate_matrix
    end

    # Compute the mean instantaneous rate by dividing by the number of matrices
    mean_rate_matrix = sum_rate_matrix / length(matrices)

    return mean_rate_matrix
end

# Function to compute selectivity index to a range of [-1, 1]
function compute_selectivity(weights, pattern_indices, non_pattern_indices,alpha=0.5)
    pattern_weights = mean(weights[pattern_indices, :], dims=1)
    non_pattern_weights = mean(weights[non_pattern_indices, :], dims=1)
    std_non_pattern_weights = std(weights[non_pattern_indices, :], dims=1)
    si = (pattern_weights .- non_pattern_weights) ./ (std_non_pattern_weights .+ 1e-6)
    snorm = tanh.(alpha .* si)
    return snorm
end

function compute_learning_speed(SI::Matrix{Float64}, learned_index::Int, non_learned_indices::Vector{Int}, k::Float64=3., T_sustain::Int=1000)
    num_timesteps = size(SI, 2)
    learning_speed = NaN
    time_threshold = NaN
    act_threshold = 0.15
    for t in 1:num_timesteps
        # Compute baseline (mean and std) from non-learned patterns
        baseline_mean = mean(SI[non_learned_indices, t])
        baseline_std = std(SI[non_learned_indices, t])
        
        # Compute threshold based on off-pattern/non-learned inputs
        threshold = baseline_mean + k * baseline_std
        
        # Check if the learned pattern crosses any threshold
        if SI[learned_index, t] > threshold && SI[learned_index, t] > act_threshold
            # Check if sustained crossing
            if all(SI[learned_index, min(t + 1, num_timesteps):min(t + T_sustain, num_timesteps)] .> threshold)
                time_threshold = t * 0.1 # in ms
                learning_speed = 1 / (time_threshold)  # Time step is 0.1 ms
                return learning_speed, time_threshold   # Return speed once detected
            end
        end
    end
    
    return learning_speed, time_threshold  # Return NaN if no learning detected
end


#=
patsv = generate_input_patsV2(rng,2000,1000,3,5.0,0.1)
inject_bursts_replace(rng,patsv[1],0.1)
#perm = sortperm(1:size(burstingpat, 1), by = row -> first_true_index(burstingpat[row, :]))
#sorted_burstingpat = burstingpat[perm, :]
spy(sorted_burstingpat,size=(250,400),ylims=(0,2000),markersize=2.0,markershape=:vline,title="Generated Pattern",markercolor=:black,xlabel="Timestep",ylabel="Unit ID")

spy(patsv[1],size=(250,400),mc=:black,ms=2.0,markerstrokewidth=3.0,markeralpha=:black,markershape=:vline,title="Generated Pattern",xlabel="Timestep",ylabel="Unit ID")


pat_variations = generate_pattern_variations(patsv[1])

mrate_map = mean_instantaneous_rate(pat_variations)

#rate_mat = instantaneous_rate(patsv[1])

centers_of_mass = center_of_mass(mrate_map)
rankings = normalized_ranks(centers_of_mass)

#perm = sortperm(1:size(burstingpat, 1), by = row -> first_true_index(burstingpat[row, :]))
#sorted_burstingpat = burstingpat[perm, :]

perm = reverse(sort(rankings))

#perm = sortperm(1:size(burstingpat, 1), by = row -> first_true_index(burstingpat[row, :]))

sorted_burstingpat = patsv[1][perm, :]
=#
