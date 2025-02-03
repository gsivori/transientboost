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

using Plots

#plotting parameters
n_in = 2000
ms_to_sec = 1e-3
sec_to_ms = 1000.
tfactor = 2.0
sim_length = 10000.0 * tfactor
sim_δt = 0.1
sim_steps = Int(sim_length/sim_δt)
ticks = 1000.0 * tfactor
bins = 50
dpi = 150
box_width = 250
box_len = box_width*sim_δt
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2))
k = zeros(Nbins)
pss = zeros(Int(sim_length/sim_δt))
pss_ci = zeros(Int(sim_length/sim_δt))
pss_in = zeros(Int(sim_length/sim_δt))
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor))) * ms_to_sec
figsize =(900,600)
default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray), framestyle = nothing, yminorgrid = true, xminorgrid = true, size=figsize, dpi=150)

#helper functions
include("funs.jl")
include("newfuns.jl")

seed = 43
bursting_percent = 0.0

#simulation
include("lif_burst_input.jl") #spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
spikes, ns, v_dend, v_soma, g, isis, cvs, stuff, poisson_in, w, cts = run_spk(sim_length,seed,bursting_percent,false);#, w, κ[end])

(ge, gi) = g;
(gain_mod,spk_train) = stuff;
(_,inputs,pat_timings,pats) = poisson_in;
(w_init, w, P) = w;
(κ, ϐ, μ) = cts;

#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
#output spikes
spikes = spikes * ms_to_sec
#output spikes
pss = bin_spikes(spikes*sec_to_ms,Nbins,box_width,sim_steps,sim_δt)

#single neuron response (m.p.)
sx = 1
ex = sim_steps
window = (0.0,20.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",
    sharex=true,layout=l,xlim=window, legend=:topleft,minorgrid=false,
    left_margin = 7Plots.mm,right_margin=7Plots.mm)
vspans(h3,timestamps,pat_timings,window,0.2)
plot!(h3[1],t,v_soma,color="black",label="Somatic";dpi=dpi)
bar!(h3[2],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss[sx:ex])),bottom_margin=5Plots.mm, legend=false)

patcolors = palette(:default)
npats = 3
si = zeros(Float64,(npats,sim_steps))
h = plot(grid=false,minorgrid=false,xlabel="Time (s)",ylabel="S.I.",legend=false,left_margin=3Plots.mm,bottom_margin=3Plots.mm,size=(600,300),dpi=150)
vspans(h,timestamps,pat_timings,window,0.2)
for eachpat in 1:npats
    tarr = Set(collect(1:n_in))
    pat_indices = [pop!(tarr,key) for key in findall(sum(pats[eachpat],dims=2)[:] .>= 1)]
    tarr = sort([each for each in tarr])
    selectivity_index = compute_selectivity(P,pat_indices,tarr)
    si[eachpat,:] .= selectivity_index[:]
    plot!(t,selectivity_index[:],color=patcolors[eachpat])
end
plot!(ylim=(-1.05,1.05))

learned_pat = argmax(si[:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(si,learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(speed^-1/0.1*sec_to_ms)
time_threshold = time_threshold*ms_to_sec

vline!([time_threshold],ls=:dash,color=:gray)
hline!([si[learned_pat,threshold_bar]],ls=:dash,color=:gray)
annotate!([(0.1,si[learned_pat,threshold_bar]+0.2,text("Threshold", 12, :left, :top, :black))])

seed = 43
bursting_percent = 0.1

#simulation
include("lif_burst_input.jl") #spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
spikes, ns, v_dend, v_soma, g, isis, cvs, stuff, poisson_in, w, cts = run_spk(sim_length,seed,bursting_percent,false);#, w, κ[end])

(ge, gi) = g;
(gain_mod,spk_train) = stuff;
(_,inputs,pat_timings,pats) = poisson_in;
(w_init, w, P) = w;
(κ, ϐ, μ) = cts;

#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
#output spikes
spikes = spikes * ms_to_sec
#output spikes
pss = bin_spikes(spikes*sec_to_ms,Nbins,box_width,sim_steps,sim_δt)

#single neuron response (m.p.)
sx = 1
ex = sim_steps
window = (0.0,20.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",
    sharex=true,layout=l,xlim=window, legend=:topleft,minorgrid=false,
    left_margin = 7Plots.mm,right_margin=7Plots.mm)
vspans(h3,timestamps,pat_timings,window,0.2)
plot!(h3[1],t,v_soma,color="black",label="Somatic";dpi=dpi)
bar!(h3[2],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss[sx:ex])),bottom_margin=5Plots.mm, legend=false)

patcolors = palette(:default)
npats = 3
si = zeros(Float64,(npats,sim_steps))
h = plot(grid=false,minorgrid=false,xlabel="Time (s)",ylabel="S.I.",legend=false,left_margin=3Plots.mm,bottom_margin=3Plots.mm,size=(600,300),dpi=150)
vspans(h,timestamps,pat_timings,window,0.2)
for eachpat in 1:npats
    tarr = Set(collect(1:n_in))
    pat_indices = [pop!(tarr,key) for key in findall(sum(pats[eachpat],dims=2)[:] .>= 1)]
    tarr = sort([each for each in tarr])
    selectivity_index = compute_selectivity(P,pat_indices,tarr)
    si[eachpat,:] .= selectivity_index[:]
    plot!(t,selectivity_index[:],color=patcolors[eachpat])
end
plot!(ylim=(-1.05,1.05))

learned_pat = argmax(si[:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(si,learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(speed^-1/0.1)
time_threshold = time_threshold*ms_to_sec

vline!([time_threshold],ls=:dash,color=:gray)
hline!([si[learned_pat,threshold_bar]],ls=:dash,color=:gray)
annotate!([(0.1,si[learned_pat,threshold_bar]+0.2,text("Threshold", 12, :left, :top, :black))])
annotate!([(si[],si[learned_pat,threshold_bar]+0.2,text("o", 12, :left, :top, :black))])




## GET DATA 

#Functions to compute rapidity
function get_tuning_rapidity_data(sim_length::Float64, seed::Int, burst_percentages::Vector{Float64}, trials::Int)
    results = Dict()
    for burst_percent in burst_percentages
        results[burst_percent] = []  # Initialize storage for each burst percentage
        print("Starting set of trials for ", burst_percent, "%.")
        for trial in 1:trials
            # Run the model
            spikes, _, _, _, _, _, _, _, poisson_in, w_data, _ = 
                run_spk(sim_length, seed + trial, burst_percent, false)

            (w_init, w, P) = w_data
            (_, _, pat_timings, pats) = poisson_in
            sim_steps = size(P, 2)  # Total simulation steps
            n_in = size(P, 1)  # Number of input neurons

            # Initialize selectivity index matrix
            si = zeros(Float64, (npats, sim_steps))
            
            # Compute selectivity index for each pattern
            for eachpat in 1:npats
                # Find indices for the current pattern
                tarr = Set(collect(1:n_in))
                pat_indices = [pop!(tarr, key) for key in findall(sum(pats[eachpat], dims=2)[:] .>= 1)]
                non_pat_indices = sort!(collect(tarr))  # Remaining indices

                # Calculate selectivity index
                selectivity_index = compute_selectivity(P, pat_indices, non_pat_indices)
                si[eachpat, :] .= selectivity_index[:]
            end

            # Determine the learned pattern and "other" patterns
            learned_pat = argmax(si[:, end])
            other_patterns = Set(1:npats)
            pop!(other_patterns,learned_pat)

            # Compute learning metrics
            speed, time_threshold = compute_learning_speed(
                si, learned_pat, sort!(collect(other_patterns)), 5.0, 10000
            )
            time_threshold = time_threshold * ms_to_sec  # Convert to seconds

            # Store trial data
            push!(results[burst_percent], Dict(
                :seed => seed + trial,
                :burst_percent => burst_percent,
                :tuning_time => time_threshold,
                :learning_speed => speed,
                :selectivity_matrix => si,
                :spike_data => spikes,
                :pat_timings => pat_timings,
                :weights => w,
                :patterns => pats
            ))
        end
    end
    return results
end


# Tuning rapidity data acquisition
using JLD2

seed = 42
bps = [0.0, 0.05, 0.08, 0.1, 0.12, 0.15]
trials = 100
npats = 3
tfactor = 2.0
sim_length = 10000.0 * tfactor
sim_δt = 0.1
sim_steps = Int(sim_length/sim_δt)



# Acquire data
results = get_tuning_rapidity_data(sim_length, seed, bps, trials)

# Save results using JLD2
@save "../jlds/tuning_rapidity_dataV4.jld2" results

# Load results
using JLD2
@load "../jlds/tuning_rapidity_dataV4.jld2" results

# Fix... 
#for burst_percent in bps
#    for trial in collect(1:trials)
#        results[burst_percent][trial][:tuning_time] *= sec_to_ms
#    end
#end
# Another Fix... 
#npats = 3
#for burst_percent in bps
#    for trial in collect(1:trials)
#        learned_pat = argmax(results[burst_percent][trial][:selectivity_matrix][:, end])
#        other_patterns = Set(1:npats)
#        pop!(other_patterns,learned_pat)
#        speed, time_threshold = compute_learning_speed(
#            results[burst_percent][trial][:selectivity_matrix], learned_pat, sort!(collect(other_patterns)), 5.0, 10000
#        )
#        time_threshold = time_threshold * ms_to_sec  # Convert to seconds
#       results[burst_percent][trial][:tuning_time] = time_threshold
#       results[burst_percent][trial][:learning_speed] = speed
#    end
#end
#A third fix!
for burst_percent in bps
    for trial in collect(1:trials)
        learned_pat = argmax(results[burst_percent][trial][:selectivity_matrix][:, end])
        other_patterns = Set(1:npats)
        pop!(other_patterns,learned_pat)
        speed, time_threshold = compute_learning_speed(
            results[burst_percent][trial][:selectivity_matrix], learned_pat, sort!(collect(other_patterns)), 5.0, 10000)
        time_threshold = time_threshold * ms_to_sec  # Convert to seconds
        learning_time = results[burst_percent][trial][:tuning_time] 
        exposures = [patstart*ms_to_sec*sim_δt for (whichpat,patstart) in results[0.0][1][:pat_timings] if whichpat==learned_pat]
        npres = length([exposure for exposure in exposures if exposure <= learning_time])
        if npres <= 0
            npres = NaN
        end
        results[burst_percent][trial][:tuning_presentation] = npres
    end
end




patcolors = palette(:default)
pat_timings = results[0.0][98][:pat_timings]
#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))

h9 = plot(size=(600,400),dpi=150,grid=false,minorgrid=false,guidefont=(12, :black),tickfont=(12,:black));
hline!([0.15],linestyle=:solid,linewidth=1.5,color=:gray,label="Activity threshold")
vspans(h9,timestamps,pat_timings,(0.,20.),0.2)
[plot!(h9,t,results[0.0][98][:selectivity_matrix][each,:],color=patcolors[each],label="") for each in 1:3]
scatter!([results[0.0][98][:spike_data] .* ms_to_sec],ones(length(results[0.0][98][:spike_data])).*0.25,ms=5.0,markershape=:vline,markercolor=:black,label="Spikes")
plot!(h9,xlims=(3,6),xlabel="Time (s)", ylabel="S.I.")
savefig("act_thresh.png")

using StatsPlots

# Data analysis: navigate through the dictionary to gather plot data.
tuning_per_condition = []
for burst_percent in bps
    tuning_times = []
    for trial in collect(1:trials)
        push!(tuning_times,results[burst_percent][trial][:tuning_presentation])
    end
    push!(tuning_per_condition,tuning_times)
end

colpal = palette(:inferno,10)[4:10]
plot(size=(600,300),dpi=300,grid=false,minorgrid=false,guidefont=(12,:black),tickfont=(12,:black),bottom_margin=3Plots.mm,left_margin=3Plots.mm)
for each_cond in 1:6
    data = [tuning_per_condition[each_cond][findall(.~ isnan.(tuning_per_condition[each_cond]))]]
    boxplot!(data,outliers=true,bar_width=0.4,color=colpal[each_cond],label="",linecolor=:black)
end
plot!(xticks = (1:6, string.(bps)),xlabel="Burst percentage", ylabel="Pattern presentation")
plot!(yticks=[0,5,10,15,20,25])
savefig("burst_pres.png")
savefig("burst_perc.pdf")

# Data analysis: navigate through the dictionary to gather plot data.
speed_per_condition = []
for burst_percent in bps
    speed_times = []
    for trial in collect(1:trials)
        learned_pat = argmax(results[burst_percent][trial][:selectivity_matrix][:, end])
        learned_at_time = results[burst_percent][trial][:tuning_time] * sec_to_ms / 0.1
        pat_pres = length([x for (x,patt) in results[burst_percent][trial][:pat_timings] if patt <= learned_at_time
            && x == learned_pat])
        push!(speed_times,results[burst_percent][trial][:learning_speed] * pat_pres)
    end
    push!(speed_per_condition,speed_times)
end


# Data analysis: navigate through the dictionary to gather plot data.
speed_per_condition = []
for burst_percent in bps
    speed_times = []
    for trial in collect(1:trials)
        learned_pat = argmax(results[burst_percent][trial][:selectivity_matrix][:, end])
        learned_at_time = results[burst_percent][trial][:tuning_time] * sec_to_ms / 0.1
        pat_pres = length([x for (x,patt) in results[burst_percent][trial][:pat_timings] if patt <= learned_at_time
            && x == learned_pat])
        push!(speed_times,results[burst_percent][trial][:learning_speed] * pat_pres)
    end
    push!(speed_per_condition,speed_times)
end

colpal = palette(:inferno,10)[4:10]
plot(size=(600,400),dpi=150,grid=false,minorgrid=false,guidefont=(12,:black),tickfont=(12,:black))
for each_cond in 1:6
    data = [speed_per_condition[each_cond][findall(.~ isnan.(speed_per_condition[each_cond]))]]
    boxplot!(data,outliers=false,box_width=0.3,color=colpal[each_cond],label="")
end
plot!(xticks = (1:6, string.(bps)),xlabel="Burst percentage", ylabel="Learning speed (Presentations/s)")

# Calculate mean and SEM for each metric
mean_tuning = mean.([tuning_per_condition[each][findall(.~ isnan.(tuning_per_condition[each]))] for each in 1:6])
sem_tuning = std.([tuning_per_condition[each][findall(.~ isnan.(tuning_per_condition[each]))] for each in 1:6]) / sqrt(trials)

mean_response = mean(speed_of_response)
sem_response = std(speed_of_response) / sqrt(trials)

mean_selectivity = mean(selectivity_indices)
sem_selectivity = std(selectivity_indices) / sqrt(trials)

# Store results in the dictionary
results[burst_percent] = (
    mean_tuning, sem_tuning,
    mean_response, sem_response,
    mean_selectivity, sem_selectivity
)


#Testing

seed = 44
bursting_percent = 0.1

#simulation
include("lif_burst_input.jl") #spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
spikes, ns, v_dend, v_soma, g, isis, cvs, stuff, poisson_in, w, cts = run_spk(sim_length,seed,bursting_percent,false);#, w, κ[end])

(ge, gi) = g;
(gain_mod,spk_train) = stuff;
(_,inputs,pat_timings,pats) = poisson_in;
(w_init, w, P) = w;
(κ, ϐ, μ) = cts;

#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
#output spikes
spikes = spikes * ms_to_sec
#output spikes
pss = bin_spikes(spikes*sec_to_ms,Nbins,box_width,sim_steps,sim_δt)

#seed 43, bursting_percent = 0.1

patcolors = palette(:default)
npats = 3
si = zeros(Float64,(npats,sim_steps))
h = plot(grid=false,minorgrid=false,xlabel="Time (s)",ylabel="S.I.",legend=false,left_margin=3Plots.mm,bottom_margin=3Plots.mm,size=(600,300),dpi=150)
vspans(h,timestamps,pat_timings,window,0.2)
for eachpat in 1:npats
    tarr = Set(collect(1:n_in))
    pat_indices = [pop!(tarr,key) for key in findall(sum(pats[eachpat],dims=2)[:] .>= 1)]
    tarr = sort([each for each in tarr])
    selectivity_index = compute_selectivity(P,pat_indices,tarr)
    si[eachpat,:] .= selectivity_index[:]
    plot!(t,selectivity_index[:],color=patcolors[eachpat])
end
plot!(ylim=(-1.05,1.05))

learned_pat = argmax(si[:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(si,learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(speed^-1/0.1*sec_to_ms)
time_threshold = time_threshold*ms_to_sec

vline!([time_threshold],ls=:dash,color=:gray)
hline!([si[learned_pat,threshold_bar]],ls=:dash,color=:gray)
annotate!([(0.1,si[learned_pat,threshold_bar]+0.2,text("Threshold", 12, :left, :top, :black))])
savefig("si.png")



#Other test

seed = 44
bursting_percent = 0.2

#simulation
include("lif_burst_input.jl") #spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
spikes, ns, v_dend, v_soma, g, isis, cvs, stuff, poisson_in, w, cts = run_spk(sim_length,seed,bursting_percent,false);#, w, κ[end])

(ge, gi) = g;
(gain_mod,spk_train) = stuff;
(_,inputs,pat_timings,pats) = poisson_in;
(w_init, w, P) = w;
(κ, ϐ, μ) = cts;

#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
#output spikes
spikes = spikes * ms_to_sec
#output spikes
pss = bin_spikes(spikes*sec_to_ms,Nbins,box_width,sim_steps,sim_δt)


function moving_average(data::Vector{Float64}, window_size::Int)
    cumsum_vec = cumsum(vcat(0.0, data))
    smooth_data = (cumsum_vec[(window_size+1):end] .- cumsum_vec[1:end-window_size]) ./ window_size
    return vcat(fill(smooth_data[1], window_size), smooth_data)
end


# Function to compute mean and s.e.m. across trials
function compute_mean_sem_selectivity(selectivity_matrices::Vector{Matrix{Float64}})
    # Stack selectivity matrices along a new dimension
    all_selectivity = hcat(selectivity_matrices...)
    mean_selectivity = mean(all_selectivity, dims=2)
    sem_selectivity = std(all_selectivity, dims=2) ./ sqrt(size(all_selectivity, 2))
    return mean_selectivity, sem_selectivity
end








# Run the test
results = test_tuning_rapidity(sim_length, seed, burst_percentages, trials)

using Plots

# Extract values for each metric
means_tuning = [results[b][1] for b in burst_percentages]
sems_tuning = [results[b][2] for b in burst_percentages]

means_response = [results[b][3] for b in burst_percentages]
sems_response = [results[b][4] for b in burst_percentages]

means_selectivity = [results[b][5] for b in burst_percentages]
sems_selectivity = [results[b][6] for b in burst_percentages]

# Plot tuning rapidity
plot(burst_percentages, means_tuning, ribbon=sems_tuning, xlabel="Bursting Percentage", ylabel="Tuning Rapidity (ms)", lw=2, label="Mean ± SEM (Tuning)")

# Plot first response time
plot!(burst_percentages, means_response, ribbon=sems_response, xlabel="Bursting Percentage", ylabel="Time to First Response (ms)", lw=2, label="Mean ± SEM (First Response)")

# Plot selectivity index
plot!(burst_percentages, means_selectivity, ribbon=sems_selectivity, xlabel="Bursting Percentage", ylabel="Selectivity Index", lw=2, label="Mean ± SEM (Selectivity)")









#input spikes
vals = []
y = []
for ci = 1:n_in
    times = view(inputs,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pat_timings,sim_δt)
#input spikes pss
for ci = 1:n_in
    for i = 1:Nbins
        k[i] = count(spk->((i-1)*box_len <= spk <= (i+1)*box_len),view(vals,ci,:)[1])
        pss_ci[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
    end
    pss_in[:] += pss_ci[:]
end

#histogram of connections
h1_e = histogram(w_init[w_init.>0],bins=bins*2,alpha=0.4, color="red",legend=false)
histogram!(w[w.>0],bins=bins*2,alpha=0.5, color="purple",legend=false,xlim=(0.0,maximum(w)))
h1_i = histogram(w_init[w_init.<0],bins=bins*2,alpha=0.4, color="red",legend=false)
histogram!(w[w.<0],bins=bins*2,color="pink",alpha=0.5,legend=false,xlim=(minimum(w),0.0))

#histogram gif
@gif for ti ∈ 1:100000
    histogram(P[:,ti],bins=bins*2,legend=false,xlim=(-5,5))
end every 100

#cv trace
gr(markersize=0.0)
h6 = plot(t,cvs,color="blue",xlabel="Time (ms)",ylabel="Coefficient of Variation (AU)",xlim=(100.0,sim_length), xticks=0:ticks:sim_length;dpi=dpi)
#isi histogram
h5 = histogram(isis, bins=bins, xlabel="ISI (ms)", ylabel="Frequency ",kde=true, legend=false)

#synaptic input w pattern bars
h4 = plot(t[sx:end],ge[sx:end],color="blue",xlabel="Time (ms)",ylabel="Synaptic input (nS)", label="Excitatory",alpha=0.5, xticks=0:ticks:sim_length;dpi=dpi)
plot!(h4, t[sx:end],-1 .* gi[sx:end],color="red",alpha=0.5,label="Inhibitory";dpi=dpi, legend=:bottomright)
vspans(h4,timestamps,pattern)

#gain trace w pattern bars
sx = 1#Int(round((sim_length-ticks)/sim_δt,digits=1))#
ex = Int(round(sim_length/sim_δt,digits=1))#ticks#
gr(markersize=0.0, markerstrokewidth=0.0)
h7 = plot(t[sx:ex],gain_mod[sx:ex],color="blue", xlabel="Time (ms)", xlim=(sx*sim_δt,ex*sim_δt),legend=false, dpi=dpi)
vspans(h7,timestamps,pattern)

#just barplot
startx = 1#sim_length-ticks#
sx = 1#Int(round((sim_length-ticks)/sim_δt,digits=1))#1#
gr(markersize=0.0)
h9 = bar(t[sx:end],pss[sx:end],xlabel="Time (ms)",ylabel="Spk. count",xlim=(startx,sim_length),xticks=0:ticks:sim_length,ylim=(0,maximum(pss[sx:end]));dpi=dpi, yminorgrid=true, legend=false)
vspans(h9,timestamps,pattern)

#input raster plot
startr=1.0
endr=sim_length
gr(markersize=2.5,markershape=:circle,legend=false, markerstrokewidth=0.0)
h0 = scatter(xs,ys,group=grouping,markercolor=[:black :green :blue :red], xlabel="Time (ms)",ylabel="Input ID",xlim=(startr,endr),ylim=(0,n_in))#,xticks=0:ticks:sim_length)

#normalizing constant change over time
h10 = plot(t,κ,color="blue",xlabel="Time (ms)",ylabel="Normalizing constant (a.u.)",legend=false,xticks=0:ticks:sim_length;dpi=dpi)
vspans(h10,timestamps,pattern)
#sum of exc. weights change over time
h11 = plot(t,ϐ,color="red",xlabel="Time (ms)",ylabel="Sum of exc. weights (a.u.)",legend=false,xticks=0:ticks:sim_length;dpi=dpi)
vspans(h11,timestamps,pattern)
#mean spiking over time
h12 = plot(t,μ,color="green",xlabel="Time (ms)",ylabel="Mean spike train",legend=false,xticks=0:ticks:sim_length,linewidth=1.0,markersize=0.0,markeralpha=0.0)
vspans(h12,timestamps,pattern)

h12b = plot(t,spk_train .- μ,color="green",xlabel="Time (ms)",ylabel="Mean spike train",legend=false,xticks=0:ticks:sim_length,linewidth=1.0,markersize=0.0,markeralpha=0.0)
vspans(h12b,timestamps,pattern)

savefig(h1, "hist_w.png")
savefig(h1_e, "hist_exc.png")
savefig(h1_i, "hist_inh.png")
savefig(h4, "syn_input.png")
#savefig(h6, "coef_var.png")
savefig(h7, "gain_mod.png")
#savefig(h9, "firing_rate.png")
#savefig(h5, "somatic_firing_hist.png")
#savefig(h0, "input_raster.png")
savefig(h3, "memb_traces.png")
savefig(h10, "norm_const.png")
savefig(h11, "sum_weights.png")
#savefig(h12, "mean_spk_train.png")

#mean-spiking activity change over time
μreal = circshift(-1 .* (μ .- spk_train),(-1000))
sx = Int(round((sim_length-ticks)/sim_δt+1000,digits=1))#1#
ex = Int(round(sim_length/sim_δt-1000,digits=1))#ticks#
h15 = plot(t[sx:ex],μreal[sx:ex],color="green",xlabel="Time (ms)",ylabel="Mean spike train",legend=false;dpi=dpi,xlim=(sx*sim_δt,ex*sim_δt))
vspans(h15,timestamps,pattern)
savefig(h15, "mean_spiking.png")

#plasticity induction w pattern bars (not full)
sx = Int(round((sim_length-ticks)/sim_δt,digits=1))#
n = 1015
gr(markersize=0.0)
h16 = plot(t[sx:end],PI[n,sx:end],color="brown",xlabel="Time (ms)",ylabel="Plasticity induction (nS)", xticks=0:ticks:sim_length;dpi=dpi, legend=false)
vspans(h16,timestamps,pattern)

#gcd activity change over time
sx = 1#Int(round((sim_length-ticks)/sim_δt,digits=1))
ex = Int(round(sim_length/sim_δt,digits=1))
h17 = plot(t[sx:ex],gcd[sx:ex],color="purple",xlabel="Time (ms)",ylabel="Coupling conductance (nS)",legend=false;dpi=dpi,xlim=(sx*sim_δt,ex*sim_δt),alpha=0.8)
vspans(h17,timestamps,pattern)
savefig(h17, "coupl_cond.png")
