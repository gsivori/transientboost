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
using StableRNGs

gr(markersize=5.,markershape=:vline, markerstrokewidth=1.,markeralpha=1.0,titlefontsize=10,markercolor=:black,guidefont=(10,:black),tickfont=(8,:black),
background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=8)

include("funs.jl")
include("newfuns.jl")

#the figure 
#gr(minorgrid=false,grid=false,background_color_legend=:transparent,foreground_color_legend=:transparent,markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0,legendfontsize=16,left_margin=7Plots.mm,right_margin=3Plots.mm,bottom_margin=6Plots.mm,guidefont=(14, :black),tickfont=(14, :black))

l = @layout [[grid(2,3){0.65w} grid(2,1){0.35w}]; [[c{0.32h}; d{0.18h}; e{0.32h}; f{0.18h}] grid(2,1)]]

hsupp0 = plot(size=(900,1100),layout=l,dpi=300,left_margin=3Plots.mm)
plot!(hsupp0[1],title = "A", titleloc = :left, titlefont = font(18))
plot!(hsupp0[7],axis=false,grid=false)
plot!(hsupp0[8],title = "B", titleloc = :left, titlefont = font(18))
plot!(hsupp0[9],title = "C", titleloc = :left, titlefont = font(18))
plot!(hsupp0[11],title = "D", titleloc = :left, titlefont = font(18))

rng = StableRNG(2023)
pat_width = 1000
n_pats = 3
in_rate = 5.
n_in = 100
sim_δt = 0.1
ms_to_sec = 1e-3
sec_to_ms = ms_to_sec^-1

pats = generate_input_patsV2(rng,n_in, pat_width, n_pats, in_rate,sim_δt)

vals = []
y = []
for ci = 1:n_in
    times = view(pats[1],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end

scatter!(hsupp0[1],vals,y,ylabel="Input ID",xlabel="",xlims=(-5.,105.),legend=false,grid=false)
annotate!(hsupp0[1],(100, 105, (string("r=5 Hz","\n","Tpat=100 ms"), :right, 8)))
#title="r = 5 Hz | Tpat = 100ms"

bursting_percent = 0.2
for each in 1:n_pats
    inject_bursts_replace(rng,pats[each],bursting_percent)
end

vals = []
y = []
for ci = 1:n_in
    times = view(pats[1],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end

scatter!(hsupp0[4],vals,y,ylabel="Input ID",xlims=(-5.,105.),legend=false,grid=false,xlabel="Time (ms)")
annotate!(hsupp0[4],(100, 105, (string("bursting=20%"), :right, 8)))

pat_width = 2000
in_rate = 5.


pats = generate_input_patsV2(rng,n_in, pat_width, n_pats, in_rate,sim_δt)

vals = []
y = []
for ci = 1:n_in
    times = view(pats[3],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end

scatter!(hsupp0[2],vals,y,ylabel="",xlabel="",xlims=(-5.,205.),legend=false,grid=false)
annotate!(hsupp0[2],(200, 105, (string("r=5 Hz","\n","Tpat=200 ms"), :right, 8)))


bursting_percent = 0.2
for each in 1:n_pats
    inject_bursts_replace(rng,pats[each],bursting_percent)
end

vals = []
y = []
for ci = 1:n_in
    times = view(pats[3],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end

scatter!(hsupp0[5],vals,y,ylabel="",xlabel="Time (ms)",xlims=(-5.,205.),legend=false,grid=false)
annotate!(hsupp0[5],(200, 105, (string("bursting=20%"), :right, 8)))


pat_width = 1000
in_rate = 10.


pats = generate_input_patsV2(rng,n_in, pat_width, n_pats, in_rate,sim_δt)

vals = []
y = []
for ci = 1:n_in
    times = view(pats[2],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end

scatter!(hsupp0[3],vals,y,ylabel="",xlabel="",xlims=(-5.,105.),legend=false,grid=false)
annotate!(hsupp0[3],(100, 105, (string("r=10 Hz","\n","Tpat=100 ms"), :right, 8)))


bursting_percent = 0.2
for each in 1:n_pats
    inject_bursts_replace(rng,pats[each],bursting_percent)
end

vals = []
y = []
for ci = 1:n_in
    times = view(pats[2],ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0 * sec_to_ms)
    push!(y,ci*ones(length(times)))
end
annotate!(hsupp0[6],(100, 105, (string("bursting=20%"), :right, 8)))
scatter!(hsupp0[6],vals,y,ylabel="",xlabel="Time (ms)",xlims=(-5.,105.),legend=false,grid=false,xticks=[0,20,40,60,80,100])

using JLD2
@load "../jlds/tuning_rapidity_dataV3.jld2" results

sim_used = 25
tfactor = 2.0
sim_length = 10000.0 * tfactor
sim_δt = 0.1
sim_steps = Int(sim_length/sim_δt)
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor))) * ms_to_sec

patcolors = palette(:default)
pat_timings = results[0.0][sim_used][:pat_timings]
#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pat_timings[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pat_timings) if index < length(pat_timings)]
push!(timestamps,(timestamps[1][2],sim_length*ms_to_sec))

learned_pat = argmax(results[0.0][sim_used][:selectivity_matrix][:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(results[0.0][sim_used][:selectivity_matrix],learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(speed^-1/0.1*sec_to_ms)
time_threshold = time_threshold*ms_to_sec

plot!(hsupp0[8],grid=false,minorgrid=false);
hline!(hsupp0[8],[0.15],linestyle=:dash,linewidth=1.5,color=:gray,label="Activity threshold",markeralpha=0.0);
vline!(hsupp0[8],[time_threshold],ls=:dash,color=:brown,label="Tuning speed",markeralpha=0.0)
vspans(hsupp0,8:8,timestamps,pat_timings,(1.5,7.5),0.2);
[plot!(hsupp0[8],t,results[0.0][sim_used][:selectivity_matrix][each,:],color=patcolors[each],label="",markeralpha=0.0) for each in 1:3];
scatter!(hsupp0[8],[results[0.0][sim_used][:spike_data] .* ms_to_sec],ones(length(results[0.0][sim_used][:spike_data])).*0.25,ms=5.0,markershape=:vline,markercolor=:black,label="Spikes",xlims=(2,7),xlabel="Time (s)", ylabel="S.I.",ylims=(-0.25,0.5));

## ADD THIS NOW 

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

seed = 43
bursting_percent = 0.0

#simulation
include("lif_burst_input.jl") 
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
#gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
vspans(hsupp0,9:10,timestamps,pat_timings,window,0.2)
plot!(hsupp0[9],t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",xlim=window,legend=:topleft,markeralpha=0.0,linewidth=0.5)
plot!(hsupp0[9],t,v_soma,color="black",label="Somatic";dpi=dpi,markeralpha=0.,linewidth=0.5)
bar!(hsupp0[10],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss[sx:ex])), legend=false,markeralpha=0.0)

#SI results
patcolors = palette(:default)
npats = 3
si = zeros(Float64,(npats,sim_steps))
vspans(hsupp0[13],13:13,timestamps,pat_timings,window,0.2)
for eachpat in 1:npats
    tarr = Set(collect(1:n_in))
    pat_indices = [pop!(tarr,key) for key in findall(sum(pats[eachpat],dims=2)[:] .>= 1)]
    tarr = sort([each for each in tarr])
    selectivity_index = compute_selectivity(P,pat_indices,tarr)
    si[eachpat,:] .= selectivity_index[:]
    plot!(hsupp0[13],t,selectivity_index[:],color=patcolors[eachpat],markeralpha=0.,label="")
end
plot!(hsupp0[13],ylim=(-1.05,1.05),grid=false)

learned_pat = argmax(si[:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(si,learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(round(speed^-1/0.1))
time_threshold = time_threshold*ms_to_sec

vspans(hsupp0,13:13,timestamps,pat_timings,window,0.2)
vline!(hsupp0[13],[time_threshold],ls=:dash,color=:gray,label="Activity threshold",markeralpha=0.0)
hline!(hsupp0[13],[si[learned_pat,threshold_bar]],ls=:dash,color=:brown,label="Tuning speed",markeralpha=0.0)
plot!(hsupp0[13],ylim=(-0.25,1.05),grid=false,xlabel="Time (s)",ylabel="S.I.")

seed = 43
bursting_percent = 0.1

#simulation
include("lif_burst_input.jl")
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
vspans(hsupp0,11:12,timestamps,pat_timings,window,0.2)
plot!(hsupp0[11],t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",xlim=window,legend=:topleft,markeralpha=0.0,linewidth=0.5)
plot!(hsupp0[11],t,v_soma,color="black",label="Somatic";dpi=dpi,markeralpha=0.,linewidth=0.5)
bar!(hsupp0[12],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss[sx:ex])), legend=false,markeralpha=0.0)

#SI results



patcolors = palette(:default)
npats = 3
si = zeros(Float64,(npats,sim_steps))
vspans(hsupp0[14],14:14,timestamps,pat_timings,window,0.2)
for eachpat in 1:npats
    tarr = Set(collect(1:n_in))
    pat_indices = [pop!(tarr,key) for key in findall(sum(pats[eachpat],dims=2)[:] .>= 1)]
    tarr = sort([each for each in tarr])
    selectivity_index = compute_selectivity(P,pat_indices,tarr)
    si[eachpat,:] .= selectivity_index[:]
    plot!(hsupp0[14],t,selectivity_index[:],color=patcolors[eachpat],markeralpha=0.,label="")
end


learned_pat = argmax(si[:,end])
others = Set([1,2,3])
pop!(others,learned_pat)
speed, time_threshold = compute_learning_speed(si,learned_pat,sort!(collect(others)), 5., 10000)
threshold_bar = Int(speed^-1/0.1)
time_threshold = time_threshold*ms_to_sec

vspans(hsupp0,14:14,timestamps,pat_timings,window,0.2)
vline!(hsupp0[14],[time_threshold],ls=:dash,color=:gray,label="Activity threshold",markeralpha=0.0)
hline!(hsupp0[14],[si[learned_pat,threshold_bar]],ls=:dash,color=:brown,label="Tuning speed",markeralpha=0.0)
plot!(hsupp0[14],ylim=(-0.25,1.05),grid=false,xlabel="Time (s)",ylabel="S.I.")

plot!()