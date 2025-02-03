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

## Data for cross-correlation
using Plots
using StableRNGs

#plotting parameters
n_in = 2000
sim_δt = 0.1
sec_to_ms = 1e-3
tfactor = 1.0
sim_length = 10000.0 * tfactor
sim_steps = Int(sim_length/sim_δt)
ticks = 1000.0 * tfactor
bins = 50
dpi = 150
box_width = 250
box_len = box_width*sim_δt
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2))
k = zeros(Nbins)
pss_ci = zeros(Int(sim_length/sim_δt))
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor)))
default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray), framestyle = nothing,
    yminorgrid = true, xminorgrid = true, size=(1800,1200), dpi=150)

#helper functions
include("funs.jl")

rng = StableRNG(2023)
seeds = sample(rng,collect(1:10000),500,replace=false)

spikes,_,_,_,_,_,_,_,poisson_in,_,_,_ = run_cell(sim_length,seeds[1])
#simulation seeds which converged pattern tuning
include("bpHVAneur.jl")
tuning_seeds = []
for pseed in seeds
    spikes,_,_,_,_,_,_,_,poisson_in,_,_,_ = run_cell(sim_length,pseed)
    (_,_,pattern,_) = poisson_in
    #timestamps
    timestamps = [(pat_time*sim_δt,round(pattern[index+1][2]*sim_δt,digits=1)) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
    push!(timestamps,(timestamps[end][2],sim_length))
    #output spikes
    pss = bin_spikes(spikes,Nbins,box_width,sim_steps,sim_δt)
    #last pattern spiking activity
    last_start = []
    last_end = []
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last green pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 1 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last blue pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 2 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last red pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 3 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    #looking at the spike output of the last pattern presentation
    auxg = mean(pss[last_start[1]:last_end[1]])
    auxb = mean(pss[last_start[2]:last_end[2]])
    auxr = mean(pss[last_start[3]:last_end[3]])

    thresh = 2.5 # much greater than 3 spikes as safety threshold
    comps = findall([auxg,auxb,auxr] .> thresh)
    if length(comps) == 1 # only one tuning
        push!(tuning_seeds,pseed)
    end
end

io = open("seeds_c2000_p100_15k_Jan29.txt", "w") do io
  for x in tuning_seeds
    println(io, x)
  end
end

#simulation
include("bpHVAneur.jl")
allchecks = []
spikes_chk = []
for pseed in tuning_seeds[1:100]
    _,_,_,_,_,_,_,_,_,_,_,chks = run_cell(sim_length,pseed)
    (spikes_chk, _,_,_) = chks
    push!(allchecks,spikes_chk)
end

#output checked spikes
n_tests = size(spikes_chk)[2]
all_pss = []
for (seed,check) in enumerate(allchecks)
    pss_chk = zeros(Int(sim_length/sim_δt),n_tests)
    for te = 1:n_tests
        for i = 1:Nbins
            k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),check[:,te])
            pss_chk[1+Int((i-1)*box_width):Int((i)*box_width),te] .= k[i]
        end
    end
    push!(all_pss,pss_chk)
end

using JLD2
@save "seeds_15k_jitters.jld2" all_pss

##

using DataFrames,CSV
using JLD2

@load "Project1/seedseeds_15k_jitters.jld2" all_pss # copy the file first

n_tests = size(all_pss[1])[2]

ccs = []
for each in all_pss
    cc = crosscor(each[:,1:n_tests-1],each[:,n_tests],[1],demean=false)
    push!(ccs,cc)
end
tests = vcat(ccs...)

test_means = hcat([mean(tests[:,10*(i-1)+1:10*i],dims=2) for i in 1:6]...)
#test_corrs = reshape(cor(pss_chk[:,1:n_tests-1],pss_chk[:,n_tests]),(20,6))
df = DataFrame(test_means,:auto)
rename!(df,["5", "10", "15","25","50","100"])
CSV.write("ccorrs_100s_c2000_p100_15k.csv",df)
#fig3b: plot with boxplot_jitters.py

#before all seeds:
#n_pat = 2000 but <780> on average.
#t_pat = 100
#overlapping = true
#only_exc = false
#what happens with different inputs? e.g.
#n_pat = 2000,1500,1000,500
#t_pat = 100,75,50,25
#overlapping = false
#only_excitatory = true

# we analyze three cases:
# 1) n_pat=1000, t_pat=100, only_excitatory=false, overlapping=true
# 2) n_pat=500, t_pat=50, only_excitatory=false, overlapping=true
# 3) n_pat=100, t_pat=100, 2x freq, only_excitatory=true, overlapping=false

#simulation seeds which converged pattern tuning for case 1)
include("bpHVAneur.jl")
tfactor = 1.5
sim_length = 10000.0 * tfactor
sim_δt = 0.1
ticks = 1000.0 * tfactor
bins = 50
dpi = 150
box_width = 250
box_len = box_width*sim_δt
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2))
k = zeros(Nbins)
pss_ci = zeros(Int(sim_length/sim_δt))
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor)))
default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray), framestyle = nothing, yminorgrid = true, xminorgrid = true, size=(1800,1200), dpi=150)
rngs = sample(1500:10000,5,replace=false)
tuning_seeds = []
for pseed in rngs
    spikes,_,_,_,_,_,_,_,poisson_in,_,_,_ = run_cell(sim_length,pseed)
    (_,pattern,_,_,_) = poisson_in
    #timestamps
    timestamps = [(pat_time*sim_δt,round(pattern[index+1][2]*sim_δt,digits=1)) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
    push!(timestamps,(timestamps[end][2],sim_length))
    #output spikes
    pss = zeros(Int(sim_length/sim_δt))
    for i = 1:Nbins
        k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
        pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
    end
    last_start = []
    last_end = []
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last green pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 1 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last blue pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 2 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last red pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 3 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    #looking at the spike output of the last pattern presentation
    auxg = mean(pss[last_start[1]:last_end[1]])
    auxb = mean(pss[last_start[2]:last_end[2]])
    auxr = mean(pss[last_start[3]:last_end[3]])

    thresh = 2.5 # much greater than 3 spikes as safety threshold
    comps = findall([auxg,auxb,auxr] .> thresh)
    if length(comps) == 1 # only one tuning
        push!(tuning_seeds,pseed)
    end
end

# 3) n_pat=100, t_pat=100, 2x freq, only_excitatory=true, overlapping=false
io = open("seeds_c100_p100_15k_oet_ovf.txt", "w") do io
  for x in tuning_seeds
    println(io, x)
  end
end
# 4509 & 4472 seeds with above parameters worked really well.
#simulation
include("bpHVAneur.jl")
spikes,ns,v_dend,v_soma,g,isis,cvs,gain_mod,poisson_in,w,cts,chks = run_cell(sim_length,tuning_seeds[1])
(ge,gi,g_cs,Icsd,Ca) = g
(w_init,w, P) = w
(inputs,pattern,pat1,pat2,pat3) = poisson_in
(κ, ϐ) = cts
(spikes_chk, ns_chk,v_soma_chk,v_dend_chk) = chks

#timestamps
timestamps = [(pat_time*sim_δt,round(pattern[index+1][2]*sim_δt,digits=1)) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
push!(timestamps,(timestamps[end][2],sim_length))
#output spikes
pss = zeros(Int(sim_length/sim_δt))
for i = 1:Nbins
    k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
    pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
end

last_start = []
last_end = []
for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last green pattern
    (pat,_) = reverse(pattern)[id]
    t0 = Int(round(t0/sim_δt,digits=1))
    t1 = Int(round(t1/sim_δt,digits=1))
    if pat == 1 && (t1 - t0) == 1000
        push!(last_start,t0)
        push!(last_end,t1)
        #break
    end
end
for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last blue pattern
    (pat,_) = reverse(pattern)[id]
    t0 = Int(round(t0/sim_δt,digits=1))
    t1 = Int(round(t1/sim_δt,digits=1))
    if pat == 2 && (t1 - t0) == 1000
        push!(last_start,t0)
        push!(last_end,t1)
        break
    end
end
for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last red pattern
    (pat,_) = reverse(pattern)[id]
    t0 = Int(round(t0/sim_δt,digits=1))
    t1 = Int(round(t1/sim_δt,digits=1))
    if pat == 3 && (t1 - t0) == 1000
        push!(last_start,t0)
        push!(last_end,t1)
        break
    end
end

#single neuron response (m.p.)
sx = 1
ex = Int(round(sim_length/sim_δt,digits=1))
#sx = last_start[1]
#ex = last_end[1]
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",
    sharex=true,layout=l,xlim=(sx*sim_δt,ex*sim_δt), legend=:bottomright,
    left_margin = 7Plots.mm, right_margin=7Plots.mm)
plot!(h3[1],t,v_soma,color="black",label="Somatic")
bar!(h3[2],t,pss,xlabel="Time (ms)",ylabel="Spk. count",xlim=(sx*sim_δt,ex*sim_δt),ylim=(0,maximum(pss[sx:ex])),
    bottom_margin = 7Plots.mm, legend=false, yminorgrid=true)
vspans(h3,1:2,timestamps,pattern,0.4)
savefig(h3,"mps.png")

#synaptic input w pattern bars
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
sx = last_start[7]
ex = last_end[7]
h4 = plot(t,ge,color="blue",xlabel="Time (ms)",ylabel="Synaptic input (nS)",
    label="Excitatory",alpha=0.3;dpi=dpi,
    left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, right_margin=5Plots.mm)
plot!(h4, t,-1 .* gi,color="red",alpha=0.3,label="Inhibitory";dpi=dpi, legend=:bottomright,xlim=(sx*sim_δt,ex*sim_δt))

green_start = []
green_end = []
for (id, (t0, t1)) in enumerate(timestamps)
    (pat,_) = pattern[id]
    t0 = Int(round(t0/sim_δt,digits=1))
    t1 = Int(round(t1/sim_δt,digits=1))
    if pat == 1 && (t1 - t0) == 1000
        push!(green_start,t0)
        push!(green_end,t1)
    end
end
#input spikes
insets = zeros(Int64,n_in,length(green_start),green_end[1]-green_start[1])
for ci = 1:n_in
    times = view(inputs,ci,:)
    for (id,start) in enumerate(green_start)
        insets[ci,id,:] = times[start:green_end[id]-1]
    end
end
pat_cells = findall(any(pat1,dims=2))
outsets = zeros(Float64,length(green_start),green_end[1]-green_start[1])
for pres in 1:length(green_start)
    ⅄ = 0.0
    for (ii, ti) in enumerate(green_start[pres]:green_end[pres]-1)
        d⅄ = sum(insets[:,pres,ii] .* P[:,ti]) - ⅄/τ⅄
        ⅄ += sim_δt*d⅄
        outsets[pres,ii] = ⅄
    end
end



h000 = plot(legend=:topleft)
wc = reverse(collect(1:10)./10)
for (ii,each) in enumerate(1:10)#length(green_start))
    plot!(h000,tpat[3:end],diff(diff(outsets[each,:])),color=cgrad(:grays)[wc[ii]])
end
plot!()

⅄ = 0.0
pat1⅄ = zeros(Float64,green_end[1]-green_start[1])
for ti in n_tpat
    d⅄ = sum(pat1[:,ti] .* w_init) - ⅄/τ⅄
    ⅄ += sim_δt*d⅄
    pat1⅄[ti] = ⅄
end
plot!(h000,tpat,pat1⅄,linewidth=2.0)
⅄ = 0.0
pat1⅄ = zeros(Float64,green_end[1]-green_start[1])
for ti in n_tpat
    d⅄ = sum(pat1[:,ti] .* w) - ⅄/τ⅄
    ⅄ += sim_δt*d⅄
    pat1⅄[ti] = ⅄
end
plot!(h000,tpat,pat1⅄,linewidth=2.0)



#simulation seeds which converged pattern tuning for case 2)
include("bpHVAneur.jl")
tfactor = 1.5
sim_length = 10000.0 * tfactor
rngs = sample(1:2000,500,replace=false)
tuning_seeds = []
for pseed in rngs
    spikes,_,_,_,_,_,_,_,poisson_in,_,_,_ = run_cell(sim_length,pseed)
    (_,pattern,_,_,_) = poisson_in
    #timestamps
    timestamps = [(pat_time*sim_δt,round(pattern[index+1][2]*sim_δt,digits=1)) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
    push!(timestamps,(timestamps[end][2],sim_length))
    #output spikes
    pss = zeros(Int(sim_length/sim_δt))
    for i = 1:Nbins
        k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
        pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
    end
    last_start = []
    last_end = []
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last green pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 1 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last blue pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 2 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    for (id, (t0, t1)) in enumerate(reverse(timestamps)) # last red pattern
        (pat,_) = reverse(pattern)[id]
        t0 = Int(round(t0/sim_δt,digits=1))
        t1 = Int(round(t1/sim_δt,digits=1))
        if pat == 3 && (t1 - t0) == 1000
            push!(last_start,t0)
            push!(last_end,t1)
            break
        end
    end
    #looking at the spike output of the last pattern presentation
    auxg = mean(pss[last_start[1]:last_end[1]])
    auxb = mean(pss[last_start[2]:last_end[2]])
    auxr = mean(pss[last_start[3]:last_end[3]])

    thresh = 2.5 # much greater than 3 spikes as safety threshold
    comps = findall([auxg,auxb,auxr] .> thresh)
    if length(comps) == 1 # only one tuning
        push!(tuning_seeds,pseed)
    end
end

io = open("seeds_c500_p50_60k.txt", "w") do io
  for x in tuning_seeds
    println(io, x)
  end
end

# 2) n_pat=500, t_pat=50, only_excitatory=false, overlapping=true
# 3) n_pat=100, t_pat=100, 2x freq, only_excitatory=true, overlapping=false
