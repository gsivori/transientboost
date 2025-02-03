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


default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray),
    framestyle = nothing, yminorgrid = true, xminorgrid = true, size=(1800,1200), dpi=150)

## The following is just for testing one trial

#plotting parameters
in_freq = 10.0
n_burst = 1
Δt = -100 #in steps
τCa = 100.0
τμCa = 20.0
sim_δt = 0.1
reps = 100
mbp = 60.0
Tperiod = Int(round(10000*(in_freq^-1),digits=0))
Tno_act = 10000
sim_length = (1050+reps*Tperiod+Tno_act)*sim_δt #we make sure there are reps repetitions
t = collect(range(0.0, stop = sim_length, length=Int(sim_length / sim_δt))) ./ 1000.0
Istr = 10000.0 # Equivalent to 10nA

#one-test
include("ca_stdp.jl")
spikes,ns,v_dend,v_soma,g,gain_mod,we,input_cell,I_ext = assess_cell(reps,Istr,Δt,n_burst,mbp,in_freq,τμCa,τCa,10)
(ge,gi,g_cs,Ics,Il,Ica,Ca,μCa) = g
(w_init,w,P,rel_chg, κ) = we
times = [float(each*index*sim_δt) for (index,each) in enumerate(input_cell[:]) if each != 0];
test3 = plot(t,v_soma,size=(1200,600),xlim=(0.0,.5))
plot!(test3,t,v_dend,size=(1200,600),xlim=(0.0,7.0))
test3 = plot(t,Ics,size=(1200,600),xlim=(0.0,0.5))
test4 = plot(t,g_cs,size=(1200,600),xlim=(0.0,0.5))
test5 = plot(t,gain_mod,size=(1200,600),xlim=(0.0,7.0))

vals = [float(spk/1000.0) for spk in spikes if spk != 0]
in_vals = [float(tind*sim_δt/1000.0) for (tind,spk) in enumerate(input_cell) if spk != 0]
test6 = plot(size=(600,300))
scatter!(test6,vals,ones(length(vals)),markershape=:vline,markersize=15.0,label="Spike out")
scatter!(test6,in_vals,ones(length(in_vals)),markershape=:vline,markersize=15.0,label="Presyn.")
plot!(test6,xlim=(0.6,0.8))

## Some plots for analyzing F-I curve

#plotting parameters
in_freq = 5.0
n_burst = 1
Δt = 0 #in steps
mbp = 60.0
τCa = 100.0
τμCa = 20.0
sim_δt = 0.1
reps = 30
Tperiod = Int(round(10000*(in_freq^-1),digits=0))
Tno_act = 10000
sim_length = (1050+reps*Tperiod+Tno_act)*sim_δt #we make sure there are reps repetitions
t = collect(range(0.0, stop = sim_length, length=Int(sim_length / sim_δt))) ./ 1000.0
howmany= 1000
Istr = collect(range(3600.0,stop=4032.0,length=howmany))
include("ca_stdp.jl")
spikes_vect = []
for icurr in Istr
    spikes,ns,v_dend,v_soma,g,stuff,we,input_cell,I_ext = assess_cell(reps,icurr,Δt,n_burst,mbp,in_freq,τμCa,τCa,10)
    push!(spikes_vect,spikes)
end
while(length(spikes_vect)!=1000)
    push!(spikes_vect,[])
end
fsdata = deepcopy(spikes_vect)
fs = zeros(howmany)
for ii in 1:howmany
    if isempty(spikes_vect[ii])
        fs[ii] = 0.0
    else
        count = length(findall(x -> 38999.0 .< x .< 48999.0,spikes_vect[ii]))
        fs[ii] = count/10.0
    end
end

using JLD2
@save "spikes_vect.jld2" fsdata Istr spikes_vect

@load "spikes_vect.jld2" fsdata Istr



l = @layout [a b c d]
test = plot(size=(1200,600),layout=l,xlim=(1.9,2.1),xticks=[1.9,2.0,2.1])
plot!(test[1],t, v_soma,linewidth=2.0, color=:purple,label=nothing)
plot!(test[2],t, Ics,linewidth=2.0, color=:midnightblue,label=nothing)
plot!(test[3],t, v_dend,linewidth=2.0, color=:black,label=nothing)
plot!(test[4],t, I_ext, color=:red,label=nothing)

test2 = plot(t,I_ext./180.0,size=(1200,600),xlim=(0.0,.5))

test3 = plot(t,v_soma,size=(1200,600))



plot(t,Ca,linewidth=2.0,color=:red,label="Ca",alpha=0.4)
plot!(t,μCa,linewidth=2.0,color=:blue,markerstyle=:dot,label="μCa",alpha=0.4)
plot!(t,gain_mod,linewidth=2.0,color=:purple,markerstyle=:dash,label="P.I.")
plot!(xlim=(49900.0,50100.0))



vlns = (circshift(sign.(gain_mod),1) .- sign.(gain_mod)) .!= 0
vlns = vlns .& (Ca .> 2e-5)
vlns = [float(each*index*sim_δt) for (index,each) in enumerate(vlns[:]) if each != 0.0]
vline!(vlns,linewidth=0.5,color=:brown,label="P.I. = 0.0")
plot!(xlim=(19800.0,21200.0))
plot!(xlim=(29800.0,31200.0))
plot!(xlim=(39800.0,41200.0))
plot!(xlim=(2000.0,2900.0),ylim=(-0.0001,0.0002))

#input spikes
y = [(32.0) for te in times];
id = findall((times .- spikes[2]) .≈ 0.0);
print(times[id])

#STDP-curve (1 post-)
include("ca_stdp.jl")
n_burst = 1
syn_chg = []
delta_ts = []
cell_times = []
for ti in -1000:10:1000
    spikes,_,_,_,_,_,we,input_cell = assess_cell(sim_length,40000.0,ti,n_burst,60.0)
    (w_init,w,P,rel_chg) = we
    times = [float(each*index*sim_δt) for (index,each) in enumerate(input_cell[:]) if each != 0];
    push!(syn_chg,rel_chg)
    push!(delta_ts,ti*sim_δt)
    push!(cell_times,times)
end

#STDP-curve (burst-spikes)
include("ca_stdp.jl")
exps = []
spks = []
spikes = []
syn_chg = []
delta_ts = []
cell_times = []
for n_burst in 1:4
    syn_chg = []
    delta_ts = []
    for ti in -1000:10:1000
        spikes,_,_,_,_,_,we,input_cell = assess_cell(sim_length,40000.0,ti,n_burst,60.0)
        (w_init,w,P,rel_chg) = we
        times = [float(each*index*sim_δt) for (index,each) in enumerate(input_cell[:]) if each != 0];
        push!(syn_chg,rel_chg)
        push!(delta_ts,ti*sim_δt)
        push!(cell_times,times)
    end
    push!(spks, spikes)
    push!(exps,(syn_chg,delta_ts,cell_times))
end


#STDP-curve
gr(markersize=0.0,markershape=:auto,markerstrokewidth=0.0,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
h0 = plot(title="STDP curve",ylabel="synaptic plasticity change (a.u.)",xlabel="time difference Δt (ms)",gridalpha=0.1)
scatter!(delta_ts,syn_chg,markersize=5.0,markershape=:o,markercolor=:black,markerbackgroundcolor=:white,markerstrokewidth=1.0,markeralpha=1.0,label=nothing)

#STDP-curve w/ burst-spikes
gr(markersize=0.0,markershape=:auto,markerstrokewidth=0.0,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
h0 = plot(title="STDP curve",ylabel="synaptic plasticity change (a.u.)",xlabel="time difference Δt (ms)",gridalpha=0.1)
for n = 1:4
    scatter!(exps[n][2],exps[n][1],markersize=4.0,markerstrokewidth=0.25,markeralpha=1.0,label=n,markercolor=:auto,color=n)
end
scatter!(exps[1][2],exps[1][1],markersize=4.0,markerstrokewidth=0.25,color=1,markeralpha=1.0,
label=nothing,subplot=2,bg_inside=:white,inset=(1, bbox(0.05, 0.05, 0.45, 0.45, :top, :left)),
xlabel="time difference Δt (ms)")


#output spikes
pss = zeros(Int(sim_length/sim_δt))
for i = 1:Nbins
    k[i] = count(spk->((i-1)*box_len <= spk <= (i+1)*box_len),spikes)
    pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
end
#input spikes
times = [float(each*index*sim_δt) for (index,each) in enumerate(input_cell[:]) if each != 0]
y = [(32.0) for te in times]

#single neuron response (m.p.)
gr(markersize=0.0,markershape=:auto,markerstrokewidth=0.0,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
l = @layout [a; b{0.1h}]
h3 = plot(t,v_dend,color="purple",ylabel="m.p. (mV)",label="Dendritic",sharex=true,legend=:bottomleft,layout=l)
plot!(h3[1],t,v_soma,color="black",label="Somatic",xlim=(0,sim_length))
scatter!(h3[1],times,y,label=nothing,markercolor=:blue,markersize=1.0,markeralpha=1.0, markerstrokewidth=1.0,markershape=:circle)
#plot!(h3[2],t,gain_mod,color="blue", xlabel="Time (ms)",ylabel="syn.pl. (a.d)",legend=false,markeralpha=0.0)
bar!(h3[2],t,pss,xlabel="time (ms)",ylabel="# spikes",ylim=(0,1), legend=false, yminorgrid=true, xlim=(0,sim_length),yticks=[0,1])

#using DataFrames,CSV
#test_means = reshape(mean(crosscor(pss_chk[:,1:n_tests-1],pss_chk[:,n_tests]),dims=1),(20,6))
#df = DataFrame(test_means,:auto)
#rename!(df,["5", "10", "15","25","50","100"])
#CSV.write("boxplot_25_74568.csv",df)


gr(markersize=0.0, markerstrokewidth=0.0,markeralpha=0.0)
h6 = plot(t,Ics,color="brown",xlabel="Time (ms)",legend=true, alpha=0.75,label="Ics",ylabel="I (pA)",linewidth=2.0)
plot!(t,Il,color="purple",xlabel="Time (ms)",legend=true, alpha=0.75,label="Il",linewidth=2.0,
    left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, right_margin=5Plots.mm)

#Ca trace
gr(markersize=0.0, markerstrokewidth=0.0)
h7a = plot(t,Cas[15],color="purple", xlabel="Time (ms)",legend=false)

#different Ca traces
gr(markersize=0.0, markerstrokewidth=0.0)
h7a = plot(xlabel="Time (ms)")
for n in 1:20
    plot!(t,Cas[300*n],label=n)
end
plot!()

#Ica trace
gr(markersize=0.0, markerstrokewidth=0.0)
h7b = plot(t,Ica,color="purple", xlabel="Time (ms)",legend=false)

#g_cs activity change over time
gr(markersize=0.0,markeralpha=0.0)
h12 = plot(t,g_cs,color="purple",xlabel="Time (ms)",ylabel="Coupling conductance (nS)",legend=false,alpha=0.8)


#STDP-curve (burst-spikes) for different mbp
include("ca_stdp.jl")
exps = []
spks = []
spikes = []
syn_chg = []
delta_ts = []
cell_times = []
mbps = []
for mbp in 25.0:5.0:70.0
    push!(mbps,mbp)
    for n_burst in 1:4
        syn_chg = []
        delta_ts = []
        for ti in -1000:50:1000
            spikes,_,_,_,_,_,we,input_cell = assess_cell(sim_length,40000.0,ti,n_burst,mbp)
            (w_init,w,P,rel_chg) = we
            times = [float(each*index*sim_δt) for (index,each) in enumerate(input_cell[:]) if each != 0];
            push!(syn_chg,rel_chg)
            push!(delta_ts,ti*sim_δt)
            push!(cell_times,times)
        end
        push!(spks, spikes)
        push!(exps,(syn_chg,delta_ts,cell_times))
    end
end
all_exps = deepcopy(exps)
exps = all_exps[37:40]


#STDP-curve w/ burst-spikes
gr(markersize=0.0,markershape=:auto,markerstrokewidth=0.0,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
h0 = plot(title="STDP curve",ylabel="synaptic plasticity change (a.u.)",xlabel="time difference Δt (ms)",gridalpha=0.1)
for n = 1:4
    scatter!(exps[n][2],exps[n][1],markersize=4.0,markerstrokewidth=0.25,markeralpha=1.0,label=n,markercolor=:auto,color=n)
end
scatter!(exps[1][2],exps[1][1],markersize=4.0,markerstrokewidth=0.25,color=1,markeralpha=1.0,
label=nothing,subplot=2,bg_inside=:white,inset=(1, bbox(0.05, 0.05, 0.45, 0.45, :top, :left)),
xlabel="time difference Δt (ms)")


include("stdp.jl")
rel_chgs = []
gain_mods = []
gcss = []
mus = []
Ys = []
for ti in -1000:200:1000
    spikes,_,_,_,stuff,gain_mod,we,input_cell,_ = assess_cell(sim_length,40000.0,ti,1)
    (_, _, Y, μ) = stuff
    (w_init,w,P,rel_chg) = we
    push!(rel_chgs,rel_chg)
    push!(gain_mods,P)
    push!(mus,μ)
    push!(Ys,Y)
end
print(rel_chgs)

plot(t,gain_mods[:],labels=loslabels,legend=:topleft,linewidth=2.0)

spikes,_,_,_,stuff,gain_mod,we,input_cell,_ = assess_cell(sim_length,40000.0,-100,1)
(_, _, Y, μ) = stuff
(w_init,w,P,rel_chg) = we
