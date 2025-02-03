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

#simulation
include("lif_neur.jl") #spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
spikes, ns, v_dend, v_soma, g, isis, cvs, stuff, poisson_in, w, cts = run_spk(sim_length,1774)#, w, κ[end])
(ge, gi) = g
(gain_mod,spk_train) = stuff
(presyns,inputs,pattern,_) = poisson_in
(w_init, w, P) = w
(κ, ϐ, μ) = cts

#timestamps
timestamps = [(pat_time*sim_δt*ms_to_sec,round(pattern[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
#output spikes
spikes = spikes * ms_to_sec
#output spikes
pss = bin_spikes(spikes*sec_to_ms,Nbins,box_width,sim_steps,sim_δt)
#input spikes
vals = []
y = []
for ci = 1:n_in
    times = view(inputs,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)
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

#single neuron response (m.p.)
sx = 1
ex = sim_steps
window = (0.0,20.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="Dendritic",
    sharex=true,layout=l,xlim=window, legend=:topleft,
    left_margin = 7Plots.mm, right_margin=7Plots.mm)
plot!(h3[1],t,v_soma,color="black",label="Somatic";dpi=dpi)
bar!(h3[2],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss[sx:ex])),bottom_margin = 7Plots.mm, legend=false, yminorgrid=true)
vspans(h3,timestamps,pattern,(.0,20.))

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
