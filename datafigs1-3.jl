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
using LaTeXStrings

#plotting parameters
ms_to_sec = 1e-3
sec_to_ms = ms_to_sec^-1
n_in = 2000
tfactor = 2.
sim_length = 10000.0 * tfactor
sim_δt = 0.1
sim_steps = Int(sim_length/sim_δt) #in steps
ticks = 1000.0 * tfactor
bins = 25
dpi = 300
box_width = 500
box_len = box_width*sim_δt
figsize = (900,600)
Nbins = Int(round(sim_length/sim_δt/box_width,digits=2))
t = collect(range(0.0, stop = sim_length, length=Int(100000 * tfactor))) ./ 1000.0
default(legendfontsize = 12, guidefont = (16, :black), guide="", tickfont = (12, :gray), framestyle = nothing, yminorgrid = true, xminorgrid = true, size=figsize, dpi=300)

seed = 2022

#helper functions
include("funs.jl")
#simulation bpNMDAfull
include("bp_full.jl")
bpn_spikes,bpn_ns,bpn_v_dend,bpn_v_soma,bpn_g,_,_,bpn_gain_mod,poisson_in,bpn_w,bpn_cts = run_bpfull(sim_length,seed)
(bpn_ge,bpn_gi,bpn_gcs,bpn_gnmda,bpn_Ica,bpn_Ca,bpn_Inmda,bpn_Icsd) = bpn_g 
(bpn_w_init,bpn_w,bpn_P) = bpn_w
(presyns,postsyns,pattern,pats) = poisson_in
(bpn_κ, bpn_ϐ) = bpn_cts
#output spikes
bpn_pss = bin_spikes(bpn_spikes,Nbins,box_width,sim_steps,sim_δt)
bpn_spikes = bpn_spikes .* ms_to_sec
#simulation bpHVAneur
include("bpHVAneur.jl")
bp_spikes,bp_ns,bp_v_dend,bp_v_soma,bp_g,_,_,bp_gain_mod,poisson_in,bp_w,bp_cts,chks = run_bp(sim_length,seed)
(bp_ge,bp_gi,bp_gcs,bp_Icsd,bp_Ca) = bp_g
(bp_w_init,bp_w,bp_P) = bp_w
(presyns,postsyns,pattern,pats) = poisson_in
(bp_κ, bp_ϐ) = bp_cts
#output spikes
bp_pss = bin_spikes(bp_spikes,Nbins,box_width,sim_steps,sim_δt)
bp_spikes = bp_spikes .* ms_to_sec
#simulation lifneur
include("lif_neur.jl")
spikes,ns,v_dend,v_soma,g,isis,cvs,stuff,poisson_in,w,cts = run_spk(sim_length,seed)
(ge, gi) = g
(gain_mod,spk_train) = stuff
(presyns,postsyns,pattern,pats) = poisson_in
(w_init, w, P) = w
(κ, ϐ, μ) = cts
#output spikes
pss = bin_spikes(spikes,Nbins,box_width,sim_steps,sim_δt)
spikes = spikes .* ms_to_sec
#timestamps
timestamps = [(pat_time*sim_δt/1000.0,round(pattern[index+1][2]*sim_δt,digits=1)/1000.0) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
push!(timestamps,(timestamps[end][2],sim_length));
print("completed!")
print(length.([bpn_spikes,bp_spikes,spikes]))

## below are test and random figures -- fig1.jl, fig2.jl, fig3.jl use the above.

#single neuron response (m.p.)
window=(0.0,20.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,bpn_v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="",
    sharex=true,layout=l,xlim=window, legend=:bottomright,
    left_margin = 7Plots.mm, right_margin=7Plots.mm,size=(800,600))
vspans(h3,timestamps,pattern,window,0.2)
plot!(h3[1],t,bpn_v_soma,color="black",label="")
bar!(h3[2],t,bpn_pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(bpn_pss)),
    bottom_margin = 7Plots.mm, legend=false, yminorgrid=true)

savefig(h3,"data_figs/full_based.pdf")

#single neuron response (m.p.)
window=(0.0,20.0)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h3 = plot(t,bp_v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="",
    sharex=true,layout=l,xlim=window, legend=:bottomright,
    left_margin = 7Plots.mm, right_margin=7Plots.mm,size=(800,600))
plot!(h3[1],t,bp_v_soma,color="black",label="")
bar!(h3[2],t,bp_pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(bp_pss)),
    bottom_margin = 7Plots.mm, legend=false, yminorgrid=true)
vspans(h3,timestamps,pattern,window,0.2)
savefig(h3,"data_figs/ca_based.pdf")

#single neuron response (m.p.)
window=(0.0,20.0)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
l = @layout [a; c{0.2h}]
h4 = plot(t,v_dend,color="purple",xlabel="", ylabel="M.p. (mV)",label="",
    sharex=true,layout=l,xlim=window, legend=:bottomright,
    left_margin = 7Plots.mm, right_margin=7Plots.mm,size=(800,600))
plot!(h4[1],t,v_soma,color="black",label="")
bar!(h4[2],t,pss,xlabel="Time (s)",ylabel="Spk. count",xlim=window,ylim=(0,maximum(pss)),
    bottom_margin = 7Plots.mm, legend=false, yminorgrid=true)
vspans(h4,timestamps,pattern,window,0.2)
savefig(h4,"data_figs/spike_based.pdf")

#single neuron response (m.p.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
h32 = plot(t,bp_Icsd,color="purple",xlabel="", ylabel="Current (pA)",label="Somato-dendritic coupling",
    sharex=true,xlim=window, legend=:bottomright,linewidth=1.5,
    left_margin = 7Plots.mm, right_margin=7Plots.mm)
vspans(h32,timestamps,pattern,window,0.2)

#gain trace w pattern bars
gr(markersize=0.0, markerstrokewidth=0.0)
h5 = plot(t,gain_mod,color="blue", xlabel="Time (ms)",linewidth=1.5,
    legend=false, dpi=dpi,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
plot!(t,bp_gain_mod,color="purple", xlabel="Time (ms)",linewidth=1.5,
    legend=false, dpi=dpi,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
    plot!(t,bpn_gain_mod,color="black", xlabel="Time (ms)",linewidth=1.5,
    legend=false, dpi=dpi,markeralpha=0.0,left_margin=7Plots.mm, bottom_margin=5Plots.mm,right_margin=5Plots.mm)
vspans(h5, timestamps,pattern,window,0.2)


#syns_comp is the presynaptic vs postsynaptic ID comparison
#input spikes
window = (4.8,6.3)
windowticks = [5.0,6.0]
gr(markersize=5.0,markershape=:vline,legend=false, markerstrokewidth=2.5, markeralpha=1.0)
l1a = @layout [a b; c d]
syns_comp = plot(size=(800,600),yminorgrid=false,xminorgrid=false,ytickfont=font(12),xtickfont=font(12),guidefont=font(12),tickfont = (12, :black),yguidefontsize=12,xguidefontsize=12,legend=false,left_margin=2Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,sharex=true,dpi=300,layout=l1a)
vals = []
y = []
for ci = 1:n_in
    times = view(presyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)
#input spikes pss
pss_in = zeros(Int(sim_length/sim_δt))
pss_ci = zeros(n_in,Int(sim_length/sim_δt))
for ci = 1:n_in
    pss_ci[ci,:] = bin_spikes(view(vals,ci,:)[1] .* sec_to_ms,Nbins,box_width,sim_steps,sim_δt)
end
pss_in = sum(pss_ci,dims=1)[1,:]
scatter!(syns_comp[1],xs,ys,group=grouping,markercolor=[:black :blue :red :green],ylabel="input ID",ylim=(595,655),xlim=window,xticks=windowticks,yticks=[600,625,650],xlabel="",title=L"p_{fail}=0.0")
plot!(syns_comp[3],t,pss_in,xlabel="Time (s)",ylabel="Σ input IDs",xlim=window,xticks=windowticks,ylim=(0,325),yticks=[0,150,300],legend=false,color=:black,linewidth=1.0,markersize=0.0, markerstrokewidth=0.)
vals = []
y = []
for ci = 1:n_in
    times = view(postsyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times./1000.0)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)
#input spikes pss
pss_in = zeros(Int(sim_length/sim_δt))
pss_ci = zeros(n_in,Int(sim_length/sim_δt))
for ci = 1:n_in
    pss_ci[ci,:] = bin_spikes(view(vals,ci,:)[1] .* sec_to_ms,Nbins,box_width,sim_steps,sim_δt)
end
pss_in = sum(pss_ci,dims=1)[1,:]
scatter!(syns_comp[2],xs,ys,group=grouping,markercolor=[:black :blue :red :green], xlabel="Time (s)",ylabel="",ylim=(595,655),xlim=window,xticks=windowticks,yticks=[600,625,650],
title=L"p_{fail}=0.3")
plot!(syns_comp[4],t,pss_in,xlabel="Time (s)",ylabel="",xlim=window,xticks=windowticks,ylim=(0,325),yticks=[0,150,300],color=:black,linewidth=1.0,markersize=0.0, markerstrokewidth=0.)
vspans(syns_comp,1:4,timestamps,pattern,window,0.2)
plot!(xlabel="Time (s)")
savefig("wbars.png")
#to save
for i=1:2; plot!(syns_comp[i],xformatter=_->""); end;
    plot!()


savefig(syns_comp,"data_figs/syns_comp.pdf")
#plot!(syns_comp[1],title = "B", titleloc = :left, titlefont = font(18))
savefig(syns_comp,"data_figs/syns_comp.png")

using LaTeXStrings
#Double-exp-kernel
gr(markersize=0.0,legend=false, markerstrokewidth=0.0, markeralpha=0.0)
τe_rise = 0.5 # rise time constant for exc synapses (ms)
τi_rise = 1.0 # rise time constant for inh synapses (ms)
τe_decay = 3.0 # decay time constant for exc synapses (ms)
τi_decay = 8.0 # decay time constant for inh synapses (ms)
τcsd_rise = 0.2 # rise time constant for coupled syn spike (ms)
τcsd_decay = 1.5 # decay time constant for coupled syn spike (ms)
tpeak_e = τe_decay*τe_rise/(τe_decay-τe_rise)*log(τe_decay/τe_rise)
tpeak_i = τi_decay*τi_rise/(τi_decay-τi_rise)*log(τi_decay/τi_rise)
tpeak_bp = τcsd_decay*τcsd_rise/(τcsd_decay-τcsd_rise)*log(τcsd_decay/τcsd_rise)
Ne = ((exp(-tpeak_e/τe_decay)-exp(-tpeak_e/τe_rise)))^-1
Ni = ((exp(-tpeak_i/τi_decay)-exp(-tpeak_i/τi_rise)))^-1
Ncsd = ((exp(-tpeak_bp/τcsd_decay)-exp(-tpeak_bp/τcsd_rise)))^-1
Ne = ((exp(-tpeak_e/τe_decay)-exp(-tpeak_e/τe_rise)))^-1
sim_δt = 0.1 # simulation time step (ms)
g_E_prev = 1.0
DEK = []
xe_rise = 0.0
xe_decay = 0.0
g_e = 0.5
x = collect(0.0:0.1:10.0)
for i in x
    xe_rise += -sim_δt*xe_rise/τe_rise + g_E_prev
    xe_decay += -sim_δt*xe_decay/τe_decay + g_E_prev
    gef = g_e*Ne*(xe_decay - xe_rise)
    g_E_prev = 0.0
    push!(DEK,gef)
end
dek = plot(x,DEK,xlim=(0,10.0),linewidth=2.0,color=:blue,alpha=0.7,xticks=[0.0,2.5,5.0,7.5, 10.0],yticks=[],label=L"g_e",
xlabel="Time (ms)",size=(500,250),yminorgrid=true,xminorgrid=true,yguidefontsize=12,tickfont=(12,:gray),ylabel="Conductance (nS)",
xguidefontsize=10,left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,dpi=300)
g_I_prev = 1.0
DEK = []
xi_rise = 0.0
xi_decay = 0.0
g_i = 0.5
x = collect(0.0:0.1:10.0)
for i in x
    xi_rise += -sim_δt*xi_rise/τi_rise + g_I_prev
    xi_decay += -sim_δt*xi_decay/τi_decay + g_I_prev
    gef = g_i*Ni*(xi_decay - xi_rise)
    g_I_prev = 0.0
    push!(DEK,gef)
end
plot!(x,DEK,xlim=(0,10.0),linewidth=2.0,color=:red,alpha=0.7,xticks=[0.0,2.5,5.0,7.5, 10.0],yticks=[],label=L"g_i",
xlabel="Time (ms)",size=(500,250),yminorgrid=true,xminorgrid=true,yguidefontsize=12,tickfont=(12,:gray),
xguidefontsize=10,left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,dpi=300)
g_C_prev = 1.0
DEK = []
xcsd_rise = 0.0
xcsd_decay = 0.0
g_csd = 0.5
x = collect(0.0:0.1:10.0)
for i in x
    xcsd_rise += -sim_δt*xcsd_rise/τcsd_rise + g_C_prev
    xcsd_decay += -sim_δt*xcsd_decay/τcsd_decay + g_C_prev
    gef = g_csd*Ncsd*(xcsd_decay - xcsd_rise)
    g_C_prev = 0.0
    push!(DEK,gef)
end
plot!(x,DEK,xlim=(0,10.0),linewidth=2.0,color=:purple,alpha=0.7,xticks=[0.0,2.5,5.0,7.5, 10.0],yticks=[],label=L"g_{csd}",
xlabel="Time (ms)",size=(500,250),yminorgrid=true,xminorgrid=true,yguidefontsize=12,tickfont=(12,:gray),
xguidefontsize=10,left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,dpi=300)
plot!(legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=16)
savefig(dek,"data_figs/dek.pdf")

starting_w = rand(Normal(),n_in)
start_hist = plot(xlim=(-4.0,4.0),linewidth=2.0,color=:brown,alpha=0.7,xticks=[-3,-2,-1,0,1,2,3],yticks=[0,200,400],ylabel="Synapses",
xlabel="Synaptic strength (a.u.)",size=(500,250),yminorgrid=true,xminorgrid=true,yguidefontsize=12,ylim=(0,400),
xguidefontsize=10,left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,dpi=300,tickfont=(12,:gray))
histogram!(start_hist,starting_w,bins=:scott,color=:pink,alpha=0.75,legendfontsize=16)
savefig(start_hist,"data_figs/start_hist.pdf")




#single neuron response (m.p.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)

pat = [j for (i,j) in pattern if i==2]
pat *= sim_δt
pat_time = 100.0
limits = (pat[end]-50.0,pat[end]+pat_time+50.0) ./ 1000.0

l = @layout [a b c]
window = limits
windowticks = [18.8,18.9] #pattern-related
firstwindow = (0.5,2.5)
firstwindowticks = [0.5,1.5,2.5]
lastwindow = (17.0, 19.0)
lastwindowticks = [17.0,18.0,19.0]
firstwindowrange = 5000:25000
lastwindowrange = 170000:190000

model_comp = plot(size=(750,400),dpi=900,
    ytickfont=font(12),xtickfont=font(12),guidefont=font(12),legendfontsize=16,sharex=true,
    yguidefontsize=12,xguidefontsize=12,left_margin=2Plots.mm,right_margin=1Plots.mm,bottom_margin=4Plots.mm,layout=l)
vspans(model_comp,timestamps,pattern,firstwindow,0.25)
vspans(model_comp,timestamps,pattern,lastwindow,0.25)
vspans(model_comp,timestamps,pattern,window,0.25)
plot!(model_comp[3],t[lastwindowrange],v_dend[lastwindowrange],color="gray",xlabel="Time (s)", ylabel="M.p. (mV)",alpha=1.0,label=L"v_d",yminorgrid=false,linewidth=1.5)
plot!(model_comp[3],t[lastwindowrange],v_soma[lastwindowrange],color="purple",label=L"v_s",alpha=1.0,xlim=window,xticks=windowticks,linewidth=1.5,yticks=[-10,-30,-50,-70],ylim=(-75,-5),xminorgrid=false)
bar!(model_comp[1],t[firstwindowrange],pss[firstwindowrange],ylabel="Spike count",ylim=(0,maximum(pss)),legend=false, yminorgrid=false,
    ytickfont=font(12),xlabel="Time (s)",
    xtickfont=font(12),
    guidefont=font(12),yguidefontsize=12,xguidefontsize=12,xminorgrid=false,
    legendfontsize=12,label=nothing,fillalpha=0.1,xlim=firstwindow,xticks=firstwindowticks)
bar!(model_comp[2],t[lastwindowrange],pss[lastwindowrange],ylim=(0,maximum(pss)),legend=false, yminorgrid=false,
    ytickfont=font(12),xlabel="Time (s)",
    xtickfont=font(12),
    guidefont=font(12),yguidefontsize=12,xguidefontsize=12,xminorgrid=false,
    legendfontsize=12,label=nothing,fillalpha=0.1,xlim=lastwindow,xticks=lastwindowticks,sharey=true,ylabel="")
plot!(model_comp[2],yformatter=_->"");
plot!(legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent)
plot!(model_comp[1],title = "E", titleloc = :left, titlefont = font(16))
savefig(model_comp,"data_figs/spike_bef_aft.png")

l = @layout [a b c]
model_comp = plot(size=(600,200),
    ytickfont=font(12),xtickfont=font(12),guidefont=font(12),legendfontsize=12,sharex=true,
    yguidefontsize=12,xguidefontsize=12,left_margin=2Plots.mm,right_margin=1Plots.mm,bottom_margin=4Plots.mm,layout=l)
vspans(model_comp,1:1,timestamps,pattern,firstwindow,0.25)
vspans(model_comp,2:2,timestamps,pattern,lastwindow,0.25)
vspans(model_comp,3:3,timestamps,pattern,window,0.25)
plot!(model_comp[3],t[lastwindowrange],bp_v_dend[lastwindowrange],color="gray",xlabel="Time (s)", ylabel="M.p. (mV)",alpha=1.0,label="Dendritic",yminorgrid=false,linewidth=1.5)
plot!(model_comp[3],t[lastwindowrange],bp_v_soma[lastwindowrange],color="purple",label="Somatic",alpha=1.0,xlim=window,xticks=windowticks,linewidth=1.5)
bar!(model_comp[1],t[firstwindowrange],bp_pss[firstwindowrange],ylabel="Spike count",ylim=(0,maximum(bp_pss)),legend=false, yminorgrid=false,
    ytickfont=font(12),xlabel="Time (s)",
    xtickfont=font(12),
    guidefont=font(12),yguidefontsize=12,xguidefontsize=12,
    legendfontsize=12,label=nothing,fillalpha=0.1,xlim=firstwindow,xticks=firstwindowticks)
bar!(model_comp[2],t[lastwindowrange],bp_pss[lastwindowrange],ylim=(0,maximum(bp_pss)),legend=false, yminorgrid=false,
    ytickfont=font(12),xlabel="Time (s)",
    xtickfont=font(12),
    guidefont=font(12),yguidefontsize=12,xguidefontsize=12,
    legendfontsize=12,label=nothing,fillalpha=0.1,xlim=lastwindow,xticks=lastwindowticks,sharey=true,ylabel="")
savefig(model_comp,"data_figs/ca-based_bef_aft.pdf")

l = @layout [a b c]
model_comp = plot(size=(600,200),
    ytickfont=font(12),xtickfont=font(12),guidefont=font(12),legendfontsize=12,sharex=true,
    yguidefontsize=12,xguidefontsize=12,left_margin=2Plots.mm,right_margin=1Plots.mm,bottom_margin=4Plots.mm,layout=l)
vspans(model_comp,timestamps,pattern,firstwindow,0.25)
vspans(model_comp,timestamps,pattern,lastwindow,0.25)
vspans(model_comp,timestamps,pattern,window,0.25)
plot!(model_comp[3],t[lastwindowrange],bpn_v_dend[lastwindowrange],color="gray",xlabel="Time (s)", ylabel="M.p. (mV)",alpha=1.0,label="Dendritic",yminorgrid=false,linewidth=1.5)
plot!(model_comp[3],t[lastwindowrange],bpn_v_soma[lastwindowrange],color="purple",label="Somatic",alpha=1.0,xlim=window,xticks=windowticks,linewidth=1.5)
bar!(model_comp[1],t[firstwindowrange],bpn_pss[firstwindowrange],ylabel="Spike count",ylim=(0,maximum(bp_pss)),legend=false, yminorgrid=false,
        ytickfont=font(12),xlabel="Time (s)",
        xtickfont=font(12),
        guidefont=font(12),yguidefontsize=12,xguidefontsize=12,
        legendfontsize=12,label=nothing,fillalpha=0.1,xlim=firstwindow,xticks=firstwindowticks)
    bar!(model_comp[2],t[lastwindowrange],bpn_pss[lastwindowrange],ylim=(0,maximum(bpn_pss)),legend=false, yminorgrid=false,
        ytickfont=font(12),xlabel="Time (s)",
        xtickfont=font(12),
        guidefont=font(12),yguidefontsize=12,xguidefontsize=12,
        legendfontsize=12,label=nothing,fillalpha=0.1,xlim=lastwindow,xticks=lastwindowticks,sharey=true,ylabel="")
plot!(legend=false)
savefig(model_comp,"data_figs/full_bef_aft.png")

zetafig = plot(ytickfont=font(12),dpi=300,minorgrid=true,grid=true,
guidefont=font(12),yguidefontsize=16,xtickfont=font(12),left_margin = 2Plots.mm,legend=:topleft,
right_margin = 3Plots.mm, bottom_margin = 4Plots.mm,legendfontsize=12,size=(500,200),
xlabel="Syn. weight magnitude (a.u.)",ylabel=L"$\zeta \;\left(\| w_{i}\|\right)}$")
gr(markersize=0.0, markerstrokewidth=0.0)
syn_w = collect(Float64, 0.0:0.01:8.0)
alpha_fun = ζ.(syn_w,1e-4,.75)
plot!(syn_w,alpha_fun,color="purple",markeralpha=0.0,legend=false,linewidth=3.0,xlim=(-0.5,8.),xticks=[0,2,4,6,8,10],yticks=[0.0,0.5,1.0])
savefig(zetafig,"data_figs/zeta_func_B.pdf")

bins=20
l = @layout [a{0.5w} c{0.5w}]
hists = plot(ytickfont=font(10),dpi=300,minorgrid=false,grid=false,
guidefont=font(10),xtickfont=font(10),
yguidefontsize=10,legendfontsize=8,
left_margin=1Plots.mm,right_margin=1Plots.mm,bottom_margin=2Plots.mm,top_margin=0.0Plots.mm,
layout=l,size=(600,300))
#histogram!(fig1d[2],w_init[w_init.>0],bins=bins,alpha=1.0, color="black",label=nothing)
histogram!(hists[2],w_init[w_init.>0],bins=10,alpha=0.55, color="black",label="Starting values")
histogram!(hists[2],w[w.>0],bins=bins,alpha=0.55, color="purple",label="Spike trace model",yticks=[0,250,500,750],ylim=(0,750))
histogram!(hists[2],bp_w[bp_w.>0],bins=bins,alpha=0.55, color="blue",label="Calcium trace model",xlim=(0.0,maximum(bp_w)),yticks=[0,250,500,750],ylim=(0,750))
histogram!(hists[2],bpn_w[bpn_w.>0],bins=bins,alpha=0.55, color="green",label="NMDA model",xlim=(0.0,maximum(bp_w)),yticks=[0,250,500,750],ylim=(0,750))
plot!(hists[2],xlabel="Exc. weights",xticks=[0.0,2.5,5.0],xguidefont=13)

#histogram!(fig1d[1],w_init[w_init.<0],bins=bins,alpha=1.0, color="black",label="Initial condition")
histogram!(hists[1],w_init[w_init.<0],bins=10,alpha=0.55, color="black",label="Starting values",ylabel="Synaptic count")
histogram!(hists[1],w[w.<0],bins=bins,alpha=0.55, color="purple",label="Spike trace model",yticks=[0,250,500,750],ylim=(0,750))
histogram!(hists[1],bp_w[bp_w.<0],bins=bins,alpha=0.55, color="blue",label="Calcium trace model",xlim=(-1 .* maximum(bp_w),0.0),yticks=[0,250,500,750],ylim=(0,750))
histogram!(hists[1],bpn_w[bpn_w.<0],bins=bins,alpha=0.55, color="green",label="NMDA model",xlim=(-1 .* maximum(bp_w),0.0),yticks=[0,250,500,750],ylim=(0,750))
plot!(hists[1],xlabel="Inh. weights",xticks=[-5.0,-2.5,0.0],xguidefont=13,yguidefont=13)
plot!(legend=:topright,background_color_legend=:transparent,foreground_color_legend=:transparent)
savefig(hists,"data_figs/hists_fig1sup.pdf")


#how does the model learn figure. (A and B)
vals = []
y = []
for ci = 1:n_in
    times = view(postsyns,ci,:)
    times = [float(each*index*sim_δt) for (index,each) in enumerate(times) if each != 0]
    push!(vals,times.*ms_to_sec)
    push!(y,ci*ones(length(times)))
end
xs, ys, grouping = groupbypat(vals,y,pattern,sim_δt)


l = @layout [a;b;c;d]
id1 = 163
id2 = 621
id1_ids = findall(ys .== id1)
id1_xs = xs[id1_ids]
id1_ys = ys[id1_ids]
id2_ids = findall(ys .== id2)
id2_xs = xs[id2_ids]
id2_ys = ys[id2_ids]
tstart = 58000
tend = 65000
window = (5.8,6.5)
windowticks = [5.8,6.0,6.2,6.4]


howlearning_A = plot(ytickfont=font(10),guidefont=font(10),
yguidefontsize=10,background_color_legend=:transparent,foreground_color_legend=:transparent,sharex=true,
xtickfont=font(10),label=nothing,left_margin=5Plots.mm,legend=false,layout=l,top_margin=0Plots.mm,xminorgrid=false,
right_margin=3Plots.mm,bottom_margin=-3Plots.mm,legendfontsize=14,size=(450,750),xlim=window,xticks=windowticks)
gr(markersize=0.0, markerstrokewidth=0.0,legend=false)
vspans(howlearning_A, timestamps,pattern,window,0.25)
scatter!(twinx(howlearning_A[4]),id1_xs,id1_ys,group=grouping[id1_ids],markeralpha=1.0,markercolor="hotpink", xlabel="",
    markersize=12.0,markerstrokewidth=3,markershape=:vline,ylabel="",yticks=[],legend=false,
    yminorgrid=false,xticks=[],xlim=window,yguidefontsize=10,ytickfont=font(10),yflip=true)
scatter!(twinx(howlearning_A[4]),id2_xs,id2_ys,group=grouping[id2_ids],markeralpha=1.0,markercolor="teal", xlabel="",
    markersize=12.0,markerstrokewidth=3,markershape=:vline,ylabel="",yticks=[],legend=false,
    yminorgrid=false,xticks=[],xlim=window,yguidefontsize=10,ytickfont=font(10),yflip=false)
plot!(howlearning_A[1],t[tstart:tend],v_dend[tstart:tend],color=:gray,xlabel="", ylabel="M.p. (mV)",alpha=1.0,label=L"v_{d}",
    yminorgrid=false,linewidth=1.5,legend=:topright)
plot!(howlearning_A[1],t[tstart:tend],v_soma[tstart:tend],color="purple",label=L"v_{s}",alpha=1.0,linewidth=1.5,yticks=[-10,-30,-50,-70],ylim=(-75,-5),ygrid=true)
gr(markersize=0.0, markerstrokewidth=0.0,legend=false)
plot!(howlearning_A[2],t[tstart:tend],spk_train[tstart:tend],color="blue",legend=:top,linewidth=2.0,ylabel="Spike traces",label=L"Y")
plot!(howlearning_A[2],t[tstart:tend],(spk_train .- gain_mod)[tstart:tend],color="indigo",linewidth=2.0,label=L"\overline{Y}",
    yticks=[.25,.75,1.25,1.75],yminorgrid=true,legend=:topright)
plot!(howlearning_A[3],t[tstart:tend],gain_mod[tstart:tend],color="midnightblue",label="",linewidth=2.5,legend=false,ylabel=L"PI_t")
plot!(howlearning_A[4],t[tstart:tend],P[id1,tstart:tend],color="hotpink",linewidth=2.5,label="ID $id1",
    legendfontsize=12,legend=:right,ylabel="w")
plot!(howlearning_A[4],t[tstart:tend],P[id2,tstart:tend],color="teal",linewidth=2.5,xlabel="Time (s)",label="ID %$id2",
    legendfontsize=12,legend=:right,ylabel="w",xticks=windowticks,yminorgrid=true)
#to save
for i=1:3; plot!(howlearning_A[i],xformatter=_->""); end;
plot!()
savefig(howlearning_A,"data_figs/howlearning_A.pdf")

id1 = 1776
id2 = 1089
id1_ids = findall(ys .== id1)
id1_xs = xs[id1_ids]
id1_ys = ys[id1_ids]
id2_ids = findall(ys .== id2)
id2_xs = xs[id2_ids]
id2_ys = ys[id2_ids]
tstart = 58000
tend = 65000
window = (5.8,6.5)
windowticks = [5.8,6.0,6.2,6.4]


lB = @layout [grid(4,1) grid(4,1)]
howlearning = plot(ytickfont=font(10),guidefont=font(12),background_color_legend=:transparent,
foreground_color_legend=:transparent,sharex=true,markersize=0.0, markerstrokewidth=0.0,grid=false,
tickfont=font(12),label=nothing,left_margin=5Plots.mm,legend=false,layout=lB,top_margin=0Plots.mm,minorgrid=false,
right_margin = 3Plots.mm,bottom_margin=3Plots.mm,legendfontsize=12,size=(1000,750),xlim=window,xticks=windowticks)
vspans(howlearning,1:4, timestamps,pattern,window,0.2)
scatter!(twinx(howlearning[4]),id1_xs,id1_ys,group=grouping[id1_ids],markeralpha=1.0, xlabel="",
    markersize=12.0,markerstrokewidth=2,markershape=:vline,ylabel="",yticks=[],legend=false,
    minorgrid=false,xticks=[],xlim=window,guidefontsize=10,ytickfont=font(12),yflip=false,markercolor="teal")
scatter!(twinx(howlearning[4]),id2_xs,id2_ys,group=grouping[id2_ids],markeralpha=1.0, xlabel="",
    markersize=12.0,markerstrokewidth=2,markershape=:vline,ylabel="",yticks=[],legend=false,
    minorgrid=false,xticks=[],xlim=window,guidefontsize=10,ytickfont=font(12),yflip=true,markercolor="hotpink")
plot!(howlearning[1],t[tstart:tend],bp_v_soma[tstart:tend],color="purple",label="",alpha=1.0,linewidth=1.25)
plot!(howlearning[1],t[tstart:tend],bp_v_dend[tstart:tend],color="black",xlabel="", ylabel="Activity (mV)",alpha=1.0,label="",
    yminorgrid=false,linewidth=1.25,yticks=[20,0,-20,-40,-60],ylim=(-75,40),ygrid=true,legend=:topleft)
plot!(howlearning[2],t[tstart:tend],bp_Ca[tstart:tend],color="blue",legend=:top,linewidth=1.25,ylabel="Calcium traces",label=L"C(t)")
plot!(howlearning[2],t[tstart:tend],(bp_Ca .- bp_gain_mod)[tstart:tend],color="indigo",linewidth=1.25,label=L"\overline{C(t)}",minorgrid=false,legend=:topright)
plot!(howlearning[3],t[tstart:tend],bp_gain_mod[tstart:tend],color="midnightblue",label="",linewidth=1.25,legend=false,ylabel=L"e(t)")
plot!(howlearning[4],t[tstart:tend],bp_P[id2,tstart:tend],color="hotpink",linewidth=1.25,xlabel="Time (s)",label="ID $id2",
    legend=:right,xticks=windowticks,minorgrid=false)
plot!(howlearning[4],t[tstart:tend],bp_P[id1,tstart:tend],color="teal",linewidth=1.25,xlabel="Time (s)",label="ID $id1",
    legend=:right,ylabel="w(t)",xticks=windowticks,minorgrid=false)
for i=1:3; plot!(howlearning[i],xformatter=_->""); end;
plot!()


id1 = 1120
id2 = 1089
id1_ids = findall(ys .== id1)
id1_xs = xs[id1_ids]
id1_ys = ys[id1_ids]
id2_ids = findall(ys .== id2)
id2_xs = xs[id2_ids]
id2_ys = ys[id2_ids]
tstart = 58000
tend = 65000
window = (5.8,6.5)
windowticks = [5.8,6.0,6.2,6.4]


vspans(howlearning,5:8,timestamps,pattern,window,0.2)
scatter!(twinx(howlearning[8]),id1_xs,id1_ys,group=grouping[id1_ids],markeralpha=1.0,markercolor="teal", xlabel="",
    markersize=12.0,markerstrokewidth=2,markershape=:vline,ylabel="",yticks=[],legend=false,
    minorgrid=false,xticks=[],xlim=window,guidefontsize=12,ytickfont=font(12),yflip=false)
scatter!(twinx(howlearning[8]),id2_xs,id2_ys,group=grouping[id2_ids],markeralpha=1.0,markercolor="hotpink", xlabel="",
    markersize=12.0,markerstrokewidth=2,markershape=:vline,ylabel="",yticks=[],legend=false,
    minorgrid=false,xticks=[],xlim=window,guidefontsize=12,ytickfont=font(12),yflip=true)
plot!(howlearning[5],t[tstart:tend],bpn_v_soma[tstart:tend],color="purple",label="",alpha=1.0,linewidth=1.25)
plot!(howlearning[5],t[tstart:tend],bpn_v_dend[tstart:tend],color="black",xlabel="", ylabel="",alpha=1.0,label="",
    minorgrid=false,linewidth=1.5,yticks=[20,0,-20,-40,-60],ylim=(-75,40),grid=false,legend=:topleft,guidefontsize=12)
plot!(howlearning[6],t[tstart:tend],bpn_Ca[tstart:tend],color="blue",legend=:top,linewidth=1.25,ylabel="",label=L"C(t)")
plot!(howlearning[6],t[tstart:tend],(bpn_Ca .- bpn_gain_mod)[tstart:tend],color="indigo",linewidth=1.25,label=L"\overline{C(t)}",minorgrid=false,legend=:topright,guidefontsize=12)
plot!(howlearning[7],t[tstart:tend],bpn_gain_mod[tstart:tend],color="midnightblue",label="",linewidth=1.25,legend=false,ylabel="",guidefontsize=12)
plot!(howlearning[8],t[tstart:tend],bpn_P[id2,tstart:tend],color="hotpink",linewidth=1.25,xlabel="Time (s)",label="ID $id2",
    ylabel="w",xticks=windowticks,minorgrid=false)
plot!(howlearning[8],t[tstart:tend],bpn_P[id1,tstart:tend],color="teal",linewidth=1.25,xlabel="Time (s)",label="ID $id1",
    legend=:bottomright,ylabel="",xticks=windowticks,minorgrid=false,guidefontsize=12)
#to save
for i=5:7; plot!(howlearning[i],xformatter=_->""); end;
plot!(tight_ticklabel=true)
savefig(howlearning,"howlearning.pdf")


## Figure for analysis of mcsd

tstart = 92000
tend = 105000
window = (9.25,10.0)
windowticks = [9.25,9.5,9.75,10.0]
bp_vals = [15.0, 60.0, 90.0]
nsteps = Int(sim_length/sim_δt)
soma_vect = zeros(Float64,6,nsteps)
dend_vect = zeros(Float64,6,nsteps)
gain_vect = zeros(Float64,6,nsteps)
Ca_vect = zeros(Float64,6,nsteps)
gcs_vect = zeros(Float64,6,nsteps)
P_vect = zeros(Float64,6,n_in,nsteps)
for (ii, sim) in enumerate(bp_vals)
    _,_,bp_v_dend,bp_v_soma,bp_g,_,_,bp_gain_mod,_,bp_w,_,_ = run_cell(sim_length,2018,zeros(n_in,1),3.75,sim)
    (bp_ge,bp_gi,bp_gcs,bp_Icsd,bp_Ca) = bp_g
    (bp_w_init,bp_w,bp_P) = bp_w
    dend_vect[ii,tstart:tend] = bp_v_dend[tstart:tend]
    soma_vect[ii,tstart:tend] = bp_v_soma[tstart:tend]
    Ca_vect[ii,tstart:tend] = bp_Ca[tstart:tend]
    gcs_vect[ii,tstart:tend] = bp_gcs[tstart:tend]
    P_vect[ii,:,tstart:tend] = bp_P[:,tstart:tend]
    gain_vect[ii,tstart:tend] = bp_gain_mod[tstart:tend]
end

lE = @layout [e;a;b;c]
id1 = 58
id2 = 56
diffmods = plot(ytickfont=font(10),
guidefont=font(10),yguidefontsize=10,background_color_legend=:transparent,foreground_color_legend=:transparent,
sharex=true,xtickfont=font(10),label=nothing,left_margin=4Plots.mm,legend=false,layout=lE,top_margin=0Plots.mm,
xminorgrid=false,right_margin = 3Plots.mm,bottom_margin=0Plots.mm,legendfontsize=12,size=(450,750))
gr(markersize=0.0, markerstrokewidth=0.0,legend=false)
vspans(diffmods,timestamps,pattern,window,0.25)
vline!(diffmods[1],windowticks,color=:gray,alpha=0.15,label="")
vline!(diffmods[2],windowticks,color=:gray,alpha=0.15)
vline!(diffmods[3],windowticks,color=:gray,alpha=0.15)
hline!(diffmods[3],[1.0],color=:gray,alpha=0.15) #missing otherwise
hline!(diffmods[4],[1.0],color=:gray,alpha=0.15) #missing otherwise
for (ii, sim) in enumerate(bp_vals)
    sim = Int(sim)
    plot!(diffmods[1],t,Ca_vect[ii,:],linewidth=2.0,label=L"%$sim}",ylabel="Calcium (μM)",legend=:top,xlabel="",xticks=[])
    plot!(diffmods[2],t,gain_vect[ii,:],linewidth=2.0,ylabel="PI (a.u.)",xlabel="",xticks=[])#,ylim=(-0.2,1.0),yticks=[-.1,.1,.3,.5,.7,.9])
    plot!(diffmods[3],t,P_vect[ii,id1,:],linewidth=2.0,ylabel="Δw (a.u.)",xlabel="",xticks=[])
    plot!(diffmods[4],t,P_vect[ii,id2,:],linewidth=2.0,ylabel="Δw (a.u.)",xlabel="Time (s)",xticks=[],ylim=(1.0,2.5))#,yticks=[.4,.7,1.0,1.3])
end
plot!(diffmods[4],xticks=windowticks)
plot!(xlim=window,sharex=true,yminorgrid=false)
annotate!(diffmods[3],[(9.375,1.1, (L"ID\;\#%$id1}", 12, :black, :left))])
annotate!(diffmods[4],[(9.375,1.1, (L"ID\;\#%$id2}", 12, :black, :left))])
intimes = xs[findall(ys .== id1)]
intimes = intimes[findall(x -> window[1] .< x .< window[2],intimes)]
ys_plot = ones(length(intimes)) .* 1.1
scatter!(diffmods[3],intimes,ys_plot,markeralpha=1.0,markersize=5.0,markershape=:vline,
legend=false,yminorgrid=false,xlabel="",xticks=[],color=:black)
intimes = xs[findall(ys .== id2)]
intimes = intimes[findall(x -> window[1] .< x .< window[2],intimes)]
ys_plot = ones(length(intimes)) .* 1.1
scatter!(diffmods[4],intimes,ys_plot,markeralpha=1.0,markersize=5.0, markershape=:vline,
legend=false,yminorgrid=false,xlabel="",xticks=[],color=:black)
#to save
savefig(diffmods,"data_figs/diffmods.pdf")



#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥ SNR data



#parameters
ms_to_sec = 1e-3
tfactor = 2.
sim_length = 10000.0 * tfactor
sim_δt = 0.1

#helper functions
include("funs.jl")

seeds = collect(1774:2023) #250 seeds (fig2_snr_tuningtable)
seeds = collect(1:500) #500 seeds
##how learning does not converge without gcsd mod

#simulation lifneur
include("lif_neur.jl")
include("bpHVAneur.jl")
include("bp_full.jl")

pat_time = 100.0

spk_output = []
Threads.@threads for seed in seeds
    spikes,ns,_,_,_,_,_,_,(_,_,pattern,_),_,_ = run_spk(sim_length,seed,zeros(2000,1),3.75,false)
    spikes *= ms_to_sec
    timestamps = [(pat_time*sim_δt*ms_to_sec,round(pattern[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
    push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
    tunings = []    
    for k in 0:3 # noise & patterns
        tuning = cumsum([sum(timestamps[ii][1] .< spikes .< timestamps[ii][2])/ns for ii in 1:length(pattern) if pattern[ii][1] == k])[end]
        push!(tunings,tuning)
    end
    snr = 20 .* log10.(tunings[2:end] ./ tunings[1])
    push!(spk_output,[seed,snr])
end
#[0.0,5.0,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.]

bp_vals = [90.,85.,80.,75.,70.,65.,60.,55.0, 50.0, 45.0, 40.,35.,25.,15.,0.]


bp_out = []
bpn_out = []
Threads.@threads for seed in seeds
    bp_data = []
    bpn_data = []
    for bp_val in bp_vals
        bp_spikes,bp_ns,_,_,_,_,_,_,(_,_,pattern,_),_,_,_ = run_bp(sim_length,seed,bp_val,false)
        bpn_spikes,bpn_ns,_,_,_,_,_,_,(_,_,pattern,_),_,_ = run_bpfull(sim_length,seed,bp_val,false)
        bp_spikes *= ms_to_sec
        bpn_spikes *= ms_to_sec
        #timestamps
        timestamps = [(pat_time*sim_δt*ms_to_sec,round(pattern[index+1][2]*sim_δt,digits=1)*ms_to_sec) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
        push!(timestamps,(timestamps[end][2],sim_length*ms_to_sec))
        bp_tunings = []
        bpn_tunings = []
        for k in 0:3 # noise & patterns
            bp_tuning = cumsum([sum(timestamps[ii][1] .< bp_spikes .< timestamps[ii][2])/bp_ns for ii in 1:length(pattern) if pattern[ii][1] == k])[end]
            bpn_tuning = cumsum([sum(timestamps[ii][1] .< bpn_spikes .< timestamps[ii][2])/bpn_ns for ii in 1:length(pattern) if pattern[ii][1] == k])[end]
            push!(bp_tunings,bp_tuning)
            push!(bpn_tunings,bpn_tuning)
        end
        snr = 20 .* log10.(bp_tunings[2:end] ./ bp_tunings[1])
        push!(bp_data,[bp_val,snr])
        snr = 20 .* log10.(bpn_tunings[2:end] ./ bpn_tunings[1])
        push!(bpn_data,[bp_val,snr])
    end
    push!(bp_out,[seed,bp_data])
    push!(bpn_out,[seed,bpn_data])
end


using DataFrames

snr_tuningtable = DataFrame()
snr_tuningtable.Seeds = [seed for (seed,_) in bp_out]
snr_tuningtable.Spike_based_r = [x for (_,(x,_,_)) in spk_output]
snr_tuningtable.Spike_based_g = [x for (_,(_,x,_)) in spk_output]
snr_tuningtable.Spike_based_b = [x for (_,(_,_,x)) in spk_output]
snr_tuningtable.Ca_based_90_r = [x for (_,((_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_90_g = [x for (_,((_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_90_b = [x for (_,((_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_r = [x for (_,(_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_g = [x for (_,(_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_b = [x for (_,(_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_r = [x for (_,(_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_g = [x for (_,(_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_b = [x for (_,(_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_r = [x for (_,(_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_g = [x for (_,(_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_b = [x for (_,(_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_r = [x for (_,(_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_g = [x for (_,(_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_b = [x for (_,(_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_r = [x for (_,(_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_g = [x for (_,(_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_b = [x for (_,(_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_r = [x for (_,(_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_g = [x for (_,(_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_b = [x for (_,(_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_r = [x for (_,(_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_g = [x for (_,(_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_b = [x for (_,(_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_r = [x for (_,(_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_g = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_b = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_r = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_g = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_b = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_25_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_)) in bp_out]
snr_tuningtable.Ca_based_25_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_)) in bp_out]
snr_tuningtable.Ca_based_25_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_)) in bp_out]
snr_tuningtable.Ca_based_15_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_)) in bp_out]
snr_tuningtable.Ca_based_15_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_)) in bp_out]
snr_tuningtable.Ca_based_15_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_)) in bp_out]
snr_tuningtable.Ca_based_0_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)))) in bp_out]
snr_tuningtable.Ca_based_0_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)))) in bp_out]
snr_tuningtable.Ca_based_0_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)))) in bp_out]
snr_tuningtable.Full_based_90_r = [x for (_,((_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_90_g = [x for (_,((_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_90_b = [x for (_,((_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_r = [x for (_,(_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_g = [x for (_,(_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_b = [x for (_,(_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_r = [x for (_,(_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_g = [x for (_,(_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_b = [x for (_,(_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_r = [x for (_,(_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_g = [x for (_,(_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_b = [x for (_,(_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_r = [x for (_,(_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_g = [x for (_,(_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_b = [x for (_,(_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_r = [x for (_,(_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_g = [x for (_,(_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_b = [x for (_,(_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_r = [x for (_,(_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_g = [x for (_,(_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_b = [x for (_,(_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_r = [x for (_,(_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_g = [x for (_,(_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_b = [x for (_,(_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_r = [x for (_,(_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_g = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_b = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_r = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_g = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_b = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_25_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_)) in bpn_out]
snr_tuningtable.Full_based_25_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_)) in bpn_out]
snr_tuningtable.Full_based_25_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_)) in bpn_out]
snr_tuningtable.Full_based_15_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_)) in bpn_out]
snr_tuningtable.Full_based_15_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_)) in bpn_out]
snr_tuningtable.Full_based_15_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_)) in bpn_out]
snr_tuningtable.Full_based_0_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)))) in bpn_out]
snr_tuningtable.Full_based_0_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)))) in bpn_out]
snr_tuningtable.Full_based_0_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)))) in bpn_out]

using JLD2 # change file names accordingly
@save "NEW2023_tuning_workspace_n500_bp.jld2" bp_out bpn_out
using CSV
CSV.write("NEW2023_fig2_snr_tuningtable_n500_bp.csv", snr_tuningtable)

 

#if computed separately

@load "../data/NEW2023_tuning_workspace_n500.jld2" spk_output

zaseeds = [seed for (seed,_) in bp_out]
spk_output = spk_output[sortperm(spk_output)][zaseeds]

using DataFrames

snr_tuningtable = DataFrame()
snr_tuningtable.Seeds = [seed for (seed,_) in spk_output]
snr_tuningtable.Spike_based_r = [x for (_,(x,_,_)) in spk_output]
snr_tuningtable.Spike_based_g = [x for (_,(_,x,_)) in spk_output]
snr_tuningtable.Spike_based_b = [x for (_,(_,_,x)) in spk_output]
snr_tuningtable.Ca_based_90_r = [x for (_,((_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_90_g = [x for (_,((_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_90_b = [x for (_,((_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_r = [x for (_,(_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_g = [x for (_,(_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_85_b = [x for (_,(_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_r = [x for (_,(_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_g = [x for (_,(_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_80_b = [x for (_,(_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_r = [x for (_,(_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_g = [x for (_,(_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_75_b = [x for (_,(_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_r = [x for (_,(_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_g = [x for (_,(_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_70_b = [x for (_,(_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_r = [x for (_,(_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_g = [x for (_,(_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_65_b = [x for (_,(_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_r = [x for (_,(_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_g = [x for (_,(_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_60_b = [x for (_,(_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_r = [x for (_,(_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_g = [x for (_,(_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_55_b = [x for (_,(_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_r = [x for (_,(_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_g = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_50_b = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_r = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_g = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_45_b = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_40_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_35_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_)) in bp_out]
snr_tuningtable.Ca_based_25_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_)) in bp_out]
snr_tuningtable.Ca_based_25_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_)) in bp_out]
snr_tuningtable.Ca_based_25_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_)) in bp_out]
snr_tuningtable.Ca_based_15_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_)) in bp_out]
snr_tuningtable.Ca_based_15_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_)) in bp_out]
snr_tuningtable.Ca_based_15_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_)) in bp_out]
snr_tuningtable.Ca_based_0_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)))) in bp_out]
snr_tuningtable.Ca_based_0_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)))) in bp_out]
snr_tuningtable.Ca_based_0_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)))) in bp_out]
snr_tuningtable.Full_based_90_r = [x for (_,((_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_90_g = [x for (_,((_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_90_b = [x for (_,((_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_r = [x for (_,(_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_g = [x for (_,(_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_85_b = [x for (_,(_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_r = [x for (_,(_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_g = [x for (_,(_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_80_b = [x for (_,(_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_r = [x for (_,(_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_g = [x for (_,(_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_75_b = [x for (_,(_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_r = [x for (_,(_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_g = [x for (_,(_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_70_b = [x for (_,(_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_r = [x for (_,(_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_g = [x for (_,(_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_65_b = [x for (_,(_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_r = [x for (_,(_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_g = [x for (_,(_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_60_b = [x for (_,(_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_r = [x for (_,(_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_g = [x for (_,(_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_55_b = [x for (_,(_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_r = [x for (_,(_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_g = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_50_b = [x for (_,(_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_r = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_g = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_45_b = [x for (_,(_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_40_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_35_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_,_)) in bpn_out]
snr_tuningtable.Full_based_25_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_,_)) in bpn_out]
snr_tuningtable.Full_based_25_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_,_)) in bpn_out]
snr_tuningtable.Full_based_25_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_,_)) in bpn_out]
snr_tuningtable.Full_based_15_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)),_)) in bpn_out]
snr_tuningtable.Full_based_15_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)),_)) in bpn_out]
snr_tuningtable.Full_based_15_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)),_)) in bpn_out]
snr_tuningtable.Full_based_0_r = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(x,_,_)))) in bpn_out]
snr_tuningtable.Full_based_0_g = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,x,_)))) in bpn_out]
snr_tuningtable.Full_based_0_b = [x for (_,(_,_,_,_,_,_,_,_,_,_,_,_,_,_,(_,(_,_,x)))) in bpn_out]

CSV.write("NEW2023_fig2_snr_tuningtable_n500_FULL.csv", snr_tuningtable)




