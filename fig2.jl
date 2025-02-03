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

#data
seed = 2022
#helper functions
include("funs.jl")

#simulation bpHVAneur no modulation
include("bpHVAneur.jl")
bp_spikes,bp_ns,bp_v_dend,bp_v_soma,_,_,_,_,_,_,_,_ = run_bp(sim_length,seed,0.0)
#output spikes
bp_pss = bin_spikes(bp_spikes,Nbins,box_width,sim_steps,sim_δt)
bp_spikes = bp_spikes .* ms_to_sec
#simulation bpHVAneur with modulation (mbp = 60.0)
bpm_spikes,bpm_ns,bpm_v_dend,bpm_v_soma,_,_,_,_,poisson_in,_,_,_ = run_bp(sim_length,seed,60.0)
#output spikes
bpm_pss = bin_spikes(bpm_spikes,Nbins,box_width,sim_steps,sim_δt)
bpm_spikes = bpm_spikes .* ms_to_sec;
(presyns,postsyns,pattern,pats) = poisson_in

#timestamps
timestamps = [(pat_time*sim_δt/1000.0,round(pattern[index+1][2]*sim_δt,digits=1)/1000.0) for (index, (_,pat_time)) in enumerate(pattern) if index < length(pattern)]
push!(timestamps,(timestamps[end][2],sim_length))
print("completed!")



using LaTeXStrings
using ProgressMeter
using Distributions
using Random
using StableRNGs
using LinearAlgebra
using StatsBase
using Statistics
function test_bp(pseed::Int64=1919,w::Float64=1.0,bp::Float64=60.0,k = 1.5,Istr=400.,bar::Bool=false)
    adapt = true
    rng = StableRNG(pseed)
    sim_length=50.0
    w_init = w
    #cell parameters
    V_th = -50.0 # somatic membrane threshold potential (mV)
    Vs_l = -70.0 # somatic resting potential (mV)
    Vd_l = -70.0 # dendritic resting potential (mV)
    Ve_r = 0.0 # exc reversal potential (mV)
    Vi_r = -75.0 # inh reversal potential (mV)
    C_som = 180.0 # membrane capacitance (pF)
    C_dend = 60.0 # dend memb capacitance (pF)
    g_ls = 12.0 # conductance of leaky soma (nS)
    g_lsr = 150.0 # leakage conductance during ref. period (nS)
    g_ld = 25.0 # conductance of leaky dendrite (nS)
    τ_ref = 2.0 # absolute refractory period (ms)
    g_cds = 108.0 # leak across comparments (nS) (coupling:0.9)
    g_csdr = 8.0 # leak across comparments (nS) (coupling:0.167)
    V_peak = 35.0 # peak upswing of somatic m.p. (mV)
    #τ = C_som/g_ls # membrane time constant = 15 (ms)
    τ_csd = 0.0 # back-propagation time delay (ms)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)
    m_cs = bp # back-propagation post-spike modulation (nS)

    #HVA Calcium dynamics parameters (based on https://www.jneurosci.org/content/13/11/4609)
    g_ca = 0.01 # conductance of calcium channels (nS)
    Vca_r = 120.0 # calcium reversal potential (mV)

    #Calcium concentration parameters
    Ca0 = 50e-6 # baseline Ca2+ intracellular concentration (mM)
    Φ = 0.15 # Ca2+ current to concentration const. (mM/mA) from ~1um/F/2*1e4
    τCa = 100.0 # Ca2+ concentration time constant (ms)
    # peak Ca changes are up to 5.29 times resting values in Maravall et al., Biophys J., 2000.

    #synaptic timescale parameters
    τe_rise = 0.5 # rise time constant for exc synapses (ms)
    τi_rise = 1.0 # rise time constant for inh synapses (ms)
    τe_decay = 3.0 # decay time constant for exc synapses (ms)
    τi_decay = 8.0 # decay time constant for inh synapses (ms)
    τcsd_rise = 0.2 # rise time constant for coupled syn spike (ms)
    τcsd_decay = 1.5 # decay time constant for coupled syn spike (ms)
    tpeak_e = τe_decay*τe_rise/(τe_decay-τe_rise)*log(τe_decay/τe_rise)
    tpeak_i = τi_decay*τi_rise/(τi_decay-τi_rise)*log(τi_decay/τi_rise)
    tpeak_bp = τcsd_decay*τcsd_rise/(τcsd_decay-τcsd_rise)*log(τcsd_decay/τcsd_rise)

    #synaptic normalizing constants
    Ne = ((exp(-tpeak_e/τe_decay)-exp(-tpeak_e/τe_rise)))^-1
    Ni = ((exp(-tpeak_i/τi_decay)-exp(-tpeak_i/τi_rise)))^-1
    Ncsd = ((exp(-tpeak_bp/τcsd_decay)-exp(-tpeak_bp/τcsd_rise)))^-1

    #simulation parameters
    sim_δt = 0.1 # simulation time step (ms)
    sim_steps = Int(sim_length/sim_δt)

    #spike parameters
    maxrate = 250.0 # maximum firing rate of cell
    max_spikes = round(Int64,maxrate*sim_length/1000.0) # maximum amount of spikes for defition purposes
    spks = zeros(max_spikes) # spike times
    n_steps = round(Int,sim_length/sim_δt) # simulation steps
    ns = 1 # number of somatic spikes
    last_spike = -100.0 # time of last spike
    spks[1] = -100.0

    epsp_time = [-5000,-5010] # 10., 30., 60., 
    ipsp_time = [-0]
    p_fail = 0.3 # probability of syn. transmission failure


    #learning parameters
    τΔ = 100.0 # low-pass filtering plasticity time constant (ms)
    τμCa = 20.0 # low-pass filtered calcium time constant (ms)
    τAt = 10.0 # activated psps time (ms)
    Tϐ = 10 # time constant of synaptic resource (ms)
    η = 0.05 # learning rate constant
    τζ = 0.75 # shape of synaptic scaling function (a.u.)
    θζ = 1e-4 # threshold of synaptic scaling function (a.u.)

    #initialization of variables
    v = Vs_l + (V_th - Vs_l)*rand(rng)
    v_dend = Vd_l + (V_th - Vd_l)*rand(rng) # dend memb potential variable
    m_inf, τ_m, h_inf, τ_h = HVAc_rates(v_dend) # based on https://www.jneurosci.org/content/13/11/4609
    m = m_inf
    h = h_inf
    spiked = false
    back_prop = false
    cv = 0.0
    gain = 0.0
    ge = 0.0
    gi = 0.0
    gca = 0.0
    g_csd = 0.0
    Ca = Ca0
    μCa = 0.0
    

    #summed input of incoming spikes
    g_E = 0.0 # excitatory
    g_I = 0.0 # inhibitory
    g_C = 0.0 # self-coupling
    g_E_prev = 0.0 # previous timestep
    g_I_prev = 0.0
    g_csd_prev = 0.0
    #difference of exponentials
    xe_rise = 0.0 # excitatory rise exp
    xe_decay = 0.0 # excitatory decay exp
    xi_rise = 0.0 # inhibitory rise exp
    xi_decay = 0.0 # inhibitory decay exp
    xcsd_rise = 0.0 # back-prop time rise
    xcsd_decay = 0.0 # back-prop time decay
    P = 0.0
    At = 0.0
    stop_in = false

    #output parameters
    o_v = zeros(n_steps)
    o_v_dend = zeros(n_steps)
    o_ge = zeros(n_steps)
    o_gi = zeros(n_steps)
    o_gcsd = zeros(n_steps)
    o_pi = zeros(n_steps)
    spk_train = zeros(n_steps)
    o_Ica = zeros(n_steps)
    o_μIca = zeros(n_steps)
    o_Ca = zeros(n_steps)
    o_Icsd = zeros(n_steps)
    o_P = zeros(n_steps)

    p = Progress(n_steps, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=10, color=:blue)

    #main simulation
    for ti = 1:n_steps
        t = round(sim_δt*ti,digits=2)
        (bar) && (ProgressMeter.next!(p))
        #initialization of conductances
        g_E = 0
        g_I = 0
        g_C = 0
        #external input
        t = ti*sim_δt
        if ~stop_in
            if ti in epsp_time
                g_E_prev = w*k
            else
                g_E_prev = 0.0
            end
            if ti in ipsp_time
                g_I_prev = w*k
            else
                g_I_prev = 0.0
            end
        else
            g_E_prev = 0.0
            g_I_prev = 0.0
        end
        #homeostatic balancing
        g_I_prev *= k
        g_E_prev *= k
        #double-exp-kernel update
        xe_rise += -sim_δt*xe_rise/τe_rise + g_E_prev;
        xe_decay += -sim_δt*xe_decay/τe_decay + g_E_prev;
        xi_rise += -sim_δt*xi_rise/τi_rise + g_I_prev;
        xi_decay += -sim_δt*xi_decay/τi_decay + g_I_prev;
        xcsd_rise += -sim_δt*xcsd_rise/τcsd_rise + g_csd_prev;
        xcsd_decay += -sim_δt*xcsd_decay/τcsd_decay + g_csd_prev;
        #coupling adaptation
        g_csd = g_csdr + Ncsd*(xcsd_decay - xcsd_rise)
        #synaptic input
        ge = g_e*Ne*(xe_decay - xe_rise)
        gi = g_i*Ni*(xi_decay - xi_rise)
        #calcium dynamics
        m_inf, τ_m, h_inf, τ_h = HVAc_rates(v_dend)
        dm = (m_inf-m)/τ_m
        m += sim_δt*dm
        dh = (h_inf-h)/τ_h
        h += sim_δt*dh
        gca = g_ca*m*m*h
        Ica = gca*(Vca_r - v_dend)
        #somato-dendritic coupling current
        if t > (last_spike + τ_ref) # not in refractory
            Icsd = g_csdr*(v - v_dend)
        else
            Icsd = g_csd*(v - v_dend)
        end
        #dendritic m.p. update
        dv_dend = (ge*(Ve_r - v_dend) + gi*(Vi_r - v_dend) + g_ld*(Vd_l - v_dend) + gca*(Vca_r - v_dend) + Icsd)/C_dend
        v_dend += sim_δt*dv_dend
        #somatic m.p. update
        if t > (last_spike + τ_ref) # not in refractory
            dv = (g_ls*(Vs_l - v) + g_cds*(v_dend - v) + Istr*rand(rng,Normal(0.,10.)))/C_som
            v += sim_δt*dv
            spiked = v > V_th
        else
            dv = (g_lsr*(Vs_l - v))/C_som
            v += sim_δt*dv
        end
        #spike dynamics
        if spiked
            stop_in = true
            spiked = false
            v = V_peak # upswing
            last_spike = t
            ns += 1
            (ns <= max_spikes) && (spks[ns] = t)
            back_prop = true
        end # end if(spiked)
        #calcium concentration
        dCa = Φ*Ica-(Ca-Ca0)/τCa
        Ca += sim_δt*dCa
        dμCa = (Ca-μCa)/τμCa
        μCa += sim_δt*dμCa
        #inputs traces
        dAt = -At/τAt
        At += g_E_prev/k + sim_δt*dAt  
        #synaptic plasticity rule
        if adapt
            #plasticity induction update
            gain = (Ca - μCa)
            #unraveling of plastic inductions
            PI = abs(w * At)
            PI = PI * ζ(abs(w),θζ,τζ)
            PI = gain * PI
            #low-pass filter (100ms) for PIs
            dP = (PI - P)/τΔ
            P += sim_δt * dP
            #induce plasticity if neuron is not refractory
            w += η*P
        end
        #backwards conductance modulation
        if back_prop && t > (last_spike + τ_csd) # back-prop instance
            g_C += m_cs
            back_prop = false
        end
        #recording variables
        o_v[ti] = v
        o_v_dend[ti] = v_dend
        o_ge[ti] = ge
        o_gi[ti] = gi
        o_gcsd[ti] = g_csd
        o_pi[ti] = gain
        o_Ca[ti] = Ca
        o_Icsd[ti] = Icsd
        o_P[ti] = w
        #back-prop input
        g_csd_prev = copy(g_C)
        # new input
        g_E_prev = copy(g_E) # if there was synaptic connectivity this would be diff from zero
        g_I_prev = copy(g_I)
    end # end loop over time
    (bar) && (ProgressMeter.finish!(p))
    #println("Simulation finished.")
    # prepare spike output
    popfirst!(spks)
    ns = ns - 1
    spks = spks[1:ns]

    return spks,ns,o_v_dend,o_v,(o_ge,o_gi,o_gcsd,o_Icsd,o_Ca),o_pi,(w_init,w,o_P)
end
seed = 2003
bp_vals = [0.,15.,25.,35.,40.,45.,55.,60.0, 65.0, 70.0, 75.,80.,85.,90.]
vsoms = []
vdends = []
gcsds = []
for bp in bp_vals
    _,_,v_dend,v_som,(_,_,gcsd,_,_),_,(_,_,_) = test_bp(seed,1.0,bp,1.0,400.)
    push!(vsoms,v_som)
    push!(vdends,v_dend)
    push!(gcsds,gcsd)
end



#the figure 
l = @layout [[a{0.5w} b{0.15w} c{0.15w} d{0.15w} e{0.3w}; a{0.5w} b{0.15w} c{0.15w} d{0.15w} e{0.3w}] [a{0.005w};b{0.005w}]; a{0.7w} [grid(3,1){0.9w} b{0.1w}]]
h2 = plot(size=(1200,1200),layout=l,dpi=600,minorgrid=false,grid=false,
left_margin=2Plots.mm,right_margin=2Plots.mm,bottom_margin=0Plots.mm,guidefont=(14, :black))
plot!(h2[1],title = "A", titleloc = :left, titlefont = font(18))
plot!(h2[6],title = "B", titleloc = :left, titlefont = font(18))
plot!(h2[13],title = "C", titleloc = :left, titlefont = font(18))
plot!(h2[14],title = "D", titleloc = :left, titlefont = font(18))
for i in 1:5
    plot!(h2[i],xformatter=_->"",grid=false,minogrid=false);
end
for i in 2:5
    plot!(h2[i],yformatter=_->"",grid=false,minogrid=false);
end
for i in 7:10
    plot!(h2[i],yformatter=_->"",grid=false,minogrid=false);
end
for i in 11:12
    plot!(h2[i],grid=false,showaxis=false,minogrid=false)
end
for i in 14:15
    plot!(h2[i],xformatter=_->"",grid=false,minogrid=false)
end
plot!(h2[13],grid=false,showaxis=false,minogrid=false)
plot!(h2[17],grid=false,showaxis=false,minogrid=false)
plot!()



bpvd, bpvs, spks = bp_v_dend,bp_v_soma,bp_spikes
#single neuron response (m.p.)
gr(markersize=0.0,markershape=:auto, markerstrokewidth=0.0,markeralpha=0.0)
#plot!(h2[1],layout=l
#l = @layout [a b{0.15w} c{0.15w} d{0.15w} e{0.15w};a b{0.15w} c{0.15w} d{0.15w} e{0.15w}]
#h2 = plot(minorgrid=false,grid=false,sharex=false,sharey=true,layout=l, legend=:bottomright,
#left_margin=3Plots.mm,right_margin=3Plots.mm,bottom_margin=3Plots.mm,size=(800,400),ylim=(0,1))
vspans(h2,1:1,timestamps,pattern,(4.9,8.6),0.2)
vspans(h2,2:2,timestamps,pattern,(5.5,5.7),0.2)
vspans(h2,3:3,timestamps,pattern,(7.5,8.0),0.2)
vspans(h2,4:4,timestamps,pattern,(8.0,8.5),0.2)
vspans(h2,5:5,timestamps,pattern,(17.0,19.0),0.2)
dt_dend = fit(UnitRangeTransform, bpvd, dims=1)
dt_soma = fit(UnitRangeTransform, bpvs, dims=1)
norm_v_dend = StatsBase.transform(dt_dend, bpvd)
norm_v_soma = StatsBase.transform(dt_soma, bpvs)
ys = ones(length(spks)).*0.95
scatter!(h2[1],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
#plot!(h2[1],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
plot!(h2[1],t,norm_v_dend,color="black", ylabel="Activity",label=L"V_d",linewidth=1.5,xlim=(4.9,8.6),tickfont=(12, :black))#,xticks=[1.5,2.0,2.5,3.0])
plot!(h2[2],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[2],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[2],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(5.59,5.7),xticks=[5.6,5.7])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h2[3],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[3],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[3],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(7.85,7.95),xticks=[7.85,7.95])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h2[4],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[4],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[4],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(8.22,8.32),xticks=[8.22,8.32])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
#plot!(h2[5],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[5],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[5],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(17.4,18.8),xticks=[17.4,18.8])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)


vspans(h2,6:6,timestamps,pattern,(4.9,8.6),0.2)
vspans(h2,7:7,timestamps,pattern,(5.5,5.7),0.2)
vspans(h2,8:8,timestamps,pattern,(7.5,8.0),0.2)
vspans(h2,9:9,timestamps,pattern,(8.0,8.5),0.2)
vspans(h2,10:10,timestamps,pattern,(17.0,19.0),0.2)
bpvd, bpvs, spks = bpm_v_dend,bpm_v_soma,bpm_spikes
dt_dend = fit(UnitRangeTransform, bpvd, dims=1)
dt_soma = fit(UnitRangeTransform, bpvs, dims=1)
norm_v_dend = StatsBase.transform(dt_dend, bpvd)
norm_v_soma = StatsBase.transform(dt_soma, bpvs)
ys = ones(length(spks)).*0.95
scatter!(h2[6],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
#plot!(h2[1],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
plot!(h2[6],t,norm_v_dend,color="black", ylabel="Activity",label=L"V_d",linewidth=1.5,xlim=(4.9,8.6))#,xticks=[1.5,2.0,2.5,3.0])
plot!(h2[7],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[7],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[7],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(5.59,5.69),xticks=[5.59,5.69])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h2[8],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[8],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[8],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(7.85,7.95),xticks=[7.85,7.95])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
plot!(h2[9],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[9],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[9],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(8.22,8.32),xticks=[8.22,8.32])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
#plot!(h2[5],t,norm_v_soma,color="purple",xlabel="", ylabel="",label=L"V_s",linewidth=1.,alpha=1.)
scatter!(h2[10],spks,ys,markersize=10.0,markershape=:vline, markerstrokewidth=1.0,markeralpha=1.0,markercolor=:purple)
plot!(h2[10],t,norm_v_dend,color="black", ylabel="",label=L"V_d",linewidth=1.5,xlim=(17.4,19.),xticks=[17.4,18.8])
plot!(legend=false,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=14)
for ii in 6:10
    plot!(h2[ii],xlabel="Time (s)",guidefont=(14,:black),tickfont=(12, :black))
end
plot!()


nsteps = 500
ti = collect(1:nsteps).*sim_δt
pal=cgrad(:roma,14,categorical=true,scale=:exp)
for (ii,bp) in enumerate(bp_vals)
    plot!(h2[14],ti,vdends[ii],ylabel=L"V_d\;\;(mV)",color=pal[ii],linewidth=2.5,alpha=0.6,label="",xlim=(0,15),tickfont=(12, :black))
    plot!(h2[15],ti,vsoms[ii],ylabel=L"V_s\;\;(mV)",color=pal[ii],linewidth=2.5,linestyle=:auto,alpha=0.6,label="",xlim=(0,15),tickfont=(12, :black))
    plot!(h2[16],ti,gcsds[ii],ylabel=L"g_{csd}\;(nS)",color=pal[ii],linewidth=2.5,alpha=0.6,label="",xlim=(0,15))
end
plot!(h2[16],ti,ones(length(gcsds[1])).*108., markerstrokewidth=0.0,markeralpha=0.0,linewidth=1.5,linecolor=:red,label="",alpha=0.6,linestyle=:dash)
#plot!(h2[17],ti,ones(length(gcsds[1])).*108., markerstrokewidth=0.0,markeralpha=0.0,linewidth=1.5,linecolor=:red,label=L"g_{cds}",alpha=0.6,linestyle=:dash)
plot!(h2[16],xlabel="Time (ms)",tickfont=(12, :black))
#plot!(h2[17],legend=:best,background_color_legend=:transparent,foreground_color_legend=:transparent,legendfontsize=12.,xlim=(50,60))
scatter!(h2[17],rand(1),rand(1),marker_z=rand(1),clims=(0.0,90.0),xlims=(1,1.05),ylims=(1.0,1.05),showaxis=false,label="",
c=cgrad(:roma),colorbar_title=L"g_{csd}", colorbar_titlefont=font(18),)


savefig("fig2.png")


