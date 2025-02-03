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

using ProgressMeter
using Distributions
using Random
using StableRNGs
using LinearAlgebra
using StatsBase
using Statistics

include("funs.jl")

function run_spk(sim_length=10000.0,pseed::Int64=1919, pretrained_w::Array{Float64,2}=zeros(2000,1), pretrained_κ::Float64=3.75,bar::Bool=true)
    load_mat = false 
    adapt = true
    rng = StableRNG(pseed)
    #cell parameters
    V_th = -50.0 # somatic membrane threshold potential (mV)
    V_peak = 30.0
    V_re = -65.0 # membrane reset potential (mV)
    Vs_l = -70.0 # somatic resting potential (mV)
    Vd_l = -70.0 # dendritic resting potential (mV)
    Ve_r = 0.0 # exc reversal potential (mV)
    Vi_r = -75.0 # inh reversal potential (mV)
    C_som = 150.0 # membrane capacitance (pF)
    C_dend = 60.0 # dend memb capacitance (pF)
    g_ls = 12.0 # conductance of leaky soma (nS)
    #g_lsr = 150.0 # same as above but during refractoriness (nS) -- not used in this model
    g_ld = 10.0 # conductance of leaky dendrite (nS)
    τ_ref = 2.0 # absolute refractory period (ms)
    g_cds = 108.0 # leak across comparments (nS) (coupling:0.9)
    g_csd = 8.0 # leak across comparments (nS) (coupling:0.167)
    #τ = C_som/g_ls # membrane time constant (ms)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)

    #synaptic timescales
    τe_rise = 0.5 # rise time constant for exc synapses (ms)
    τi_rise = 1.0 # rise time constant for inh synapses (ms)
    τe_decay = 3.0 # decay time constant for exc synapses (ms)
    τi_decay = 8.0 # decay time constant for inh synapses (ms)
    tpeak_e = τe_decay*τe_rise/(τe_decay-τe_rise)*log(τe_decay/τe_rise)
    tpeak_i = τi_decay*τi_rise/(τi_decay-τi_rise)*log(τi_decay/τi_rise)

    #synaptic normalizing constants
    Ne = ((exp(-tpeak_e/τe_decay)-exp(-tpeak_e/τe_rise)))^-1
    Ni = ((exp(-tpeak_i/τi_decay)-exp(-tpeak_i/τi_rise)))^-1

    #simulation parameters
    sim_δt = 0.1 # simulation time step (ms)
    sim_steps = Int(sim_length/sim_δt)

    #spike parameters
    maxrate = 250.0 # maximum firing rate of cell
    max_spikes = round(Int64,maxrate*sim_length/1000.0) # maximum amount of spikes for defition purposes
    spks = zeros(max_spikes) # spike times
    dspks = zeros(max_spikes) # dend. spike times
    isis = zeros(max_spikes) # inter-spike intervals
    disis = zeros(max_spikes) # dend. inter-spike intervals
    n_steps = round(Int,sim_length/sim_δt) # simulation steps
    ns = 1 # number of somatic spikes
    last_spike = -100.0 # time of last spike
    spks[1] = -100.0

    #synaptic input parameters
    n_in = 2000
    pat_width = 1000 # 100 ms in steps
    in_rate = 5.0
    n_pats = 3
    pats = generate_input_pats(rng,n_in, pat_width, n_pats, in_rate,sim_δt)

    #synaptic input strength parameters
    (load_mat) && (w = pretrained_w)
    (~load_mat) && (w = rand(rng,Normal(),n_in))
    (load_mat) && (κ = pretrained_κ)
    (~load_mat) && (κ = 3.5) # # normalizing constant
    ϐ = sum(abs.(w)) # sum of baseline displacements
    p_fail = 0.3 # probability of syn. transmission failure

    #learning parameters
    τst = 100.0 # time constant for low-pass filter spike train
    τΔ = 100.0 # low-pass filtering plasticity time constant (ms)
    τμ = 20.0 # low-pass filtered LPF of post-spikes time constant (ms)
    τAt = 10.0 # activated psps time (ms)
    Tϐ = 10 # time constant of synaptic resource (1ms)
    η = 0.05 # learning rate constant
    τζ = 0.75 # shape of synaptic scaling function (a.u.)
    θζ = 1e-4 # threshold of synaptic scaling function (a.u.)

    #initialization of variables
    v = V_re + (V_th - V_re)*rand(rng)
    v_dend = V_re + (V_th - V_re)*rand(rng) # dend memb potential variable
    spiked = false
    back_prop = false
    cv = 0.0
    gain = 0.0
    ge = 0.0
    gi = 0.0
    St = 0.0
    Y = 0.0
    μ = 0.0 # mean somatic spike train
    #summed input of incoming spikes
    g_E = 0.0 # excitatory
    g_I = 0.0 # inhibitory
    g_C = 0.0 # self-coupling
    g_E_prev = 0.0 # previous timestep
    g_I_prev = 0.0
    #difference of exponentials
    xe_rise = 0.0 # excitatory rise exp
    xe_decay = 0.0 # excitatory decay exp
    xi_rise = 0.0 # inhibitory rise exp
    xi_decay = 0.0 # inhibitory decay exp
    P = zeros(n_in)
    At = zeros(n_in)
    w_init = deepcopy(w) #save previous syn. structure

    #output parameters
    o_v = zeros(n_steps)
    o_v_dend = zeros(n_steps)
    o_ge = zeros(n_steps)
    o_gi = zeros(n_steps)
    o_cvs = zeros(n_steps)
    o_pi = zeros(n_steps)
    spk_train = zeros(n_steps)
    o_ϐ = zeros(n_steps)
    o_κ = zeros(n_steps)
    o_μ = zeros(n_steps)
    o_P = zeros(n_in,n_steps)

    #trial-dependent input
    input_mat, input_pat = generate_input_mat(rng,n_in, sim_steps, pats, pat_width, in_rate, sim_δt)
    #input_mat, input_pat = generate_input_mat(rng,n_in, sim_steps, [pat1,pat2,pat3], pat_width, in_rate, sim_δt, [pat1_n, pat2_n, pat3_n])

    presyns = deepcopy(input_mat)

    p = Progress(n_steps, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=10, color=:green)

    #main simulation
    for ti = 1:n_steps
        t = round(sim_δt*ti,digits=2)
        (bar) && (ProgressMeter.next!(p))
        #initialization of conductances
        g_E = 0
        g_I = 0
        #external input
        for (index, active) in enumerate(input_mat[:,ti])
            if active
                if rand(rng) > p_fail
                    if w[index] < 0
                        g_I_prev -= w[index]
                    else
                    	g_E_prev += w[index]
                    end
                else
                    input_mat[index,ti] = false # remove syn.
                end
            end
        end
        #homeostatic balancing
        g_I_prev *= κ
        g_E_prev *= κ
        #double-exp-kernel update
        xe_rise += -sim_δt*xe_rise/τe_rise + g_E_prev
        xe_decay += -sim_δt*xe_decay/τe_decay + g_E_prev
        xi_rise += -sim_δt*xi_rise/τi_rise + g_I_prev
        xi_decay += -sim_δt*xi_decay/τi_decay + g_I_prev
        #synaptic input
        ge = g_e*Ne*(xe_decay - xe_rise)
        gi = g_i*Ni*(xi_decay - xi_rise)
        #dendritic m.p. update
        dv_dend = (g_csd*(v - v_dend) + ge*(Ve_r - v_dend) + gi*(Vi_r - v_dend) + g_ld*(Vd_l - v_dend))/C_dend
        v_dend += sim_δt*dv_dend
        if t > (last_spike + τ_ref) # not in refractory
            #somatic m.p. update
            dv = (g_ls*(Vs_l - v) + g_cds*(v_dend - v))/C_som
            v += sim_δt*dv;
            spiked = v > V_th
        #else #-- not used in this model
        #    dv = (g_lsr*(Vs_l - v))/C_som
        #    v += sim_δt*dv
        end
        St = Int(spiked)
        dY = (τst*St-Y)/τst
        Y += sim_δt*dY
        #spike dynamics
        if spiked
            spiked = false
            v = V_re#V_peak # upswing
            last_spike = t
            ns += 1
            if ns <= max_spikes
                spks[ns] = t
                isis[ns-1] = spks[ns] - spks[ns-1]
            end
        end # end if(spiked)
        #average somatic activity integration
        dμ = (Y - μ)/τμ
        μ += sim_δt*dμ
        #inputs traces
        dAt = -At/τAt
        At += Float64.(view(input_mat,:,ti)) + sim_δt*dAt
        #coefficient of variation
        cv =  coef_var(diff(spks[.~iszero.(spks)]))
        (isnan(cv)) && (cv = 0.0)
        #always updating: timescales τAt = 10 (10ms), Tϐ = 10 (1ms)
        if adapt
            #plasticity induction update
            gain = (Y-μ)
            #unraveling of plastic inductions
            PI = @. abs.(w * At)
            PI = @. PI * ζ(abs.(w),θζ,τζ)
            PI = @. gain * PI
            #low-pass filter (100ms) for PIs
            dP = (PI .- P)/τΔ
            P += sim_δt .* dP
            #plasticity boundaries
            P = P .* any((sign.(w+η*P) .* sign.(w)) .== 1.0,dims=2) # zero limits
            #induce plasticity if neuron is not refractory
            w += @. η*P
            #homeostatic plasticity
            if (ti%Tϐ) == 0.0
                newϐ = sum((abs.(w)))
                κ*= 1-((κ*newϐ)-κ*ϐ)/(κ*ϐ)
                ϐ = newϐ
            end
        end
        #recording variables
        o_μ[ti] = μ
        o_κ[ti] = κ
        o_ϐ[ti] = ϐ
        spk_train[ti] = Y
        o_v[ti] = v
        o_v_dend[ti] = v_dend
        o_ge[ti] = ge
        o_gi[ti] = gi
        o_cvs[ti] = cv
        o_pi[ti] = gain
        o_P[:,ti] = w
        # new input
        g_E_prev = copy(g_E) # if there was synaptic connectivity this would be diff from zero
        g_I_prev = copy(g_I)
    end # end loop over time
    (bar) && (ProgressMeter.finish!(p))
    # prepare spike output
    popfirst!(spks)
    popfirst!(isis)
    ns = ns - 1
    spks = spks[1:ns]
    isis = isis[1:length(spks)-1]

    return spks, ns, o_v_dend, o_v, (o_ge, o_gi), isis, o_cvs, (o_pi, spk_train), (presyns,input_mat,input_pat,pats), (w_init, w, o_P), (o_κ, o_ϐ, o_μ)
end
