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

using Distributions
using Random
using LinearAlgebra
using StatsBase
using Statistics
using StableRNGs

include("funs.jl")

function run_ca_stdp(reps::Int64=60,Ipulse::Float64=500.0,t_delay::Int64=0,n::Int64=1,in_freq::Float64=1,seed::Int64=2)
    rng = StableRNG(seed)

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
    g_ld = 10.0 # conductance of leaky dendrite (nS)
    τ_ref = 5.0 # absolute refractory period (ms)
    g_cds = 108.0 # leak across comparments (nS) (coupling:0.9)
    g_csdr = 2.0 # leak across comparments (nS) (coupling:0.167)
    V_peak = 35.0 # peak upswing of somatic m.p. (mV)
    #τ = C_som/g_ls # membrane time constant = 15 (ms)
    τ_csd = 0.0 # back-propagation time delay (ms)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)
    m_cs = 60. # max back-propagation post-spike modulation (nS)
    min_m = 15.0 # min bAP post-spike modulation (nS) -- not used

    #HVA Calcium dynamics parameters (based on https://www.jneurosci.org/content/13/11/4609)
    g_ca = 0.01 # conductance of calcium channels (nS)
    Vca_r = 120.0 # calcium reversal potential (mV)

    #Calcium concentration parameters
    Ca0 = 50e-6 # baseline Ca2+ intracellular concentration (mM)
    Φ = 0.15 # Ca2+ current to concentration const. (mM/pA)
    τCa = 100. # Ca2+ concentration time constant (ms)

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

    #spike parameters
    Tperiod = Int(round(10000*(in_freq^-1),digits=0))
    Tno_act = 10000
    sim_length = (reps*Tperiod+Tno_act)*sim_δt #we make sure there are reps repetitions
    maxrate = 250.0
    max_spikes = round(Int64,maxrate*sim_length/1000.0) # maximum amount of spikes for defition purposes
    spks = zeros(max_spikes) # spike times
    n_steps = round(Int,sim_length/sim_δt) # simulation steps
    ns = 1 # number of somatic spikes
    last_spike = -100.0 # time of last spike
    spks[1] = -100.0

    #synaptic input parameters
    #time of input parameters
    n_rep = reps
    input_cell = zeros(Bool,(1,n_steps))
    I_pulse = zeros(n_steps)
    len_of_pulse = 15 #times 2 for actual length
    #t_delay = tpre-tpost
    for j in range(Tno_act,stop=n_steps-1)# for output & input
        if j % Tperiod == 0
            if n_rep > 0
                input_cell[j+t_delay] = 1 # delayed against post
                n_rep -= 1
            end
            if n == 1
                I_pulse[j-len_of_pulse:j+len_of_pulse] .= Ipulse
            end
            if n == 2
                spike_delay = 110
                I_pulse[j-len_of_pulse:j+len_of_pulse] .= Ipulse
                I_pulse[j-len_of_pulse-spike_delay:j+len_of_pulse-spike_delay] .= Ipulse
            end
        end
    end
    #synaptic input strength parameters
    κ = 1.0 # normalizing constant
    #ϐ = sum(abs.(w)) # sum of baseline displacements
    p_fail = 0.3 # probability of syn. transmission failure

    #learning parameters
    τΔ = 100.0 # low-pass filtering plasticity time constant (ms)
    τA = 10.0 # psps time in steps (ms)
    Tp = 100
    η = 0.05 # learning rate constant
    #Tϐ = 10 # time constant of synaptic resource (ms)
    τμCa = 20.
    τζ = 0.75 # shape of synaptic scaling function (a.u.)
    θζ = 1e-4 # threshold of synaptic scaling function (a.u.)

    #dynamically generating input
    #circ_affmats = zeros(Bool,n_in,τA/sim_δt+1)
    w = abs.(rand(rng,Normal()))
    circ_affvec = zeros(Bool,Tp+1)

    #initialization of variables
    v = Vs_l
    v_dend = Vd_l #+ (V_th - Vd_l)*0.5#*rand() # dend memb potential variable
    m_inf, τ_m, h_inf, τ_h = HVAc_rates(v_dend) # based on https://www.jneurosci.org/content/13/11/4609
    m = m_inf
    h = h_inf
    Ca = Ca0
    spiked = false
    back_prop = false
    cv = 0.0
    gain = 0.0
    ge = 0.0
    gi = 0.0
    g_csd = 0.0
    gca = 0.0
    Ica = 0.0
    Icsd = 0.0
    μCa = 0.0
    I_ext = 0.0
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
    w_init = deepcopy(w) #save previous syn. structure
    #input_cell = zeros(Bool,(1,n_steps))
    o_CA = zeros(n_steps)
    o_μCa = zeros(n_steps)
    o_gain = zeros(n_steps)
    o_At = zeros(n_steps)
    o_vsoma = zeros(n_steps)
    o_Ica = zeros(n_steps)
    o_gcsd = zeros(n_steps)
    o_vdend = zeros(n_steps)
    #main simulation
    for ti = 1:n_steps
        t = round(sim_δt*ti,digits=2)
        #initialization of conductances
        g_E = 0
        g_I = 0
        g_C = 0
        #external input (dynamically generated)
        if input_cell[ti]
            circ_affvec[end] = true
            g_E_prev += w
        end
        #homeostatic balancing (κ=1 here though)
        g_I_prev *= κ
        g_E_prev *= κ
        #double-exp-kernel update
        xe_rise += -sim_δt*xe_rise/τe_rise + g_E_prev;
        xe_decay += -sim_δt*xe_decay/τe_decay + g_E_prev;
        xi_rise += -sim_δt*xi_rise/τi_rise + g_I_prev;
        xi_decay += -sim_δt*xi_decay/τi_decay + g_I_prev;
        xcsd_rise += -sim_δt*xcsd_rise/τcsd_rise + g_csd_prev;
        xcsd_decay += -sim_δt*xcsd_decay/τcsd_decay + g_csd_prev;
        #transient coupling increase
        g_csd = g_csdr + Ncsd*(xcsd_decay - xcsd_rise)
        #synaptic input
        ge = g_e*Ne*(xe_decay - xe_rise)
        gi = g_i*Ni*(xi_decay - xi_rise)
        #HVA-Ca current dynamics
        m_inf, τ_m, h_inf, τ_h = HVAc_rates(v_dend)
        dm = (m_inf-m)/τ_m
        m += sim_δt*dm
        dh = (h_inf-h)/τ_h
        h += sim_δt*dh
        gca = g_ca*m*m*h
        Ica += gca*(Vca_r - v_dend)
        #somato-dendritic coupling current
        if t > (last_spike + τ_ref) # not in refractory
            Icsd = g_csdr*(v - v_dend)
        else
            Icsd = g_csd*(v - v_dend)
        end
        #external pulses
        I_ext = I_pulse[ti]
        #dendritic m.p. update
        dv_dend = (Icsd + ge*(Ve_r - v_dend) + gi*(Vi_r - v_dend) + g_ld*(Vd_l - v_dend) + Ica)/C_dend
        v_dend += sim_δt*dv_dend
        #somatic m.p. update
        if t > (last_spike + τ_ref) # not in refractory
            dv = (g_ls*(Vs_l - v) + g_cds*(v_dend - v) + I_ext)/C_som
            v += sim_δt*dv
            spiked = v > V_th
        else
            dv = (g_lsr*(Vs_l - v))/C_som
            v += sim_δt*dv
        end
        #spike dynamics
        if spiked
            spiked = false
            tdiff = last_spike - t
            v = V_peak # upswing
            last_spike = t
            ns += 1
            if ns <= max_spikes
                spks[ns] = t
            end
            back_prop = true
        end # end if(spiked)
        #calcium concentration
        dCa = Φ*Ica-(Ca-Ca0)/τCa
        Ca += sim_δt*dCa
        dμCa = (Ca - μCa)/τμCa
        μCa += sim_δt*dμCa
        dAt = -At/τA
        At += Float64(input_cell[ti]) + sim_δt*dAt
        #learning rule
        gain = Ca-μCa-5*Ca0
        #single plastic induction
        PI = w * At
        PI = PI * ζ(w,θζ,τζ)
        PI = gain * PI
        #low-pass filter (100ms) for PIs
        dP = (PI - P)/τΔ
        P += sim_δt * dP
        #plasticity boundaries
        P = P * any((sign(w+η*P) * sign(w)) == 1.0) # zero limits
        #induce plasticity if neuron is not refractory
        w += η*P
        #backwards conductance modulation
        if back_prop && t > (last_spike + τ_cs) # back-prop instance
            g_C += rand(Uniform(min_m,m_cs))#m_cs
            back_prop = false
        end
        #back-prop input
        g_csd_prev = copy(g_C)
        #new input
        g_E_prev = 0
        g_I_prev = 0
        o_CA[ti] = Ca
        o_μCa[ti] = μCa
        o_gain[ti] = gain
        o_At[ti] = At
        o_vsoma[ti] = v
        o_vdend[ti] = v_dend
        o_Ica[ti] = Ica
        circ_affvec = circshift(circ_affvec,-1)
        circ_affvec[end] = false
    end # end loop over time
    # prepare spike output
    popfirst!(spks)
    ns = ns - 1
    spks = spks[1:ns]
    rel_chg = w/w_init
    in_spks = [ii*sim_δt for (ii,each) in enumerate(input_cell) if each == true]

    return rel_chg, (spks,in_spks), (o_gain,o_At), (o_vsoma, o_vdend)
end
