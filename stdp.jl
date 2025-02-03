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

function run_stdp(reps::Array{Int64}=[20,30],Ipulse::Float64=5000.0,t_delay::Int64=0,n::Int64=1,in_freq::Float64=5.0,seed::Int64=2023,bar::Bool=true)
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
    τ_ref = 3. # absolute refractory period (ms) #5
    g_cds = 108.0 # leak across comparments (nS) (coupling:0.9)
    g_csdr = 8.0 # leak across comparments (nS) (coupling:0.167)
    V_peak = 35.0 # peak upswing of somatic m.p. (mV)
    τ_csd = 0.0 # back-propagation time delay (ms)
    m_cs = 60. # back-propagation post-spike modulation (nS) (equivalent--Check Methods section)

    #NMDA synaptic parameters (not used)
    pf = 0.05 # current factor NMDA-to-Ca (a.u.)
    mg = 1.0 # magnesium concentration (mM)
    Vnmda_r = 0.0 # NMDA synapse reversal potential (mV)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)
    ghat_nmda = 5e-4 # NMDA peak conductance (nS)

    #HVA Calcium dynamics parameters (based on https://www.jneurosci.org/content/13/11/4609)
    g_ca = 0.01 # conductance of calcium channels (nS)
    Vca_r = 120.0 # calcium reversal potential (mV)

    #Calcium concentration parameters
    Ca0 = 50e-6 # baseline Ca2+ intracellular concentration (mM)
    Φ = 0.15 # Ca2+ current to concentration const. (mM/pA)
    τCa = 100.0 # Ca2+ concentration time constant (ms)
    # peak Ca changes are up to 5.29 times resting values in Maravall et al., Biophys J., 2000.

    #synaptic timescale parameters
    τe_rise = 0.5 # rise time constant for exc synapses (ms)
    τi_rise = 1.0 # rise time constant for inh synapses (ms)
    τe_decay = 3.0 # decay time constant for exc synapses (ms)
    τi_decay = 8.0 # decay time constant for inh synapses (ms)
    τnmda_rise = 3.3 # rise time constant for nmda channel (ms)
    τnmda_decay = 102.38 # decay time constant for nmda channel (ms)
    τcsd_rise = 0.2 # rise time constant for coupled syn spike (ms)
    τcsd_decay = 1.5 # decay time constant for coupled syn spike (ms)
    tpeak_e = τe_decay*τe_rise/(τe_decay-τe_rise)*log(τe_decay/τe_rise)
    tpeak_i = τi_decay*τi_rise/(τi_decay-τi_rise)*log(τi_decay/τi_rise)
    tpeak_nmda = τnmda_decay*τnmda_rise/(τnmda_decay-τnmda_rise)*log(τnmda_decay/τnmda_rise)
    tpeak_bp = τcsd_decay*τcsd_rise/(τcsd_decay-τcsd_rise)*log(τcsd_decay/τcsd_rise)

    #synaptic normalizing constants
    Ne = ((exp(-tpeak_e/τe_decay)-exp(-tpeak_e/τe_rise)))^-1
    Ni = ((exp(-tpeak_i/τi_decay)-exp(-tpeak_i/τi_rise)))^-1
    Nnmda = ((exp(-tpeak_nmda/τnmda_decay)-exp(-tpeak_nmda/τnmda_rise)))^-1
    Ncsd = ((exp(-tpeak_bp/τcsd_decay)-exp(-tpeak_bp/τcsd_rise)))^-1

    #spike parameters
    sim_δt = 0.1 # simulation time step (ms)
    Tperiod = Int(round(10000*(in_freq^-1),digits=0))
    Tno_act = 10000
    reps = rand(reps[1]:reps[2])
    sim_length = (reps*Tperiod+Tno_act)*sim_δt #we make sure there are reps repetitions
    max_spikes = reps*n_burst+1 # maximum amount of spikes for defition purposes
    spks = zeros(max_spikes) # spike times
    n_steps = round(Int,sim_length/sim_δt) # simulation steps
    ns = 1 # number of somatic spikes
    last_spike = -100.0 # time of last spike
    if n>1
        t_delay -= 50 #compensate APs
    end
    spks[1] = -100.0

    #synaptic input parameters
    Inoise = 100. #Inoise = 100 on dendrite has no effect. Same results as if Inoise = 0.
    #time of input parameters
    n_rep = reps
    input_cell = zeros(Bool,(1,n_steps))
    I_pulse = zeros(n_steps)
    len_of_pulse = 15 #times 2 for actual length (3ms)
    for j in range(Tno_act,stop=n_steps-1)# for output & input
        if j % Tperiod == 0
            if n_rep > 0
                input_cell[j+t_delay] = 1 # delayed against post: t_delay = tpre-tpost
                n_rep -= 1
            end
            if n == 1
                I_pulse[j-len_of_pulse:j+len_of_pulse] .= Ipulse
            end
            if n == 2
                spike_delay = 110
                I_pulse[j-len_of_pulse:j+len_of_pulse] .= Ipulse #second AP
                I_pulse[j-len_of_pulse-spike_delay:j+len_of_pulse-spike_delay] .= Ipulse
            end
        end
    end
    in_spks = [ii*sim_δt for (ii,each) in enumerate(input_cell) if each == true]
    #synaptic input strength parameters
    p_fail = 0.3
    κ = 1.0 # normalizing constant
    #synaptic input strength parameters
    w = abs(rand(rng,Normal()))

    #learning parameters
    τΔ = 100.0 # low-pass filtering plasticity time constant (ms)
    τμCa = 20.0 # low-pass filtered calcium time constant (ms)
    τAt = 10.0 # activated psps time (ms)
    Tϐ = 10 # time constant of synaptic resource (ms) -- not used here
    η = 0.04 # learning rate constant
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
    gain = 0.0
    ge = 0.0
    gi = 0.0
    gca = 0.0
    Ca = Ca0
    μCa = 0.0
    P = 0.0
    At = 0.0
    Inmda = 0.0
    g_csd = 0.0
    #summed input of incoming spikes
    g_E = 0.0 # excitatory
    g_I = 0.0 # inhibitory
    g_NMDA = 0.0
    g_C = 0.0 # self-coupling
    g_E_prev = 0.0 # previous timestep
    g_I_prev = 0.0
    g_NMDA_prev = 0.0
    g_csd_prev = 0.0
    #difference of exponentials
    xe_rise = 0.0 # excitatory rise exp
    xe_decay = 0.0 # excitatory decay exp
    xi_rise = 0.0 # inhibitory rise exp
    xi_decay = 0.0 # inhibitory decay exp
    xnmda_rise = 0.0 # nmda rise exp
    xnmda_decay = 0.0 # nmda decay exp
    xcsd_rise = 0.0 # back-prop time rise
    xcsd_decay = 0.0 # back-prop time decay
    w_init = deepcopy(w) #save previous syn. structure

    #output parameters
    o_v = zeros(n_steps)
    o_v_dend = zeros(n_steps)
    o_ge = zeros(n_steps)
    o_gi = zeros(n_steps)
    o_gcsd = zeros(n_steps)
    o_gnmda = zeros(n_steps)
    o_pi = zeros(n_steps)
    o_At = zeros(n_steps)
    o_κ = zeros(n_steps)
    o_Ica = zeros(n_steps)
    o_Ca = zeros(n_steps)
    o_P = zeros(n_steps)
    o_Inmda = zeros(n_steps)
    o_Icsd = zeros(n_steps)

    #main simulation
    for ti = 1:n_steps
        t = round(sim_δt*ti,digits=2)
        #initialization of conductances
        g_E = 0
        g_I = 0
        g_C = 0
        g_NMDA = 0
        #external input (dynamically generated)
        if rand(rng) > p_fail && input_cell[ti]
            g_E_prev += w
            #g_NMDA_prev += w (not used) #NMDA would not activate
        end
        #homeostatic balancing
        g_E_prev *= κ
        g_I_prev *= κ
        #double-exp-kernel update
        xe_rise += -sim_δt*xe_rise/τe_rise + g_E_prev;
        xe_decay += -sim_δt*xe_decay/τe_decay + g_E_prev;
        xi_rise += -sim_δt*xi_rise/τi_rise + g_I_prev;
        xi_decay += -sim_δt*xi_decay/τi_decay + g_I_prev;
        xnmda_rise += -sim_δt*xnmda_rise/τnmda_rise + g_NMDA_prev;
        xnmda_decay += -sim_δt*xnmda_decay/τnmda_decay + g_NMDA_prev;
        xcsd_rise += -sim_δt*xcsd_rise/τcsd_rise + g_csd_prev;
        xcsd_decay += -sim_δt*xcsd_decay/τcsd_decay + g_csd_prev;
        #bPAP coupling modulation
        g_csd = g_csdr + Ncsd*(xcsd_decay - xcsd_rise)
        #nmda dynamics
        gnmda = ghat_nmda*Nnmda*(xnmda_decay - xnmda_rise)*Mg_block(v_dend,mg)
        Inmda = gnmda*(Vnmda_r - v_dend)*(1-pf)
        Ica = gnmda*(Vnmda_r - v_dend)*pf
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
        Ica += gca*(Vca_r - v_dend)
        #somato-dendritic coupling current
        if t > (last_spike + τ_ref) # not in refractory
            Icsd = g_csdr*(v - v_dend)
        else
            Icsd = g_csd*(v - v_dend)
        end
        #dendritic m.p. update
        dv_dend = (Icsd + ge*(Ve_r - v_dend) + gi*(Vi_r - v_dend) + g_ld*(Vd_l - v_dend) + Ica + Inmda + Inoise*rand(Normal()))/C_dend
        v_dend += sim_δt*dv_dend
        #external pulses
        I_ext = I_pulse[ti]
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
        At += Float64(input_cell[ti]) + sim_δt*dAt
        #synaptic plasticity rule
        gain = (Ca-μCa-5*Ca0)
        #unraveling of plastic inductions
        PI = gain * abs(w * At) * ζ(abs(w),θζ,τζ)
        #low-pass filter for PIs
        dP = (PI - P)/τΔ
        P += sim_δt * dP
        #plasticity boundaries
        P = P * any((sign(w+η*P) * sign(w)) == 1.0) # zero limits
        #induce plasticity if neuron is not refractory
        w += η*P
        #backwards conductance modulation
        if back_prop && t > (last_spike + τ_csd) # back-prop instance
            g_C += m_cs
            back_prop = false
        end
        #recording variables
        o_P[ti] = w
        o_κ[ti] = κ
        o_v[ti] = v
        o_v_dend[ti] = v_dend
        o_ge[ti] = ge
        o_gi[ti] = gi
        o_gcsd[ti] = g_csd
        o_gnmda[ti] = gnmda
        o_pi[ti] = gain
        o_Ca[ti] = Ca
        o_Ica[ti] = Ica
        o_Inmda[ti] = Inmda
        o_Icsd[ti] = Icsd
        o_At[ti] = At
        # new input
        g_E_prev = copy(g_E) # if there was synaptic connectivity this would be diff from zero
        g_I_prev = copy(g_I)
        g_NMDA_prev = copy(g_NMDA)
        g_csd_prev = copy(g_C)
    end # end loop over time
    #prepare spike output
    popfirst!(spks)
    rel_chg = w/w_init

    return rel_chg, (spks,in_spks), (o_pi,o_At),(o_v,o_v_dend, o_gcsd)
end
