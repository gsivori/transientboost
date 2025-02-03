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

function run_bp(sim_length=10000.0,pseed::Int64=1919,bp::Float64=60.0,bar::Bool=true,pretrained_w::Array{Float64,2}=zeros(2000,1),pretrained_κ::Float64=3.75)
    load_mat = false
    adapt = true
    jitter_checks = false
    rng = StableRNG(pseed)

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
    τ_ref = 2.0 # absolute refractory period (ms)
    g_cds = 108.0 # leak across comparments (nS)
    g_csdr = 8.0 # leak across comparments (nS)
    V_peak = 35.0 # peak upswing of somatic m.p. (mV) 
    #τ = C_som/g_ls # membrane time constant = 15 (ms)
    τ_csd = 0.0 # back-propagation time delay (ms)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)
    m_cs = bp # back-propagation post-spike modulation (nS)

    #HVA Calcium dynamics parameters
    g_ca = 0.01 # conductance of calcium channels (nS)
    Vca_r = 120.0 # calcium reversal potential (mV)

    #Calcium concentration parameters
    Ca0 = 50e-6 # baseline Ca2+ intracellular concentration (mM)
    Φ = 0.15 # Ca2+ current to concentration const. (mM/mA) from ~1um/F/2*1e4
    τCa = 100.0 # Ca2+ concentration time constant (ms)

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
    isis = zeros(max_spikes) # inter-spike intervals
    n_steps = round(Int,sim_length/sim_δt) # simulation steps
    ns = 1 # number of somatic spikes
    last_spike = -100.0 # time of last spike
    spks[1] = -100.0

    #synaptic input parameters
    n_in = 2000
    pat_width = 1000 # 100 ms in steps
    in_rate = 5.0 # Hz
    n_pats = 3
    pats = []
    pats = generate_input_pats(rng,n_in,pat_width,n_pats,in_rate,sim_δt)

    #synaptic input strength parameters
    (load_mat) && (w = pretrained_w)
    (~load_mat) && (w = rand(rng,Normal(),n_in))
    (load_mat) && (κ = pretrained_κ)
    (~load_mat) && (κ = 3.5) # normalizing constant
    ϐ = sum(abs.(w)) # sum of baseline displacements
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
    m_inf, τ_m, h_inf, τ_h = HVAc_rates(v_dend)
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
    P = zeros(n_in)
    At = zeros(n_in)
    w_init = deepcopy(w) #save previous syn. structure

    #output parameters
    o_v = zeros(n_steps)
    o_v_dend = zeros(n_steps)
    o_ge = zeros(n_steps)
    o_gi = zeros(n_steps)
    o_gcsd = zeros(n_steps)
    o_cvs = zeros(n_steps)
    o_pi = zeros(n_steps)
    spk_train = zeros(n_steps)
    o_ϐ = zeros(n_steps)
    o_κ = zeros(n_steps)
    o_Ica = zeros(n_steps)
    o_μIca = zeros(n_steps)
    o_Ca = zeros(n_steps)
    o_Icsd = zeros(n_steps)
    o_P = zeros(n_in,n_steps)

    #trial-dependent input
    input_mat, input_pat = generate_input_mat(rng,n_in, sim_steps, pats, pat_width, in_rate, sim_δt)
    presyns = deepcopy(input_mat)

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
            dv = (g_ls*(Vs_l - v) + g_cds*(v_dend - v))/C_som
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
            if ns <= max_spikes
                spks[ns] = t
                isis[ns-1] = spks[ns] - spks[ns-1]
            end
            back_prop = true
        end # end if(spiked)
        #coefficient of variation
        cv =  coef_var(diff(spks[.~iszero.(spks)]))
        (isnan(cv)) && (cv = 0.0)
        #calcium concentration
        dCa = Φ*Ica-(Ca-Ca0)/τCa
        Ca += sim_δt*dCa
        dμCa = (Ca-μCa)/τμCa
        μCa += sim_δt*dμCa
        #inputs traces
        dAt = -At/τAt
        At += Float64.(view(input_mat,:,ti)) + sim_δt*dAt
        #synaptic plasticity rule
        if adapt
            #plasticity induction update
            gain = (Ca - μCa)
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
        #backwards conductance modulation
        if back_prop && t > (last_spike + τ_csd) # back-prop instance
            g_C += m_cs
            back_prop = false
        end
        #recording variables
        o_κ[ti] = κ
        o_ϐ[ti] = ϐ
        o_v[ti] = v
        o_v_dend[ti] = v_dend
        o_ge[ti] = ge
        o_gi[ti] = gi
        o_gcsd[ti] = g_csd
        o_cvs[ti] = cv
        o_pi[ti] = gain
        o_Ca[ti] = Ca
        o_Icsd[ti] = Icsd
        o_P[:,ti] = w
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
    popfirst!(isis)
    ns = ns - 1
    spks = spks[1:ns]
    isis = isis[1:length(spks)-1]

    n_jitters = 6
    n_reps = 10
    n_tests = n_jitters*n_reps
    o_v_chk = zeros(Float32, n_steps,n_tests+1)
    o_v_dend_chk = zeros(Float32, n_steps,n_tests+1)
    spks_chk = zeros(Float32, max_spikes,n_tests+1)
    rngs_chk = sample(rng,1:500,n_tests,replace=false)
    ns_chk = ones(Int32, n_tests+1)

    if jitter_checks
        # checking against jittered input
        jitter = zeros(n_reps,n_jitters)
        jitter[:,1] .= 5.0
        jitter[:,2] .= 10.0
        jitter[:,3] .= 15.0
        jitter[:,4] .= 25.0
        jitter[:,5] .= 50.0
        jitter[:,6] .= 100.0
        jitter = vec(jitter)
        print("Performing jitter tests...\n")
        #main check simulation
        for te = 1:n_tests+1
            last_spike = -100.0
            spks_chk[1,te] = last_spike
            if te <= n_tests
                rng = StableRNG(rngs_chk[te])
                shuffled_input = jitter_data(input_mat,jitter[te],sim_δt)
            else
                shuffled_input = input_mat
            end
            for ti = 1:n_steps
                t = round(sim_δt*ti,digits=2)
                #initialization of conductances
                g_E = 0
                g_I = 0
                g_C = 0
                #external input
                for (index, active) in enumerate(shuffled_input[:,ti])
                    if active
                        if w[index] < 0
                            g_I_prev -= w[index] * κ
                        else
                            g_E_prev += w[index] * κ
                        end
                    end
                end
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
                    dv = (g_ls*(Vs_l - v) + g_cds*(v_dend - v))/C_som
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
                    ns_chk[te] += 1
                    if ns_chk[te] <= max_spikes
                        spks_chk[ns_chk[te],te] = t
                    end
                    back_prop = true
                end # end if(spiked)
                #backwards conductance modulation
                if back_prop && t > (last_spike + τ_csd) # back-prop instance
                    g_C += m_cs
                    back_prop = false
                end
                #recording variables
                o_v_chk[ti,te] = v
                o_v_dend_chk[ti,te] = v_dend
                #back-prop input
                g_csd_prev = copy(g_C)
                # new input
                g_E_prev = copy(g_E) # if there was synaptic connectivity this would be diff from zero
                g_I_prev = copy(g_I)
            end # end loop over time
        end
        spks_chk = circshift(spks_chk,(-1,0))
        spks_chk[end,:] .= 0.0
        ns_chk = ns_chk .- 1
    end

    return spks,ns,o_v_dend,o_v,(o_ge,o_gi,o_gcsd,o_Icsd,o_Ca),isis,o_cvs,o_pi,(presyns,input_mat,input_pat,pats),(w_init,w,o_P),(o_κ,o_ϐ), (spks_chk,ns_chk,o_v_chk,o_v_dend_chk)
end
