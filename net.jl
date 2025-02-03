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
using LinearAlgebra
using StatsBase
using Statistics
using SparseArrays
using StableRNGs

include("funs.jl")

function run_network(sim_length::Float64=10000.0, N::Int64=1000, pseed::Int64=2021)
    clustered = true
    rec_plastic = true
    rng = StableRNG(pseed)
    #network parameters 80/20
    n_exc = Int(N - 0.2*N)
    n_inh = Int(0.2*N)
    n_in = 2000
    #cell parameters
    V_th = -50.0 # somatic membrane threshold potential (mV)
    Vs_l = -70.0 # somatic resting potential (mV)
    Vd_l = -70.0 # dendritic resting potential (mV)
    Ve_r = 0.0 # exc reversal potential (mV)
    Vi_r = -75.0 # inh reversal potential (mV)
    C_som = 180.0 # membrane capacitance (pF)
    C_dend = 60.0 # dend memb capacitance (pF)
    C_inh = 150.0 # membrane capacitance of inh cells (pF)
    g_ls = 12.0 # conductance of leaky soma (nS)
    g_lsr = 150.0 # leakage conductance during ref. period (nS)
    g_ld = 10.0 # conductance of leaky dendrite (nS)
    τ_ref = 3.0 # absolute refractory period (ms)
    g_cs = 108.0 # leak across comparments (nS)
    g_csdr = 2.0 # leak across comparments (nS)
    V_peak = 35.0 # peak upswing of somatic m.p. (mV)
    τ_bp = 0.0 # back-propagation time delay (ms)

    #synaptic peak conductances
    g_e = 5e-1 # AMPA peak conductance (nS)
    g_i = 5.5e-1 # GABA peak conductance (nS)
    m_bp = 0.0 # back-propagation post-spike modulation (nS) exc cells
    m_bp_inh = 0.0 # back-propagation post-spike modulation (nS) inh cells

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
    input_steps = Int(sim_length/sim_δt)+1#Int(sim_length/sim_δt*0.8)

    #learning parameters
    τΔ = 100.0 # low-pass filtering plasticity time constant (ms)
    τst = 100.0 #Y time constant
    τμ = 20.0 # low-pass filtered calcium time constant (ms)
    τAt = 10.0 # activated psps time (ms)
    Tϐ = 10 # time constant of synaptic resource (ms)
    η = 0.05 # learning rate constant for afferent input
    η_r = 0.05 # learning rate constant for recurrent input
    Tsample = 10000 # sample recurrent structure every second (ms)
    τζ = 0.75 # shape of synaptic scaling function (a.u.)
    θζ = 1e-4 # threshold of synaptic scaling function (a.u.)
    Tp = 100 #for rotating input mats (steps)

    #spike parameters
    maxrate = 250.0 # maximum firing rate of cell
    max_spikes = round(Int64,maxrate*sim_length/1000.0) # maximum amount of spikes for defition purposes
    spks = zeros(N, max_spikes) # somatic spike times matrix
    ns = ones(Int64, N) # number of somatic spikes per cell
    last_spike = -100.0*ones(N) # time of last spike
    spks[:,1] .= -100.0

    #parameter vectors
    C = zeros(N)
    C[1:n_exc] .= C_som
    C[(n_exc+1):N] .= C_inh
    exc_cells = collect(1:n_exc)
    inh_cells = collect((n_exc+1):N)
    Vth = zeros(N)
    Vth[1:n_exc] .= V_th
    Vth[(n_exc+1):N] .= V_th
    mbp = zeros(N)
    mbp[1:n_exc] .= m_bp
    mbp[(n_exc+1):N] .=  m_bp_inh

    #synaptic input strength parameters
    signs = sign.(rand(rng,Normal(),n_in))
    w_in = (abs.(rand(rng,Normal(),n_in,N)) .* signs)
    p_fail = 0.3 # probability of syn. transmission failure
    #recurrent connection probabilities
    Pee = .1
    Pei = .3
    Pie = .4
    Pii = .5

    #recurrent network
    homeo_k = 3.5
    w_r = spzeros(N,N)
    #exc->exc
    pres, posts = get_connections(rng,exc_cells, exc_cells, n=Pee*n_exc*n_exc)
    w_r += sparse(pres,posts,abs.(rand(rng,LogNormal(-0.42,0.79),length(pres))),N,N) 
    #exc->inh
    pres, posts = get_connections(rng,exc_cells, inh_cells, n=Pei*n_exc*n_inh)
    w_r += sparse(pres,posts,abs.(rand(rng,LogNormal(0.48,0.63),length(pres))),N,N) 
    #inh->exc
    pres, posts = get_connections(rng,inh_cells, exc_cells, n=Pie*n_inh*n_exc)
    w_r += sparse(pres,posts,-1 .* abs.(rand(rng,LogNormal(-0.74,0.96),length(pres))),N,N) 
    #inh->inh
    pres, posts = get_connections(rng,inh_cells, inh_cells, n=Pii*n_inh*n_inh)
    w_r += sparse(pres,posts,-1 .* abs.(rand(rng,LogNormal(-0.39,0.78),length(pres))),N,N) 

    #normalizing params
    κ = ones(N)*homeo_k # normalizing constant
    ϐ = sum(abs.(w_in),dims=1) + sum(abs.(w_r),dims=1) # sum of baseline displacements

    # cluster connectivity parameters
    P_c = .5 #in-cluster connectivity probability
    μ_cs = 18
    std_cs = 3
    #parameters for clusters (Normal-distributed weights)
    μ_c = 3.0 #3.0 for strong. 0.0 for weak
    std_c = 1.0
    #within-cluster connections
    csid = []
    if clustered
        cs = get_cluster_sizes(rng,n_exc,μ_cs,std_cs)
        pres, posts, syns, csid = get_clusters_conns(rng,exc_cells, cs, P_c, μ_c, std_c)
        w_r += sparse(pres,posts,syns,N,N)
    end
    #no self connections
    w_r[diagind(w_r)] .= 0
    dropzeros!(w_r)
    #vector of recurrent weight matrix over snapshots
    o_Pr = []
    o_Paf = []
    push!(o_Pr,deepcopy(w_r))
    push!(o_Paf,deepcopy(w_in))

    #synaptic input parameters
    offset = 1 # .1 ms offset
    pat_width = 1000 # 100 ms in steps
    avg_rate = 5.0 # Hz
    std_rate = 1.0 # Hz
    npat = 3
    #trial-unique-pattern input
    pats = generate_input_patsV2(rng,n_in,pat_width,npat,avg_rate,sim_δt)
    #trial-dependent input
    input_mat, input_pat = generate_input_matV2(rng,n_in,sim_steps,pats,pat_width,avg_rate,sim_δt)

    #initialization of variables
    v = Vs_l .+ (V_th - Vs_l)*rand(rng,N)
    v_dend = Vd_l .+ (Vth .- Vd_l) .* rand(rng,N) #dend memb potential variable
    spiked = zeros(Bool,N)
    back_prop = zeros(Bool,N)
    gain = zeros(N)
    P_in = zeros(n_in,N)
    Y = zeros(N)
    μ = zeros(N)
    P_r = zeros(N,N)
    At_in = zeros(n_in,N)
    At_r = zeros(N,N)
    circ_recmats = zeros(Bool,N,N,Tp+1)
    circ_affmats = zeros(Bool,n_in,N,Tp+1)
    PI_in = zeros(n_in) # instantaneous plasticity induction (gain) vector (aff. in)
    PI_r = zeros(N) # instantaneous plasticity induction (gain) vector (recurr. in)

    #summed input of incoming spikes
    g_E = zeros(N) #excitatory
    g_I = zeros(N) #inhibitory
    g_C = zeros(N)
    g_E_prev = zeros(N) #previous timestep
    g_I_prev = zeros(N)
    g_csd_prev = zeros(N)

    #difference of exponentials
    xe_rise = zeros(N) #excitatory rise exp
    xe_decay = zeros(N) #excitatory decay exp
    xi_rise = zeros(N) #inhibitory rise exp
    xi_decay = zeros(N) #inhibitory decay exp
    xcsd_rise = zeros(N)
    xcsd_decay = zeros(N)

    o_v = zeros(N,sim_steps)
    o_v_dend = zeros(N,sim_steps)
    o_pi = zeros(N,sim_steps)
    o_gcsd = zeros(N,sim_steps)
    o_ge = zeros(N,sim_steps)
    o_gi = zeros(N,sim_steps)

    p = Progress(sim_steps, dt=0.5,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        barlen=10,showspeed=true)

    #main simulation
    for ti = 1:sim_steps
        t = round(sim_δt*ti,digits=2)
        ProgressMeter.next!(p)
        #initialization of conductances
        g_E .= 0
        g_I .= 0
        g_C .= 0
        #new afferent input
        if ti <= input_steps
            circ_affmats[:,:,end] .= copy(input_mat[:,ti])
        end
        for ci = 1:N
            if ti <= input_steps
                #external input
                for (index, active) in enumerate(circ_affmats[:,ci,end])
                    if active
                        if rand(rng) > p_fail
                            if w_in[index,ci] < 0
                                g_I_prev[ci] -= w_in[index,ci]
                            else
                                g_E_prev[ci] += w_in[index,ci]
                            end
                        else
                            circ_affmats[index,ci,end] = 0 # remove syn.
                        end
                    end
                end
                #recurrent input addressed post-spike
            end
            g_I_prev[ci] *= κ[ci]
            g_E_prev[ci] *= κ[ci]
            #double-exp-kernel update
            xe_rise[ci] += -sim_δt*xe_rise[ci]/τe_rise + g_E_prev[ci]
            xe_decay[ci] += -sim_δt*xe_decay[ci]/τe_decay + g_E_prev[ci]
            xi_rise[ci] += -sim_δt*xi_rise[ci]/τi_rise + g_I_prev[ci]
            xi_decay[ci] += -sim_δt*xi_decay[ci]/τi_decay + g_I_prev[ci]
            xcsd_rise[ci] += -sim_δt*xcsd_rise[ci]/τcsd_rise + g_csd_prev[ci];
            xcsd_decay[ci] += -sim_δt*xcsd_decay[ci]/τcsd_decay + g_csd_prev[ci];
            #coupling adaptation
            gcsd = g_csdr + Ncsd*(xcsd_decay[ci] - xcsd_rise[ci])
            #synaptic input
            ge = g_e*Ne*(xe_decay[ci] - xe_rise[ci])
            gi = g_i*Ni*(xi_decay[ci] - xi_rise[ci])
            #back-prop current
            if t > (last_spike[ci] + τ_ref) # not in refractory
                Icsd = g_csdr*(v[ci] - v_dend[ci])
            else
                Icsd = gcsd*(v[ci] - v_dend[ci])
            end
            #dendritic m.p. update
            dv_dend = (ge*(Ve_r - v_dend[ci]) + gi*(Vi_r - v_dend[ci]) + g_ld*(Vd_l - v_dend[ci]) + Icsd)/C_dend
            v_dend[ci] += sim_δt*dv_dend
            #somatic m.p. update
            if t > (last_spike[ci] + τ_ref) # not in refractory
                dv = (g_ls*(Vs_l - v[ci]) + g_cs*(v_dend[ci] - v[ci]))/C[ci]
                v[ci] += sim_δt*dv
                spiked[ci] = v[ci] > Vth[ci]
            else
                dv = (g_lsr*(Vs_l - v[ci]))/C[ci]
                v[ci] += sim_δt*dv
            end
            #spike dynamics
            St = Int(spiked[ci])
            dY = (τst*St-Y[ci])/τst
            Y[ci] += sim_δt*dY
            if spiked[ci]
                spiked[ci] = false
                v[ci] = V_peak
                last_spike[ci] = t
                ns[ci] += 1
                (ns[ci] <= max_spikes) && (spks[ci,ns[ci]] = t) # save spike
                back_prop[ci] = true
                #spike propagation
                if ci > n_exc #inhibitory cell
                    for (j,w) in zip(findnz(w_r[ci,:])...)
                        g_I[j] -= w
                        circ_recmats[ci,j,end] = true #unused; algorithmically slower with same result
                    end
                else #excitatory synapses
                    for (j,w) in zip(findnz(w_r[ci,:])...)
                        g_E[j] += w
                        circ_recmats[ci,j,end] = true
                    end
                end #end loop over synaptic projections
            end #end if(spiked)
            #average somatic activity integration
            dμ = (Y[ci] - μ[ci])/τμ
            μ[ci] += sim_δt*dμ
            #gain change in Ca concentration and buffering dynamics
            gain[ci] = Y[ci] - μ[ci]
            ## <<  afferent input plastic update >>
            fill!(PI_in,0)
            #inputs trace decays
            dAt_in = @. -At_in[:,ci]/τAt
            newin = Float64.(view(circ_affmats,1:n_in,ci,Tp))
            At_in[:,ci] = @. At_in[:,ci] + newin + sim_δt*dAt_in
            #unraveling of input plastic inductions
            #PI_in = any(view(circ_affmats,1:n_in,ci,1:Tp),dims=2)
            PI_in = @. abs(w_in[:,ci] * At_in[:,ci])
            PI_in = @. PI_in * ζ(abs.(w_in[:,ci]),θζ,τζ)
            PI_in = @. gain[ci] * PI_in
            #low-pass filter for PIs
            dP = (PI_in .- P_in[:,ci])/τΔ
            P_in[:,ci] += sim_δt .* dP
            #plastic boundaries
            P_in[:,ci] = P_in[:,ci] .* zero_bounding(w_in[:,ci],η,P_in[:,ci]) # zero limits
            #plastic update
            w_in[:,ci] += η*P_in[:,ci]
            ## << recurrent input plastic update >>
            if rec_plastic
                fill!(PI_r,0)
                #inputs trace decays
                dAt_r = @. -At_r[:,ci]/τAt
                newr = Float64.(view(circ_recmats,1:N,ci,Tp))
                At_r[:,ci] = @. At_r[:,ci] + newr + sim_δt*dAt_r
                #unraveling of recurrent plastic inductions
                #PI_r = any(view(circ_recmats,1:N,ci,1:Tp),dims=2)
                PI_r = @. w_r[:,ci] * At_r[:,ci]
                PI_r = @. PI_r * ζ(abs.(w_r[:,ci]),θζ,τζ) # WA: plot ((x - 1e-4)/0.75)*exp(-(x-1e-4-0.75)/0.75) from x=0 to x=8
                PI_r = @. gain[ci] * PI_r
                #low-pass filter for PIs
                dP = (PI_r .- P_r[:,ci])/τΔ
                P_r[:,ci] += sim_δt .* dP
                #plasticity boundaries
                P_r[:,ci] = P_r[:,ci]  .* zero_bounding(w_r[:,ci],η,P_r[:,ci])
                #plastic update
                w_r[:,ci] += η_r*P_r[:,ci]
            end
            ## << homeostatic plastic update >>
            if (ti%Tϐ) == 0.0 && back_prop[ci] #adjust homeostatic when backprop occurs every Tϐ
                newϐ = sum((abs.(w_in[:,ci]))) + sum(abs.(w_r[:,ci]),dims=1)[1]
                κ[ci] *= 1-((κ[ci]*newϐ)-κ[ci]*ϐ[ci])/(κ[ci]*ϐ[ci])
                ϐ[ci] = newϐ
            end
            ## << backwards conductance modulation >>
            if back_prop[ci] && t > (last_spike[ci] + τ_bp) # back-prop instance
                g_C[ci] += mbp[ci]
                back_prop[ci] = false
            end
            o_gcsd[ci,ti] = gcsd
            o_ge[ci,ti] = ge
            o_gi[ci,ti] = gi
        end#end loop over network neurons (below lines affect next timestep)
        dropzeros!(w_r) # safety check
        #recording variables
        o_v[:,ti] = v
        o_v_dend[:,ti] = v_dend
        o_pi[:,ti] = gain
        #pat activity syn weight change
        ((ti%Tsample) == 0.0) && (push!(o_Pr,deepcopy(w_r)))
        ((ti%Tsample) == 0.0) && (push!(o_Paf,deepcopy(w_in)))
        #copy new input (recurrent)
        g_E_prev = copy(g_E)
        g_I_prev = copy(g_I)
        g_csd_prev = copy(g_C)
        #rotating for new input matrices
        circ_affmats = circshift(circ_affmats,(0,0,-1))
        circ_affmats[:,:,end] .= false
        circ_recmats = circshift(circ_recmats,(0,0,-1))
        circ_recmats[:,:,end] .= false
    end #end loop over time
    ProgressMeter.finish!(p)
    #prepare output variables
    spks = circshift(spks,(0,-1))
    spks[:,end] .= 0.0
    ns = ns .- 1
    push!(o_Pr,deepcopy(w_r))
    push!(o_Paf,deepcopy(w_in))

    return spks, ns, o_v_dend, o_v, o_pi, (o_ge, o_gi, o_gcsd), κ, (o_Paf, o_Pr, csid), (input_mat,input_pat,pats)
end
