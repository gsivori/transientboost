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

using LinearAlgebra
using StatsBase
using SparseArrays
using DataStructures
using Distributions
using Plots
using StableRNGs

#quick functions
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
tosteps(a) = Int(round(a * sec_to_ms / sim_δt))
totime(a) = a * ms_to_sec * sim_δt

import Base: zero

function zero(::Type{Vector{Int64}})
    return Vector{Int64}()
end

#Simple helper function to sort by row depending on first spike.
function first_true_index(row)
    idx = findfirst(identity, row)
    return isnothing(idx) ? Inf : idx
end
#e.g.
#perm = sortperm(1:size(pats[1], 1), by = row -> first_true_index(pats[1][row, :]))
#sorted_pats1 = pats[1][perm, :]

function bool_matrix_to_sparse_vector(bool_matrix::Matrix{Bool})
    size_in = size(bool_matrix, 2)
    sparse_vector = Vector{Vector{Int}}(undef,size_in)
    for i in 1:size_in
        sparse_vector[i] = findall(x -> x, bool_matrix[:, i])
    end

    return sparse(sparse_vector)
end

"""
    zscorer(weights::Vector{Float64},sim_δt::Float64=0.1)

Slowly returns weights back to baseline Gaussian values.

# Inputs:
- `weights::Vector{Float64}`: vector of synaptic weights.
- `τ_z::Float64`: timescale of return (in ms).
- `μ::Float64=0.0`: mean of baseline Gaussian.
- `σ::Float64=1.0`: standard deviation of baseline Gaussian.
- `sim_δt::Float64=0.1`: simuation time step (in ms).
"""
function zscorer(weights::Vector{Float64}, τ_z::Float64, sim_δt::Float64=0.1)
    zs = copy(weights)
    signs = sign.(weights)
    μ_zs = mean(zs)
    σ_zs = std(zs)

    target_zs = @. ((zs - μ_zs) / σ_zs )

    dzs = @. (target_zs - zs) * abs.(zs) / τ_z

    zs_weights = @. (zs + dzs * sim_δt )

    return zs_weights
end


"""
    pfail(weight::Float64)

Probability of transmission failure computation.

# Inputs:
- `weight::Float64`: synaptic weight.
"""
function pfail(weight::Float64)
    pf = 1-0.8/(1+exp(-2(abs(weight)-1)))
    return pf
end

"""
    get_cluster_sizes(rng::StableRNG,n_exc::Int64, μ_cs::Int64, σ_cs::Int64, min_cs::Int64=9)

Samples different cluster sizes to fill up the network connectivity.

# Inputs:
- `rng::StableRNG`: Stable random number generator variable.
- `n_exc::Int64`: number of excitatory cells in network.
- `μ_cs::Int64`: average cluster size.
- `σ_cs::Int64`: standard deviation of cluster sizes.
- `min_cs::Int64`: minimal bound for cluster size.
"""
function get_cluster_sizes(rng::StableRNG,n_exc::Int64, μ_cs::Int64, σ_cs::Int64, min_cs::Int64=14)
    remain = n_exc
    cs = []
    while remain > min_cs
        new_cs = Int(ceil(rand(rng,Normal(μ_cs,σ_cs))))
        remain -= new_cs
        append!(cs,new_cs)
    end
    if remain <= min_cs
        cs[end] += remain
    else
        append!(cs,remain)
    end
    return cs
end

"""
    zero_bounding(w::AbstractArray, eta::Float64, PI::AbstractArray)

Detects changes of signs for enforcing Dale's Law.

# Inputs:
- `w::AbstractArray`: synaptic weight structure.
- `eta::Float64`: learning rate.
- `PI::AbstractArray`: plasticity induction vector.
"""
function zero_bounding(w::AbstractArray, eta::Float64, PI::AbstractArray)
    return any((sign.(w .+ eta*PI)  .* sign.(w)) .== 1.0,dims=2)
end

"""
    jitter_data(data::AbstractArray, jitter::Float64, sim_δt::Float64)

Performs jittering of {data} by uniformly sampling {jitter} steps each input.

# Inputs:
- `S::AbstractArray`: input boolean vector of spikes.
- `jitter::Float64`: jittering time (in ms).
- `sim_δt`: simulation time step (in ms).
"""
function jitter_data(data::AbstractArray, jitter::Number, sim_δt::Float64)
    jit_steps = Int(jitter/sim_δt)
    (n_in, sim_len) = size(data)
    shuffled_input = zeros(eltype(data),(n_in, sim_len))
    for ci = 1:n_in
        times = view(data,ci,:)
        for (index,each) in enumerate(times)
            if each != 0
                adj = rand(-jit_steps:jit_steps)
                if (index + adj < 1) || (index + adj > sim_len)
                    adj = -adj # fastest way
                end
                shuffled_input[ci,index+adj] = 1
            end
        end
    end
    return shuffled_input
end

"""
    jitter_data(data::AbstractArray, jitter::Float64, sim_δt::Float64)

Performs jittering of {data} by uniformly sampling {jitter} steps each input.

# Inputs:
- `S::SparseMatrixCSC{Bool, Int64}`: input boolean sparse matrix.
- `jitter::Float64`: jittering time (in ms).
- `sim_δt`: simulation time step (in ms).
"""
function jitter_data(data::SparseMatrixCSC{Bool, Int64}, jitter::Number, sim_δt::Float64)
    jit_steps = Int(jitter/sim_δt)
    rows = Int64[]
    cols = Int64[]
    vals = Bool[]

    for ci = 1:data.m
        times,_ = findnz(data[ci,:])
        adj = rand(-jit_steps:jit_steps,length(times))
        aux = findall(times[1:5] .+ adj[1:5] .< 1)
        adj[aux] = -adj[aux]
        aux = findall(times[end-4:end] .+ adj[end-4:end] .> data.n) .- 5
        adj[length(adj) .+ aux] *= -1
        times += adj
        append!(cols, times)
        append!(vals, trues(length(times)))
        append!(rows, ci*ones(length(times)))
    end
    return sparse(rows, cols, vals, data.m, data.n)
end


"""
    NA_rates(v::Float64)

Returns rate parameters for sodium current. HH model.
    Ref: Dayan P, Abbott LF (2001) Theoretical Neuroscience; (Eqs. 5.24)
# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function NA_rates(v::Float64)
    mα =  0.1*(v+40)/(1-exp(-(v+40)/10))
    mβ =  4*exp(-(v+65)/18)
    m_inf = mα/(mα + mβ)
    τ_m = 1/(mα + mβ)
    hα =  0.07*exp(-(v+65)/20)
    hβ =  1/(1+exp(-(v+35)/10))
    h_inf = hα/(hα + hβ)
    τ_h = 1/(hα + hβ)
    return m_inf, τ_m, h_inf, τ_h
end

"""
    K_rates(v::Float64)

Returns rate parameters for potassium current. HH model.
    Ref: Dayan P, Abbott LF (2001) Theoretical Neuroscience; (Eqs. 5.22)

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function K_rates(v::Float64)
    nα =  0.01*(v+55)/(1-exp(-(v+55)/10))
    nβ =  0.125*exp(-(v+65)/80)
    n_inf = nα/(nα + nβ)
    τ_n = 1/(nα + nβ)
    return n_inf, τ_n
end

"""
    NAP_rates(v::Float64)

Returns rate parameters for persistent sodium current.
    Ref: Lipowsky et al, J.Neurophys 76:2181-2191, 1996.
    https://journals.physiology.org/doi/epdf/10.1152/jn.1996.76.4.2181

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function NAP_rates(v::Float64)
    mα = -1.74*(v-11)/(exp(-(v-11)/12.94)-1)
    mβ = 0.06*(v-5.9)/(exp((v-5.9)/4.47)-1)
    m_inf = 1/(1+exp(-(v-(-49.))/5))
    m_exp = 1 - exp(-0.1*(mα+mβ))
    
    return m_inf, m_exp
end

"""
    Ka_rates(v::Float64)

Returns rate parameters for A-type potassium current (proximal).
    Ref: Migliore et al 1999;

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function Ka_rates(v::Float64)
    qt = 1^((35-24)/10) # at 35 C
    zeta = -1.5 + -1/(1+exp((v-(-40))/5))
    nα =  exp(1.e-3*zeta*(v-11)*9.648e4/(8.315*(273.16+35))) 
    nβ =  exp(1.e-3*zeta*0.55*(v-11)*9.648e4/(8.315*(273.16+35)))
    lα = exp(1.e-3*3*(v-(-56))*9.648e4/(8.315*(273.16+35))) 
    lβ = exp(1.e-3*3*1*(v-(-56))*9.648e4/(8.315*(273.16+35))) 
    n_inf = 1/(1 + nα)
    τ_n = nβ/(qt*0.05*(1+nα))
    l_inf = 1/(1 + lα)
    τ_l = lβ/(qt*0.05*(1+lα))#0.26*(v+50)/1
    return n_inf, τ_n, l_inf, τ_l
end


"""
    H_rates(v::Float64)

Returns rate parameters for Ih current.
    Ref: Welie et al. (2006)

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function H_rates(v::Float64)
    qt=4.5^((35-33)/10) #35 celcius
    lα = exp(0.0378*2.2*(v+75))
    lβ = exp(0.0378*2.2*0.4*(v+75))
    linf = 1/(1+exp(-(v-(-73))/-8)) # -81 in code
    τ_l = lβ/(1.0*qt*0.011*(1+lα))
    #denom =  1/(1+exp((v+73)/8)) in paper
    #τ_l = 182*exp((v+75)/30.1)/(1+exp((v+75)/12)) in paper
    return linf, τ_l
end

"""
    Kdr_rates(v::Float64)

Returns rate parameters for delayed rectifying potassium current.
    Ref: Migliore et al 1999;

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function Kdr_rates(v::Float64)
    qt = 1^((35-24)/10) # at 35 C
    nα =  exp(1.e-3*-3*(v-13)*9.648e4/(8.315*(273.16+35))) 
    nβ =  exp(1.e-3*-3*0.7*(v-13)*9.648e4/(8.315*(273.16+35))) 
    n_inf = 1/(1 + nα)
    τ_n = nβ/(qt*0.02*(1+nα))
    return n_inf, τ_n
end





"""
    M_rates(v::Float64)

Returns rate parameters for M- potassium current.
    Ref: Yamada, W.M., Koch, C. and Adams, P.R.  Multiple 
    channels and calcium dynamics.  In: Methods in Neuronal Modeling, 
    edited by C. Koch and I. Segev, MIT press, 1989, p 97-134.

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function M_rates(v::Float64)
    tadj = 2.3 ^ ((35-36)/10) #35 celcius
	τ_peak = 1000. / tadj #in ms
	m_inf = 1 / ( 1 + exp(-(v+35)/10) )
	τ_m = τ_peak / ( 3.3 * exp((v+35)/20) + exp(-(v+35)/20) )
    
    return m_inf, τ_m
end

"""
    HVAc_rates(v::Float64)

Returns rate parameters for HVA calcium current.
    Refs: Reuveni et al., 1993; Mainen and Sejnowski, 1996.

# Inputs:
- `v::Float64`: membrane potential (in mV).
"""
function HVAc_rates(v::Float64)
    mα =  (0.055*(-27-v))/(exp((-27-v)/3.8) - 1)
    mβ  =  (0.94*exp((-75-v)/17))
    m_inf = mα/(mα + mβ)
    τ_m = (1/(mα + mβ))
    hα =  (0.000457*exp((-13-v)/50))
    hβ  =  (0.0065/(exp((-v-15)/28)+1))
    h_inf = hα/(hα + hβ)
    τ_h = (1/(hα + hβ))
    return m_inf, τ_m, h_inf, τ_h
end

"""
    KCa_rates(Ca::Float64)

Returns rate parameters for K+ calcium-dependent current.
    Refs: Mainen and Sejnowski, 1996.

# Inputs:
- `v::Float64`: membrane potential (in mV).
#Typically:
nk_inf, nk_exp, tadj = Kv_rates(Ca)
nk = nk + nexp*(nk_inf - nk)
gk = ghat_k*tadj*nk
Ik = gk*(Vk_r - v_dend) ; Vk_r = -75.0
"""
function KCa_rates(Ca::Float64)
    α =  0.01*Ca^1.0 
    β  =  0.02
    tadj = 2.72 #2.3^((35-23)/10)

    n_inf = α/(α + β)
    τ_n = (1/tadj/(α + β))

    return n_inf, τ_n, tadj
end

"""
    Mg_block(v::Float64, Mg::Float64)

Returns NMDA channels open probability.
Ref: Dayan P, Abbott LF (2001) Theoretical Neuroscience;
# Inputs:
- `v::Float64`: membrane potential (in mV).
- `Mg::Float64`: Magnesium concentration (in mM).
"""
function Mg_block(v::Float64, Mg::Float64)
    return (1+exp(-0.062*v)*(Mg/3.57))^-1
end

"""
    S(x::Float64,β::Float64,α::Float64=1.0,θ::Float64=0.5)

Returns the activation for parameter x.

# Inputs:
- `x::Float64`: voltage trace (mV).
- `β::Float64`: TBW.
- `α::Float64=1.0`: TBW.
- `θ::Float64=0.5`: TBW.
"""
function S(x::Float64,β::Float64,α::Float64=1.0,θ::Float64=0.5)
    # Sigmoidal activation function
    return (1+α*exp(β*(-x+θ)))^-1
end

"""
    moving_average(A::AbstractArray, m::Int)

Returns the moving average of array A with sliding window m (number of points).
"""
function moving_average(A::AbstractArray, m::Int)
    out = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = m÷2 * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            s += A[J]
            n += 1
        end
        out[I] = s/n
    end
    return out
end

"""
    ζ(x::Float64,θ::Float64=0.1,τ::Float64=1.0)

Returns scaled weight based on Alpha function. This function is inspired from spine head volume distributions.

# Inputs:
- `x::Float64`: synaptic weight (a.u.).
- `θ::Float64=0.1`: weight threshold, i.e, x-axis crossing.
- `τ::Float64=1.0`: Alpha function time constant.
"""
function ζ(x::Float64,θ::Float64=0.1,τ::Float64=1.0)
    # alpha function
    return ((x - θ)/τ)*exp(-(x-θ-τ)/τ)
end

function coef_var(x::AbstractVector)
    #Estimate coefficient of variation from data vector x with a sliding window.
    return sqrt(var(x[.~iszero.(x)]))/mean(x[.~iszero.(x)])
end

function cv_estimate(x::AbstractVector)
    #Estimate coefficient of variation from data vector x.
    #Assumes x is log-normally distributed.
    return sqrt(exp(var(log.(x[.~iszero.(x)])))-1)
end

function clip(x::AbstractVector, min::Float64, max::Float64)
    #Clamps a vector of values between a range(min,max).
    clamped_x = x[x .> min]
    return clamped_x[clamped_x .< max]
end

function last_input(S::Array{Bool,2})
    #Return last spike that ocurred (id,time in steps)
    (n,tend) = size(S)
    last_input = (1,1)
    for ti = tend:-1:1
        if any(S[:,ti] .== 1.0)
            for ci in 1:n
                if S[ci,ti]
                    which = (ci,ti)
                    return which
                end
            end
        end
    end
    return last_input
end

"""
    groupbypat(data::Vector{Any}, y_id::Vector{Any}, by::Vector{Any}, sim_δt::Float64)

Returns grouping parameters for patterns in structured input.

# Inputs:
- `data::Vector{Any}`: vector of all spike times.
- `y_id::Vector{Any}`: vector of indices for spike times on data vector.
- `by::Vector{Any}`: group id in an indexed vector representing pattern.
- `sim_δt::Float64`: simulation time step (in ms).
"""
function groupbypat(data::Vector{Any}, y_id::Vector{Any}, by::Vector{Any}, sim_δt::Float64)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    for (index,(pat,pat_time)) in enumerate(by)
        grouping[xs .>= pat_time*sim_δt] .= pat
    end
    return xs, ys, grouping
end

"""
    groupybypat(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)

Returns grouping parameters for patterns in structured input.

# Inputs:
- `xs::Vector{T}`: spike times in an indexed vector.
- `ys::Vector{T}`: neuron ids in an indexed vector.
- `by`: group id in an indexed vector representing pattern.
- `sim_δt`: simulation time step (in ms).
"""
function groupbypatsyns(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, pat_syns::Tuple, pat_width::Int64, sim_δt::Float64)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    for (index, (pat, pat_time)) in enumerate(by)
        if pat == 0
            grouping[xs .>= pat_time*sim_δt] .= pat
        else
            grouping[(indexin(ys, pat_syns[pat]) .!= nothing) .== (xs .>= pat_time*sim_δt) .== ((xs .<= (pat_time+pat_width)*sim_δt))] .= pat
        end
    end
    return xs, ys, grouping
end

"""
    groupybynat(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)

Returns grouping parameters for patterns in structured input.

# Inputs:
- `xs::Vector{T}`: spike times in an indexed vector.
- `ys::Vector{T}`: neuron ids in an indexed vector.
- `by`: synaptic structure.
- `sim_δt`: simulation time step (in ms).
"""
function groupbynat(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    for (index,sign) in enumerate(sign.(by))
        if sign == -1 #inhibitory
            grouping[ys .== index] .= 0
        else
            grouping[ys .== index] .= 1
        end
    end
    return xs, ys, grouping
end

"""
    groupybynat(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)

Returns grouping parameters for patterns in structured input.

# Inputs:
- `xs::Vector{T}`: spike times in an indexed vector.
- `ys::Vector{T}`: neuron ids in an indexed vector.
"""
function groupbynat(data::AbstractArray, y_id::AbstractArray)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    grouping[xs .< 0.0] .= 1
    return xs, ys, grouping
end

"""
    groupbynat(data::AbstractArray, y_id::AbstractArray, exc_n::Int64)

Returns grouping parameters for exc/inh raster plot.

# Inputs:
- `data::Vector{T}`: spike times in an indexed vector.
- `ys::Vector{T}`: neuron ids in an indexed vector.
- `exc_n`: number of excitatory cells.
"""
function groupbynat(data::AbstractArray, y_id::AbstractArray, exc_n::Int64)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    grouping[ys .> exc_n] .= 1
    return xs, ys, grouping
end

"""
    compute_fr(S::AbstractArray, sim_length::Float64, tbin::Float64=100.0)

Returns estimated firing rates (tbin=100.0ms.) of a single cell.

# Inputs:
- `S::AbstractArray`: vector of spike times of a single cell.
- `sim_length::Float64`: simulation time length.
- `tbin::Float64`: sliding window time length.
"""
function compute_fr(S::AbstractArray, sim_length::Float64, tbin::Float64=100.0)
    frs = zeros(Int(sim_length/tbin)-1)
    for fr in collect(1:length(frs))
        count = 0
        for (id,sp) in enumerate(S)
            if sp > (fr-1)*tbin
                if sp < fr*tbin
                    count += 1
                else
                    frs[fr] = count/tbin*1000.0
                    break
                end
            end
        end
    end
    return Array{Float64}(frs)
end

"""
    get_act_neurs(S::AbstractArray, t::Float64, tbin::Float64=100.0)

Returns active neurons within a sliding window of time (tbin=5.0ms).

# Inputs:
- `S::AbstractArray`: vector of spike times of a single cell.
- `ti::Float64`: simulation time.
- `tbin::Float64`: sliding window time length.
"""
function get_act_neurs(S::AbstractArray, t::Float64, tbin::Float64=5.0)
    act_neurs = falses(size(S)[1])
    for ci in collect(1:length(act_neurs))
        for (id,st) in enumerate(S)
            if st > t-tbin
                if st < t
                    act_neurs[ci] = 1
                end
            else
                break
            end
        end
    end
    return Array{Float64}(act_neurs)
end

"""
    get_exc(W::AbstractArray, exc_n::Int64)

Returns overall excitability of each cell defined as the sum of positive synaptic input.

# Inputs:
- `W::AbstractArray`: recurrent matrix of synaptic connectivity.
- `exc_n::Int64`: number of excitatory cells in the network.
"""
function get_exc(W::AbstractArray, exc_n::Int64)
    exc = zeros(size(W)[1])
    for ci in collect(1:length(exc))
        exc[ci] = sum(W[1:exc_n,ci])
    end
    return Array{Float64}(exc)
end


"""
    get_connections(rng::StableRNG,x::Vector{T}, y::Vector{T}; n=length(x)*length(y)) where {T}

Returns the pairs of a given number of connections between two vectors x and y.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `x::Vector{T}`: first list of indices
- `y::Vector{T}`: second list of indices, starting at size(x)[1]+1.
- `n`: amount of connections.
"""
function get_connections(rng::StableRNG,x::Vector{T}, y::Vector{T}; n=length(x)*length(y)) where {T}
	n = round(Int64, n)
    xout = Vector{T}(undef,n)
    yout = Vector{T}(undef,n)
    p = length(x)*length(y)
    s = zeros(Int,n)
    StatsBase.knuths_sample!(rng,1:p,s)
    @inbounds idx = CartesianIndices((length(x),length(y)))[s];
    @inbounds for i = 1:n
        xout[i] = x[ idx[i][1] ]
        yout[i] = y[ idx[i][2] ]
    end
    return xout, yout
end


"""
    all_to_all(x::Vector{T}, y::Vector{T}; n=length(x)*length(y)) where {T}

Returns the pairs of a given number of connections between two vectors x and y.

# Inputs:
- `x::Vector{T}`: first list of indices
- `y::Vector{T}`: second list of indices.
- `n`: amount of connections.
"""
function all_to_all(x::Vector{T}, y::Vector{T}; n=length(x)*length(y)) where {T}
    n = round(Int64, n)
    xout = Vector{T}(undef,n)
    yout = Vector{T}(undef,n)
    xout = repeat(x,length(y))
    for j = 1:length(y)
        yout[1+(j-1)*length(x):j*length(x)] .= y[j]
    end
    return xout, yout
end

"""
get_dist_clusters_conns(rng::StableRNG,x::Vector{T}, y::Int64, pop_sizes::Array{Int64,1}, conn_probs::Array{Float64,1}) where {T}

	Returns the pairs of a given number of clustered connections (defined by a distribution) in a network of x cells.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `x::Vector{T}`: list of cells.
- `y::Int64`: starting index of inh cells.
- `pop_sizes::Array{Int64,1}`: array of clustered connections sizes.
- `conn_probs::Array{Float64,1}`: probability of clustered connections.
- `μ::Array{Float64,1}`: means of synaptic weight distribution (LogNormal).
- `σ::Array{Float64,1}`: standard deviations of synaptic weight distribution (LogNormal).
"""
function get_dist_clusters_conns(x::Vector{T}, y::Int64, pop_sizes::Array{Int64,1}, conn_probs::Array{Float64,1}, μ::Array{Float64,1}, σ::Array{Float64,1}) where {T}
    n = length(pop_sizes) # total number of clusters
    n_inh = Int.(ceil.(pop_sizes .* .1)) # number of inh cells in each cluster
    p = [pop_sizes .* pop_sizes, pop_sizes .* n_inh, pop_sizes .* n_inh, n_inh .* n_inh] # total number of connections x->y
    nee_ids = Int.(ceil.(p[1] .* conn_probs[1]))
    nei_ids = Int.(ceil.(p[2] .* conn_probs[2]))
    nie_ids = Int.(ceil.(p[3] .* conn_probs[3]))
    nii_ids = Int.(ceil.(p[4] .* conn_probs[4]))
    n_ids = [nee_ids, nei_ids, nie_ids, nii_ids]
    pops = length(n_ids)
    s = [zeros.(Int,conns) for conns in n_ids]
	xout = Vector{T}(undef,0)
	yout = Vector{T}(undef,0)
	zout = Vector{Float64}(undef,0)
	idx = CartesianIndices((length(x), length(x))) # all indices
    pope_start = pope_end = x[1]
    popi_start = popi_end = Int(y)
	for pop_i = 1:n
		pope_start = pope_end
        pope_end = pope_start + pop_sizes[pop_i]-1
        popi_start = popi_end
        popi_end = popi_start + n_inh[pop_i]-1
        for conn in 1:pops
            if ~isempty(1:p[conn][pop_i])
                StatsBase.knuths_sample!(rng,1:p[conn][pop_i],s[conn][pop_i])
            end
        end
        for i = 1:n_ids[1][pop_i] #exc->exc
            push!(xout, x[ idx[pope_start:pope_end,pope_start:pope_end][s[1][pop_i]][i][1] ])
			push!(yout, x[ idx[pope_start:pope_end,pope_start:pope_end][s[1][pop_i]][i][2] ])
            weight = rand(rng,Normal(μ[1],σ[1]))
            weight = weight * sign(weight)
			push!(zout, weight)
        end
        for i = 1:n_ids[4][pop_i] #inh->inh
            push!(xout, x[ idx[popi_start:popi_end,popi_start:popi_end][s[4][pop_i]][i][1] ])
			push!(yout, x[ idx[popi_start:popi_end,popi_start:popi_end][s[4][pop_i]][i][2] ])
            weight = rand(rng,Normal(μ[4],σ[4]))
            weight = -1 * weight * sign.(weight)
			push!(zout, weight)
        end
        for i = 1:n_ids[2][pop_i] #exc->inh
            push!(xout, x[ idx[pope_start:pope_end,popi_start:popi_end][s[2][pop_i]][i][1] ])
			push!(yout, x[ idx[pope_start:pope_end,popi_start:popi_end][s[2][pop_i]][i][2] ])
            weight = rand(rng,Normal(μ[2],σ[2]))
            weight = weight * sign.(weight)
            push!(zout, weight)
        end
        for i = 1:n_ids[3][pop_i] #inh->exc
            push!(xout, x[ idx[popi_start:popi_end,pope_start:pope_end][s[3][pop_i]][i][1] ])
			push!(yout, x[ idx[popi_start:popi_end,pope_start:pope_end][s[3][pop_i]][i][2] ])
            weight = rand(rng,Normal(μ[3],σ[3]))
            weight = -1 * weight * sign.(weight)
            push!(zout, weight)
        end
	end
    return xout, yout, zout
end

"""
function get_clusters_conns(rng::StableRNG,x::Vector{T}, pop_sizes::Vector{Any}, conn_prob::Float64, μ::Float64, σ::Float64) where {T}

	Returns the pairs of a given number of clustered connections (defined by a distribution) in a network of x cells.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `x::Vector{T}`: list of cells.
- `y::Int64`: starting index of inh cells.
- `pop_sizes::Vector{Any}`: array of clustered connections sizes.
- `conn_probs::Array{Float64,1}`: probability of clustered connections.
- `μ::Array{Float64,1}`: means of synaptic weight distribution (LogNormal).
- `σ::Array{Float64,1}`: standard deviations of synaptic weight distribution (LogNormal).
"""
function get_clusters_conns(rng::StableRNG,x::Vector{T}, pop_sizes::Vector{Any}, conn_prob::Float64, μ::Float64, std::Float64) where {T}
    n = length(pop_sizes) # total number of clusters
    p = pop_sizes .* pop_sizes # total number of connections per cluster
    ids = Int.(ceil.(p .* conn_prob))
    s = [zeros.(Int,conns) for conns in ids]
    xout = Vector{T}(undef,0)
    yout = Vector{T}(undef,0)
    zout = Vector{Float64}(undef,0)
    idx = CartesianIndices((length(x), length(x))) # all indices
    pope_start = x[1]
    pope_end = 0
    csid = []
    for pop_i = 1:n
        pope_start = pope_end+1 # separate clusters
        pope_end += pop_sizes[pop_i]
        push!(csid, pope_start:pope_end)
        StatsBase.knuths_sample!(rng,1:p[pop_i],s[pop_i])
        for i = 1:ids[pop_i] #exc->exc
            push!(xout, x[ idx[pope_start:pope_end,pope_start:pope_end][s[pop_i]][i][1] ])
            push!(yout, x[ idx[pope_start:pope_end,pope_start:pope_end][s[pop_i]][i][2] ])
            weight = abs(rand(rng,Normal(μ,std)))
            push!(zout, weight)
        end
    end
    return xout, yout, zout, csid
end

"""
vspans(timestamps::Vector{Tuple{Float64, Float64}},pattern::Vector{T}) where {T}

	Plots vertical spans of input patterns on current plot.

# Inputs:
- `timestamps::Vector{T}`: array of start/finish timestamps.
- `pattern::Vector{T}`: array of pattern id and startime.
"""
function vspans(h::Plots.Plot{Plots.GRBackend}, timestamps::Vector{Tuple{Float64, Float64}}, pattern::Vector{T}, lims::Tuple{Float64, Float64},alpha::Float64=.05) where {T}
    patcl = palette(:default)
    patcl = patcl[1:maximum(pattern)[1]]
    tstart = lims[1]
    tend = lims[2]
    for (id, (t0, t1)) in enumerate(timestamps)
        (pat,_) = pattern[id]
        if pat > 0 && t0 > tstart && t1 < tend
            cl = patcl[pat]
            vspan!([t0,t1],fillalpha=alpha,color=cl,alpha=0.0,label=nothing,xlim=lims)
        end
    end
    plot!()
end


"""
vspans(timestamps::Vector{Tuple{Float64, Float64}},pattern::Vector{T}) where {T}

	Plots vertical spans of input patterns on h subplots.

# Inputs:
- `h::Plots.Plot{Plots.GRBackend}`: Plots figure.
- `timestamps::Vector{T}`: array of start/finish timestamps.
- `pattern::Vector{T}`: array of pattern id and startime.
"""
function vspans(h::Plots.Plot{Plots.GRBackend},whichones::UnitRange{Int64},timestamps::Vector{Tuple{Float64, Float64}}, pattern::Vector{T}, lims::Tuple{Float64, Float64},alpha::Float64=.05) where {T}
    patcl = palette(:default)
    patcl = patcl[1:maximum(pattern)[1]]
    subplots=length(h.subplots)
    tstart = lims[1]
    tend = lims[2]
    for (id, (t0, t1)) in enumerate(timestamps)
        (pat,_) = pattern[id]
        if pat > 0 && t0 > tstart && t1 < tend
            cl = patcl[pat]
            for i in whichones
                vspan!(h[i],[t0,t1],fillalpha=alpha,color=cl,alpha=0.0,label=nothing,xlim=lims)
            end
        end
    end
end

"""
vspans_states(h::Plots.Plot{Plots.GRBackend}, timestamps::Vector{Tuple{Float64, Float64, Symbol}}, lims::Vector{Float64},alpha::Float64=.05)

	Plots vertical spans of state patterns on h subplots.

# Inputs:
- `h::Plots.Plot{Plots.GRBackend}`: Plots figure.
- `timestamps::Vector{T}`: array of start/finish timestamps.
- `lims::Vector{Any}`: vector of xlims and ylims concatenated.
- `alpha::Float64`: transparency. 
"""
function vspans_states(h::Plots.Plot{Plots.GRBackend},whichones::UnitRange{Int64},timestamps::Vector{Tuple{Float64, Float64, Symbol}}, lims::Vector{Float64}, alpha::Float64=.05)
    patcl = palette(:lightrainbow)
    states = unique([state for (_,_,state) in timestamps])
    patcl = patcl[1:length(states)]
    subplots=length(h.subplots)
    tstart = lims[1]
    tend = lims[2]
    ystart = lims[3]
    yend = lims[4]
    cl = patcl[1]
    for (t0, t1, symb) in timestamps
        if t0 >= tstart && t1 <= tend
            for (id,state) in enumerate(states)
                if state == symb
                    cl = patcl[id]
                end
            end
            for i in whichones
                bar!(h[i],[(t1+t0)/2],[yend],bar_width=[t1-t0],fillalpha=alpha,color=cl,linealpha=0.0,linewidth=0.0,label="",xlim=lims[1:2])
                #vspan!(h[i],[t0,t1],fillalpha=alpha,color=cl,alpha=0.0,label=nothing,xlim=lims)
            end
        end
    end
    plot!()
end

"""
vspans_states(h::Plots.Plot{Plots.GRBackend}, timestamps::Vector{Tuple{Float64, Float64, Symbol}}, lims::Vector{Float64},alpha::Float64=.05)

	Plots vertical spans of state patterns on h subplots.

# Inputs:
- `h::Plots.Plot{Plots.GRBackend}`: Plots figure.
- `timestamps::Vector{T}`: array of start/finish timestamps.
- `lims::Vector{Any}`: vector of xlims and ylims concatenated.
- `alpha::Float64`: transparency. 
"""
function vspans_states(h::Plots.Subplot{Plots.GRBackend},whichones::UnitRange{Int64},timestamps::Vector{Tuple{Float64, Float64, Symbol}}, lims::Vector{Float64}, alpha::Float64=.05)
    patcl = palette(:lightrainbow)
    states = unique([state for (_,_,state) in timestamps])
    patcl = patcl[1:length(states)]
    tstart = lims[1]
    tend = lims[2]
    ystart = lims[3]
    yend = lims[4]
    cl = patcl[1]
    for (t0, t1, symb) in timestamps
        if t0 >= tstart && t1 <= tend
            for (id,state) in enumerate(states)
                if state == symb
                    cl = patcl[id]
                end
            end
            for i in whichones
                bar!(h[i],[(t1+t0)/2],[yend],bar_width=[t1-t0],fillalpha=alpha,color=cl,linealpha=0.0,linewidth=0.0,label="",xlim=lims[1:2])
                #vspan!(h,[t0,t1],fillalpha=alpha,color=cl,alpha=0.0,label=nothing,xlim=lims)
            end
        end
    end
    plot!()
end

"""
function vspans_reward(h::Plots.Plot{Plots.GRBackend},whichones::UnitRange{Int64},pattern_rewards::Vector{Tuple{Float64, Float64}},lims::Tuple{Float64, Float64},opacity=0.3)

	Plots vertical spans of reward area in plots.

# Inputs:
- `h::Plots.Plot{Plots.GRBackend}`: Plots figure.
- `pattern_rewards::Vector{Float64}`: vector of reward pattern times in seconds.
- `lims::Tuple{Float64, Float64}`: Figure x-axis limits for plotting.
"""
function vspans_reward(h::Plots.Plot{Plots.GRBackend},whichones::UnitRange{Int64},pattern_rewards::Vector{Tuple{Float64, Float64}},lims::Vector{Float64},opacity=0.3)
    clrew = palette(:lightrainbow)[5]
    tstart = lims[1]
    tend = lims[2]
    for (id, (t1, t2)) in enumerate(pattern_rewards)
        for i in whichones
            if t1 >= tstart && t2 <= tend
                vspan!(h[i],[t1,t2],fillalpha=opacity,color=clrew,alpha=0.0,label=nothing)
            end
        end
    end
    plot!()
end

"""
function bin_spikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64)

	Bins array of spikes for plotting.

# Inputs:
- `spikes::Array{Float64,1}`: array of spike times.
- `n_bins::Int64`: number of bins (boxes) for counting spikes.
- `box_width::Int64`: size of box in time steps.
- `sim_steps::Int64`: simulation time steps.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function bin_spikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64) where T<:Real
    k = zeros(n_bins) 
    pss = zeros(sim_steps)
    box_len = box_width*sim_δt
    for i = 1:n_bins
        k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
        pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
    end
    return pss
end

"""
function bin_netspikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64)

	Generates array of binned network spikes for plotting.

# Inputs:
- `spikes::Array{Float64,1}`: array of spike times.
- `n_bins::Int64`: number of bins (boxes) for counting spikes.
- `box_width::Int64`: size of box in time steps.
- `sim_steps::Int64`: simulation time steps.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function bin_netspikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64) where T<:Real
    N,_ = size(spikes)
    pss = zeros(sim_steps,N)
    for ci = 1:N
        pss[:,ci] = bin_spikes(spikes[ci,:],n_bins,box_width,sim_steps,sim_δt)
    end
    return pss
end

"""
function rasters(spikes::AbstractArray{T},nspikes::Vector{Int64}) where T<:Real

	Generates x and y coordinates of cell and spike time IDs for raster plotting.

# Inputs:
- `spikes::Array{Float64,1}`: array of spike times.
- `nspikes::Vector{Int64}`: number of spikes per cell.
"""
function rasters(spikes::AbstractArray{T},nspikes::Vector{Int64}) where T<:Real
    N,_ = size(spikes)
    vals = []
    y = []
    for ci = 1:N
        push!(vals,spikes[ci,1:nspikes[ci]]./1000.0) #raster plot in seconds.
        push!(y,ci*ones(length(spikes[ci,1:nspikes[ci]])))
    end
    return vals, y
end

"""
generate_input_pats(n_in::Int64, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64)

	Generates binary input matrix given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_pats(rng::StableRNG,n_in::Int64, pat_width::Int64, npat::Int64, poiss_rate::Float64, sim_δt::Float64)
    pats = []
    offset = 1
    for i in 1:npat
        push!(pats,zeros(Bool,(n_in,pat_width)))
    end
    for i in range(1,stop=n_in) # creates patterns of Poisson-like input
        for j in range(1,stop=pat_width-offset)
            for k in 1:npat #CHANGE LATER
                (rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)) && (pats[k][i,j] = 1)
            end
        end
    end
    return pats
end

"""
generate_single_input_pat(n_in::Int64, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64)

	Generates a single binary input matrix (pattern) given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_single_input_pat(rng::StableRNG,n_in::Int64, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64)
    pat = zeros(Bool,(n_in,pat_width))
    offset = 1
    for i in range(1,stop=n_in) # creates patterns of Poisson-like input
        for j in range(1,stop=pat_width-offset)
            (rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)) && (pat[i,j] = 1)
        end
    end
    return pat
end

"""
generate_input_patsV2(n_in::Int64, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64)

	Generates binary input matrix given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_patsV2(rng::StableRNG,n_in::Int64, pat_width::Int64, npat::Int64, poiss_rate::Float64, sim_δt::Float64)
    pats = []
    offset = 1
    for i in 1:npat
        push!(pats,zeros(Bool,(n_in,pat_width)))
    end
    for k in 1:npat
        for i in range(1,stop=n_in) # creates patterns of Poisson-like input
            for j in range(1,stop=pat_width-offset)
                (rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)) && (pats[k][i,j] = 1)
            end
        end
    end
    return pats
end


"""
generate_input_mat(n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)

	Generates binary input matrix given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `sim_steps::Int64`: simulation length in timesteps.
- `pats::Vector{Matrix{Bool}}`: vector of input patterns (boolean matrix).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_mat(rng::StableRNG,n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)
    input_mat = zeros(Bool,(n_in, sim_steps))
    input_pat = []
    chunk = 1
    while chunk < sim_steps # creates the input time matrix and the pattern timings
        rand_width = Int(ceil(rand(rng,Uniform(pat_width,3*pat_width))))
        if rand_width > (sim_steps - chunk)
            rand_width = sim_steps - chunk
        end
        rand_mat = zeros(Bool,(n_in,rand_width))
        for i in range(1,stop=n_in)
            for j in range(1,stop=rand_width-1)
                if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
                    rand_mat[i,j] = 1
                end
            end
        end
        input_mat[:,chunk:chunk+rand_width-1] =  rand_mat
        push!(input_pat,(0,chunk))
        chunk += rand_width
        if pat_width < (sim_steps - chunk)
            pat_dice = rand(rng)
            if pat_dice < 1/3
                input_mat[:,chunk:chunk+pat_width-1] = pats[1]
                push!(input_pat,(1,chunk))
            elseif pat_dice < 2/3
                input_mat[:,chunk:chunk+pat_width-1] = pats[2]
                push!(input_pat,(2,chunk))
            else
                input_mat[:,chunk:chunk+pat_width-1] = pats[3]
                push!(input_pat,(3,chunk))
            end
            chunk += pat_width
        end
    end
    return input_mat, input_pat
end


"""
generate_input_matV2(n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)

	Generates binary input matrix given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `sim_steps::Int64`: simulation length in timesteps.
- `pats::Vector{Matrix{Bool}}`: vector of input patterns (boolean matrix).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_matV2(rng::StableRNG,n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)
    input_mat = falses(n_in, sim_steps)
    input_pat = []
    dice = 1:size(pats)[1] # dice roll struct
    offset = 2
    chunk = 1
    while chunk < sim_steps # creates the input time matrix and the pattern timings
        rand_width = Int(ceil(rand(rng,Uniform(pat_width,3*pat_width))))
        (rand_width > (sim_steps - chunk)) && (rand_width = sim_steps - chunk)
        rand_mat = zeros(Bool,(n_in,rand_width))
        for i in range(1,stop=n_in)
            for j in range(1,stop=rand_width-offset)
                (rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)) && (rand_mat[i,j] = 1)
            end
        end
        input_mat[:,chunk:chunk+rand_width-1] =  rand_mat
        push!(input_pat,(0,chunk))
        chunk += rand_width
        if chunk > pat_width
            if pat_width < (sim_steps - chunk)
                pat = rand(rng,dice)
                input_mat[:,chunk:chunk+pat_width-1] = pats[pat]
                push!(input_pat,(pat,chunk)) #starting point of pat
                chunk += pat_width
            end
        end
    end
    return input_mat, input_pat
end

"""
generate_input_mat(rng::StableRNG,n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64, pat_ns::Vector{Vector{Int64}})

	Generates binary input matrix given sizes and vector of subsets neuron IDs for each pattern.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `sim_steps::Int64`: simulation length in timesteps.
- `pats::Vector{Matrix{Bool}}`: vector of input patterns (boolean matrix).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
- `pat_ns::Vector{Vector{Int64}}`: vector of subsets of neuron IDs for each pattern.
"""
function generate_input_mat(rng::StableRNG,n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64, pat_ns::Vector{Vector{Int64}})
    input_mat = zeros(Bool,(n_in, sim_steps))
    input_pat = []
    chunk = 1
    while chunk < sim_steps # creates the input time matrix and the pattern timings
        rand_width = Int(ceil(rand(rng,Uniform(pat_width,3*pat_width))))
        if rand_width > (sim_steps - chunk)
            rand_width = sim_steps - chunk
        end
        rand_mat = zeros(Bool,(n_in,rand_width))
        for i in range(1,stop=n_in)
            for j in range(1,stop=rand_width-1)
                if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
                    rand_mat[i,j] = 1
                end
            end
        end
        input_mat[:,chunk:chunk+rand_width-1] =  rand_mat
        push!(input_pat,(0,chunk))
        chunk += rand_width
        if pat_width < (sim_steps - chunk)
            pat_dice = rand(rng)
            if pat_dice < 1/3
                rand_ins = setdiff!(collect(1:n_in),pat_ns[1])
                input_mat[pat_ns[1],chunk:chunk+pat_width-1] = pats[1]
                push!(input_pat,(1,chunk))
            elseif pat_dice < 2/3
                rand_ins = setdiff!(collect(1:n_in),pat_ns[2])
                input_mat[pat_ns[2],chunk:chunk+pat_width-1] = pats[2]
                push!(input_pat,(2,chunk))
            else
                rand_ins = setdiff!(collect(1:n_in),pat_ns[3])
                input_mat[pat_ns[3],chunk:chunk+pat_width-1] = pats[3]
                push!(input_pat,(3,chunk))
            end
            for i in rand_ins
                for j in chunk:chunk+pat_width-1
                    if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
                        input_mat[i,j] = 1
                    end
                end
            end
            chunk += pat_width
        end
    end
    return input_mat, input_pat
end

"""
replace_input_pat_in_stream(input_mat::AbstractArray{Bool}, pat::Matrix{Bool}, pattern_times::Vector{Any}, pat_width::Int64, thisone::Int64, pat_percent::Int64, changepats_ti::Int64)

	Literally replaces part of a pattern of the input stream.

# Inputs:
- `input_mat::Matrix{Bool}`: random number generator from StableRNG package.
- `pat::Matrix{Bool}`: vector of input patterns (boolean matrix).
- `pattern_times::Vector{Any}`: pattern timings.
- `pat_width::Int64`: pattern width in timesteps.
- `thisone::Int64`: the pattern ID which will be replaced.
- `pat_percent::Int64`: pattern intrinsic timestamp. must be less than pat_width.
- `changepats_ti::Int64`: moment in timestep when pattern is replaced.
"""
function replace_input_pat_in_stream(input_mat::AbstractArray{Bool}, pat::Matrix{Bool}, pattern_times::Vector{Any}, pat_width::Int64, thisone::Int64, pat_percent::Int64,changepats_ti::Int64)
    input_stream = copy(input_mat)
    for (whichpat,timestamp) in pattern_times
        if whichpat == thisone && timestamp >= changepats_ti
            input_stream[:,timestamp+pat_percent:timestamp+pat_width] = pat[:,pat_percent:end]
        end
    end
    return input_stream
end

"""
generate_input_noisymat(rng::StableRNG,n_in::Int64, sim_steps::Int64, poiss_rate::Float64,sim_δt::Float64)

	Generates binary input matrix of Poisson spikes given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `sim_steps::Int64`: simulation length in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_noisymat(rng::StableRNG,n_in::Int64, sim_steps::Int64, poiss_rate::Float64,sim_δt::Float64)
    input_mat = zeros(Bool,(n_in, sim_steps))
    for i in range(1,stop=n_in)
        for j in range(1,stop=sim_steps-1)
            (rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)) && (input_mat[i,j] = 1)
        end
    end
    return input_mat
end


"""
generate_input_noisyvec(n_in::Int64, sim_steps::Int64, pats::Vector{Matrix{Bool}}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)

	Generates binary input vector of Poisson spikes at single time step.

# Inputs:
- `n_in::Int64`: input size (# of cells).
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_noisyvec(rng::StableRNG,n_in::Int64, poiss_rate::Float64,sim_δt::Float64)
    input_vec = zeros(Bool,n_in)
    for i in range(1,stop=n_in)
        if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
            input_vec[i] = 1
        end
    end
    return input_vec
end


"""
function optimal_bin_size(spikes::AbstractArray{T},sim_steps::Int64,sim_δt::Float64)

    Method for selecting bin size.

# Inputs:
- `spikes::Array{Float64,1}`: array of spike times.
- `sim_steps::Int64`: simulation time steps.
- `sim_δt::Float64`: simulation time constant (ms).

plot(buckets,C,linewidth=2.0,color=:black,label=L"C_Δ")
"""
function optimal_bin_size(spikes::AbstractArray{T},sim_steps::Int64,sim_δt::Float64) where T<:Real
    buckets = collect(10:10:2000)
    CΔ = zeros(length(buckets))
    pss = zeros(sim_steps)
    for (index,bucket) in enumerate(buckets)
        n_bins = Int(round(sim_steps/bucket,digits=0))
        k = zeros(n_bins)
        box_len = bucket*sim_δt
        for i = 1:n_bins
            k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
        end
        CΔ[index] = (2*mean(k)-var(k))/(bucket^2)
    end
    return buckets,CΔ
end

"""
function bin_spikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64)

	Generates array of binned spikes for plotting.

# Inputs:
- `spikes::Array{Float64,1}`: array of spike times.
- `n_bins::Int64`: number of bins (boxes) for counting spikes.
- `box_width::Int64`: size of box in time steps.
- `sim_steps::Int64`: simulation time steps.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function bin_spikes(spikes::AbstractArray{T},n_bins::Int64,box_width::Int64,sim_steps::Int64,sim_δt::Float64) where T<:Real
    k = zeros(n_bins) 
    pss = zeros(sim_steps)
    box_len = box_width*sim_δt
    for i = 1:n_bins
        k[i] = count(spk->((i-1)*box_len <= spk < (i)*box_len && spk != 0.0),spikes)
        pss[1+Int((i-1)*box_width):Int((i)*box_width)] .= k[i]
    end
    return pss
end



"""
generate_input_mat(n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64, sim_δt::Float64, pat_ns::Vector{Vector{Int64}})

	Generates binary input matrix given sizes and vector of subsets neuron IDs for each pattern.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_steps::Int64`: simulation length in timesteps.
- `pats::Vector{Any}`: vector of input patterns (spike matrix of each pattern).
- `pat_ids::Vector{Any}`: vector of input patterns ids (targets in MNIST dataset).
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_mnist_input(rng::StableRNG,sim_steps::Int64, pats::Array{Float64,3}, pat_ids::Vector{Int64}, poiss_rate::Float64, sim_δt::Float64) 
    n_in, npats, pat_width = size(pats)
    pat_id_array = sample(rng,collect(1:128),npats,replace=true)
    input_mat = zeros(Bool,(n_in, sim_steps))
    input_pat = [] # pattern order
    chunk = 1
    id = 1
    while chunk < sim_steps # creates the input time matrix and the pattern timings
        rand_width = Int(ceil(rand(rng,Uniform(pat_width,3*pat_width))))
        if rand_width > (sim_steps - chunk)
            rand_width = sim_steps - chunk
        end
        rand_mat = zeros(Bool,(n_in,rand_width))
        for i in range(1,stop=n_in)
            for j in range(1,stop=rand_width-1)
                if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
                    rand_mat[i,j] = 1
                end
            end
        end
        input_mat[:,chunk:chunk+rand_width-1] =  rand_mat
        push!(input_pat,(-1,chunk))
        chunk += rand_width
        if pat_width < (sim_steps - chunk)
            if id > size(pat_id_array)[1]
                append!(pat_id_array,sample(rng,collect(1:128),npats,replace=true))
            end
            input_mat[:,chunk:chunk+pat_width-1] = pats[:,pat_id_array[id],:]
            push!(input_pat,(pat_ids[pat_id_array[id]],chunk))
            #for i in range(1,stop=n_in)
            #    for j in range(chunk,stop=chunk+pat_width-1)
            #        if rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
            #            input_mat[i,j] = 1
            #        end
            #    end
            #end
            chunk += pat_width
            id = id + 1
        end
    end
    return input_mat, input_pat
end

"""
vspans_mnist(timestamps::Vector{Tuple{Float64, Float64}},pattern::Vector{T}) where {T}

	Plots vertical spans of input patterns on current plot.

# Inputs:
- `h::Plots.Plot{Plots.GRBackend}`: Plots figure.
- `timestamps::Vector{T}`: array of start/finish timestamps.
- `pattern::Vector{T}`: array of pattern id and startime.
- `start::Float64`: Figure x-axis start.
- `stop::Float64`: Figure x-axis end.
"""
function vspans_mnist(h::Plots.Plot{Plots.GRBackend}, timestamps::Vector{Tuple{Float64, Float64}}, pattern::Vector{T}, lims::Tuple{Float64, Float64}, α::Float64) where {T}
    patcl = palette(:default)
    patcl = patcl[1:maximum(pattern)[1]]
    subplots=length(h.subplots)
    start = lims[1]
    stop = lims[2]
    for (id, (t0, t1)) in enumerate(timestamps)
        (pat,_) = pattern[id]
        if t0 > start && t1 < stop
            if pat >= 0
                if pat == 0
                    cl = patcl[1]
                else
                    cl = patcl[pat]
                end
                for i in 1:subplots
                    vspan!(h[i],[t0,t1],fillalpha=α,color=cl,alpha=0.0,label=nothing)
                end
            end
        end
    end
    plot!()
end


"""
poisson_spike(rng::StableRNG,poiss_rate::Float64,sim_δt::Float64)

	Generates a spike from inhomogeneous Poisson process for a given time step.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function poisson_spike(rng::StableRNG,poiss_rate::Float64,sim_δt::Float64)
    return rand(rng) <= rand(rng)*poiss_rate*sim_δt*10^(-3)
end

"""
poisson_spike_vector(rng::StableRNG,poiss_rate::Float64,sim_δt::Float64)

	Generates a vector of spikes from inhomogeneous Poisson processes for a given time step.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function poisson_spike_vector(rng::StableRNG,n_in::Int64, poiss_rate::Float64,sim_δt::Float64)
    spikes_vect = zeros(Bool,n_in)
    for ii in 1:n_in
        spikes_vect[ii] = poisson_spike(rng,poiss_rate,sim_δt)
    end
    return spikes_vect
end

"""
position_to_rate(pos::Float64,pf::Array{Float64,1},fr_min::Float64, fr_max::Float64)

	Generates a Poisson rate given a position wrt a centered Gaussian Place Field (pf).

# Inputs:
- `pos::Float64`: position along track (1D).
- `pf::Array{Float64,1}`: Gaussian-like Place Field parameters [μ,σ]
- `fr_min::Float64`: minimum firing rate in Hz.
- `fr_max::Float64`: maximum firing rate in Hz.
"""
function position_to_rate(pos::Float64,pf::Array{Float64,1},fr_min::Float64, fr_max::Float64)
    return exp(-(pos-pf[1])^2/(2*(pf[2])^2))*(fr_max-fr_min)+fr_min
end


"""
ornstein_uhlenbeck(pos::Float64, sim_δt::Float64, drift::Float64=0.0, noise_scale::Float64=0.02, coherence_time::Float64=5.0)

	Generates a single data point of a Ornstein-Uhlenbeck process.

# Inputs:
- `pos::Float64`: position along the x-axis (in m).
- `sim_δt::Float64`: simulation time constant (ms).
- `drift::Float64=0.0`: drift constant of the UO process (m/sec).
- `noise_scale::Float64`: deviations from drift. Default: 0.2 (20 cm/sec with position in meters.)
- `coherence_time::Float64`: time scale of changes (in ms).
"""
function ornstein_uhlenbeck(rng::StableRNG,pos::Float64, sim_δt::Float64,RewardField::Vector{Float64},drift::Float64=0.0, noise_scale::Float64=0.2, coherence_time::Float64=5.0)
    σ = sqrt((2 * noise_scale^2) / (coherence_time * sim_δt))
    θ = 1 / coherence_time
    if (RewardField[1] - RewardField[2]) .< pos .< (RewardField[1] + RewardField[2])
        dpos = θ * (drift - pos) * sim_δt + σ * abs(rand(rng,Normal(0.0,sim_δt)) * 0.005)
    else
        dpos = θ * (drift - pos) * sim_δt + σ * abs(rand(rng,Normal(0.0,sim_δt)))
    end
    pos += dpos*sim_δt
    #periodic
    (pos <= 0.0) && (pos = 2.0+pos)
    (pos >= 2.0) && (pos = 2.0-pos)
    return pos
end

"""
simulate_treadmill(rng::StableRNG, sim_δt::Float64, sim_length::Float64, drift::Float64=40.)

	Generates a position vector representing a rodent running on a treadmill and stopping for licking to reward.
    Reward delivery varies from 1.-4. seconds taken from Uniform distribution. 

# Inputs:
- `pos::Float64`: position along the x-axis (in m).
- `sim_δt::Float64`: simulation time constant (ms).
- `drift::Float64=0.0`: drift constant of the UO process (m/sec).
- `noise_scale::Float64`: deviations from drift. Default: 0.2 (20 cm/sec with position in cm.)
- `coherence_time::Float64`: time scale of changes (in ms).
"""
function simulate_treadmill(rng::StableRNG, sim_δt::Float64, sim_length::Float64, drift::Float64=40.,rew_params::Vector{Any}=[110.,(20.,95.)])
    length = 200.   #cm
    x_reset = 0.0   #cm
    x_th=rew_params[1]#cm
    x_out=rew_params[1]+5.#cm (out of reward zone)
    t_ref=2.0       #sec
    tau_m=20.       #sec

    x=x_reset
    t=0.0

    traj_t=[] # list of times
    traj_x=[] # list of corresponding positions
    reward_time=[-100.]  # list of times of each reward times
    reward_off =[-100.]

    reward_start = rew_params[2][1]

    last_lap = -100.
    lap_flag = true
    #trajectory simulation
    while t<sim_length # in sec!
        if t > (last_lap + t_ref) # if not receiving reward
            x += sim_δt*((- x)/tau_m) + drift*sim_δt
            if t > reward_start
                if x>x_th && x < x_out && lap_flag
                    lap_flag = false
                    append!(reward_time,t)
                    t_ref = abs(rand(rng,Uniform(1.0,3.0)))
                    append!(reward_off,t+t_ref)
                    last_lap = t
                end
            end
            if x>length
                x = x_reset
                lap_flag = true
            end
        end
        t += sim_δt
        append!(traj_t,t)
        append!(traj_x,x)
    end
    #reward times
    popfirst!(reward_time) #remove the initial fake reward_time
    popfirst!(reward_off)
    reward_times = []
    for (ron,roff) in zip(reward_time,reward_off)
        push!(reward_times,[ron,roff])
    end

    return traj_x,reward_times
end



"""
fsm_treadmill(rng::StableRNG, sim_δt::Float64, sim_steps::Int64)

	Generates a position vector representing a rodent running on a treadmill and stopping for licking to reward.
    Reward delivery varies from 0.5-2 seconds taken from Uniform distribution. 
    The behavior is modeled as a finite state machine.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `sim_steps::Int64`: simulation length in timesteps.
"""
function fsm_treadmill(rng::StableRNG, sim_δt::Float64, sim_steps::Int64, rew_params::Vector{Any}=[120.,(25.,200.)])
    PEAK_STATE = :PEAK_STATE # cm/s
    REWARD_STATE = :REWARD_STATE # slower speed cm/s
    NO_REWARD_STATE = :NO_REWARD_STATE # slow speed cm/s
    ACCELERATING_STATE = :ACCELERATING_STATE # cm/s
    DECELERATING_STATE = :DECELERATING_STATE # cm/s

    belt_length = 200.
    reward_cue = 20. # in cm
    proximity_threshold = 5. # in cm
    rew_site = rew_params[1] # in cm
    lick_prob = 0.001
    start_pos = 0.0
    peak_acc = 100*ms_to_sec^2
    peak_deacc = 100*ms_to_sec^2
    peak_vel = 50*ms_to_sec
    min_vel = 2*ms_to_sec
    rew_acc = 2*ms_to_sec^2
    rew_vel = 5*ms_to_sec
    flag_state = false
    
    rew_flag = false # switches to true between reward time delivery

    state = ACCELERATING_STATE
    v = min_vel
    a = peak_acc


    vect_pos = zeros(sim_steps)
    vect_vel = zeros(sim_steps)
    state_vector = []

    ti = 2
    ticount = 0
    while ti <= sim_steps
        dist = rew_site - vect_pos[ti-1]
        if flag_state && state == PEAK_STATE
            # Peak velocity state; runs at peak velocity
            if abs(dist) <= reward_cue
                state = DECELERATING_STATE
                a = -peak_deacc
                flag_state = false
            end
        end
        if flag_state && state == ACCELERATING_STATE
            # Acceleration state; accelerates with ramp-up
            if vect_vel[ti-1] >= peak_vel
                state = PEAK_STATE
                a = 0.00
            end
            if (0 < dist <= proximity_threshold) && rew_flag
                state = DECELERATING_STATE
                a = -peak_deacc
                flag_state = false
                timer_rew = rand(rng,Uniform(0.1,0.25)) #0 to 0.5 sec
                ticount = Int(round(timer_rew * sec_to_ms / sim_δt,digits=0))
            end
        end
        if flag_state && state == REWARD_STATE
            # Licking state; walks at a slow pace
            if (ticount <= 0) || (abs(dist) > proximity_threshold)
                ticount = 0
                state = ACCELERATING_STATE
                a = peak_acc 
                flag_state = false
            else
                ticount -= 1
            end
        end
        if flag_state && state == NO_REWARD_STATE
            # Licking but not rewarded state; walks at a slow pace
            if ticount == 0 
                state = ACCELERATING_STATE
                a = peak_acc 
                flag_state = false
            else
                ticount -= 1
            end
        end
        if flag_state && state == DECELERATING_STATE
            # Deceleration state; decelerates with ramp-down
            if (0 < dist <= proximity_threshold) && rew_flag
                state = REWARD_STATE
                a = rew_acc 
                flag_state = false
                timer_rew = rand(rng,Uniform(2.,4.0)) #2 to 4 sec
                ticount = Int(round(timer_rew * sec_to_ms / sim_δt,digits=0))
            end
            if (rand(rng) <= lick_prob) && (dist > proximity_threshold)
                state = NO_REWARD_STATE
                a = rew_acc # (cm/s)/s
                flag_state = false
                timer_rew = rand(rng,Uniform(0.1,0.25)) #0.1 to 0.25 sec
                ticount = Int(round(timer_rew * sec_to_ms / sim_δt,digits=0))
            end
            if (abs(dist) >= reward_cue) || (vect_vel[ti-1] <= min_vel) # should not happen
                state = ACCELERATING_STATE
                a = peak_acc # (cm/s)/s
                flag_state = false
            end
        end
        if rew_params[2][1] <= ti*ms_to_sec*sim_δt <= rew_params[2][2]
            rew_flag = true
        else
            rew_flag = false
        end
        push!(state_vector,state)
        # velocity integration
        vect_vel[ti] = vect_vel[ti-1] + a*sim_δt
        # bounding velocity conditions
        (state == PEAK_STATE && vect_vel[ti] > peak_vel) && (vect_vel[ti] = peak_vel)
        (state == REWARD_STATE && vect_vel[ti] > rew_vel) && (vect_vel[ti] = rew_vel)
        (state == NO_REWARD_STATE && vect_vel[ti] > rew_vel) && (vect_vel[ti] = rew_vel)
        (state == DECELERATING_STATE && vect_vel[ti] <= min_vel) && (vect_vel[ti] = min_vel)
        # position integration
        vect_pos[ti] = vect_pos[ti-1] + vect_vel[ti-1]*sim_δt + 0.5*a*(sim_δt)^2
        if vect_pos[ti] >= belt_length || vect_pos[ti] < start_pos
            vect_pos[ti] = start_pos
        end
        ti += 1
        flag_state = true
    end
    # state change timings
    aux = ACCELERATING_STATE
    state_changes = [(0.0,0.0,aux)]
    for (ti,state) in enumerate(state_vector)
        if state != aux
            t0 = round(state_changes[end][2],digits=4)
            t1 = round(ti*ms_to_sec*sim_δt,digits=4)
            push!(state_changes,(t0,t1,aux))
        end
        aux = state
    end
    popfirst!(state_changes)
    return vect_pos, vect_vel, state_changes
end


"""
generate_circulartrack_input(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, PlaceFields::Vector{Vector{Float64}}, fr_min::Float64, fr_max::Float64) 

    Generates binary input matrix resembling place field-like spiking and UO process for moving across a 1-D linear track.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `sim_steps::Int64`: simulation length in timesteps.
- `n_in::Int64`: number of input cells.
- `PlaceFields::Vector{Vector{Float64}}`: vector of place field information [[mean, standard deviation],[..]..] in Hz.
- `fr_min::Float64, fr_max::Float64`: position vector.
"""
function generate_circulartrack_input(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, 
    PlaceFields::Vector{Vector{Float64}}, fr_min::Float64, fr_max::Float64, pos_vect::Vector{Float64})
    npats = size(PlaceFields)[1]-1#last == first
    samples = collect(1:n_in)#shuffle(rng,collect(1:n_in))
    npat = Int(n_in/npats)
    pf_id = zeros(Int,n_in)
    input_mat = zeros(Bool,(n_in, sim_steps))
    ti = 1
    for (id,j) in enumerate(collect(1:npat:n_in))
        pf_id[samples[j:j+npat-1]] .= id
    end
    while ti <= sim_steps # creates the input time matrix
        for ci in range(1,stop=n_in)
            actual_pf = pf_id[ci]
            if (pos_vect[ti] >= PlaceFields[end][1]-2*PlaceFields[end][2]) && (actual_pf == 1)
                actual_pf = length(PlaceFields)
            end
            ci_rate = position_to_rate(pos_vect[ti],PlaceFields[actual_pf],fr_min,fr_max)
            input_mat[ci,ti] = poisson_spike(rng,ci_rate,sim_δt)
        end
        ti += 1
    end
    return input_mat, pf_id
end

"""
generate_circulartrack_inh(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, PlaceFields::Vector{Vector{Float64}}, fr_min::Float64, fr_max::Float64, pos_vect::Vector{Float64})

    Generates binary input (inh) matrix resembling place field-like spiking and UO process for moving across a 1-D linear track.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `sim_steps::Int64`: simulation length in timesteps.
- `n_in::Int64`: number of input cells.
- `PlaceFields::Vector{Vector{Float64}}`: vector of place field information [[mean, standard deviation],[..]..] in Hz.
- `fr_min::Float64, fr_max::Float64`: minimum and maximum poisson spike rate in Hz.
- `vec_pos::Vector{Float64}`: position vector.
"""
function generate_circulartrack_inh(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, 
    PlaceFields::Vector{Vector{Float64}}, fr_min::Float64, fr_max::Float64, pos_vect::Vector{Any}) 
    npats = size(PlaceFields)[1]-1 #last == first
    samples = collect(1:n_in)#shuffle(rng,collect(1:n_in))
    npat = Int(n_in/npats)
    pf_id = zeros(Int,n_in)
    for (id,j) in enumerate(collect(1:npat:n_in))
        pf_id[samples[j:j+npat-1]] .= id
    end
    input_mat = zeros(Bool,(n_in, sim_steps))
    ti = 1
    while ti <= sim_steps # creates the input time matrix
        for ci in range(1,stop=n_in)
            actual_pf = pf_id[ci]
            if pos_vect[ti] >= 170. && actual_pf == 1
                actual_pf = length(PlaceFields) #last == first
            end
            ci_rate = position_to_rate(pos_vect[ti],PlaceFields[actual_pf],fr_min,fr_max)
            input_mat[ci,ti] = poisson_spike(rng,ci_rate,sim_δt)
        end
        ti += 1
    end
    return input_mat, pf_id
end



"""
generate_rewardcue_input(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, pos_vect::Vector{Float64},RewardField::Vector{Vector{Float64}}, fr_min::Float64, fr_max::Float64) 


	Generates binary input matrix resembling reward cue spiking at given position in 1-D linear track.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `sim_steps::Int64`: simulation length in timesteps.
- `n_in::Int64`: number of input cells.
- `pos_vect::Vector{Float64}`: 1-D position vector.
- `RewardField::Vector{Vector{Float64}}`: place field-like reward information [[mean, standard deviation],[..]..] in Hz.
- `fr_min::Float64, fr_max::Float64`: minimum and maximum poisson spike rate in Hz.
- `rew_timebounds::Vector{Int64}`: time boundaries for reward delivery.
"""
function generate_rewardcue_input(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, pos_vect::Vector{Float64}, 
    RewardField::Vector{Float64}, fr_min::Float64, fr_max::Float64, rew_timebounds::Vector{Int64}) 
    ci_rate = fr_min
    input_mat = zeros(Bool,(n_in, sim_steps))
    ti = 1
    while ti <= sim_steps
        for ci in range(1,stop=n_in)
            if rew_timebounds[1] <= ti <= rew_timebounds[2]
                ci_rate = position_to_rate(pos_vect[ti],RewardField,fr_min,fr_max)
            else
                ci_rate = fr_min
            end
            input_mat[ci,ti] = poisson_spike(rng,ci_rate,sim_δt)
        end
        ti += 1
    end
    return input_mat
end


"""
generate_rewardcue_pattern(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, pos_vect::Vector{Any}, 

	Generates binary input matrix resembling reward cue spiking at given position in 1-D linear track.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `sim_steps::Int64`: simulation length in timesteps.
- `n_in::Int64`: number of input cells.
- `pos_vect::Vector{Float64}`: 1-D position vector.
- `RewardField::Vector{Vector{Float64}}`: place field-like reward information [[mean, standard deviation],[..]..] in Hz.
- `fr_min::Float64, fr_max::Float64`: minimum and maximum poisson spike rate in Hz.
- `rew_timebounds::Vector{Int64}`: time boundaries for reward delivery.
- `bounds::Vector{Int64}`: reward acquisition time boundaries (min time, max time) in steps.
"""
function generate_rewardcue_pattern(rng::StableRNG,sim_δt::Float64, sim_steps::Int64, n_in::Int64, pos_vect::Vector{Any}, 
    RewardField::Vector{Float64}, fr_min::Float64, fr_max::Float64, rew_timebounds::Vector{Int64}, max_rew_delivery::Int64) 
    ci_rate = fr_min
    pat_length = max_rew_delivery
    pattern_mat = zeros(Bool,(n_in, pat_length))
    input_mat = zeros(Bool,(n_in, sim_steps))
    ti = 1
    tj = 1
    pattern_flag = false
    for ti in 1:max_rew_delivery
        for ci in range(1,stop=n_in)
            ci_rate = fr_max
            pattern_mat[ci,ti] = poisson_spike(rng,ci_rate,sim_δt)
        end
    end
    while ti <= sim_steps
        for ci in range(1,stop=n_in)
            if rew_timebounds[1] <= ti <= rew_timebounds[2]
                ti_rate = position_to_rate(pos_vect[ti],RewardField,fr_min,fr_max)
                if fr_max-fr_min <= ti_rate <= fr_max+fr_min
                    pattern_flag = true
                else
                    pattern_flag = false
                    tj = 1 #outside pattern
                    ci_rate = ti_rate
                end
            else
                ci_rate = fr_min
            end
            if pattern_flag
                input_mat[ci,ti] = pattern_mat[ci,tj]
            else
                input_mat[ci,ti] = poisson_spike(rng,ci_rate,sim_δt)
            end
        end
        tj += 1
        if tj == max_rew_delivery
            tj = 1
        end
        ti += 1
    end
    return input_mat
end


"""
groupbypf(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)


	Grouping of spike data by place field-like information.

# Inputs:
- `xs::Vector{T}`: spike times in an indexed vector.
- `ys::Vector{T}`: neuron ids in an indexed vector.
- `by::AbstractArray`: place field-like membership of each id.
"""
function groupbypf(data::AbstractArray, y_id::AbstractArray, by::AbstractArray, sim_δt::Float64)
    xs = []
    ys = []
    for sp in data
        for each in sp
            push!(xs, each)
        end
    end
    for ids in y_id
        for each in ids
            push!(ys,each)
        end
    end
    grouping = zeros(Int64,length(xs))
    for (index,pf) in enumerate(by)
        grouping[ys .== index] .= pf
    end
    return xs, ys, grouping
end




"""
TO BE DELETED
function position_to_pattern(pos_vector::Vector{Float64},rew_area::Tuple{Float64, Float64}, sim_δt::Float64,timeconv::Float64=sec_to_ms)

	Returns a vector of reward pattern times for vspans_reward function.

# Inputs:
- `pos_vector::Vector{Float64}`: Position vector in meters.
- `rew_area::Tuple{Float64, Float64}`: Tuple of reward area boundaries in meters.
- `sim_δt::Float64`: simulation time constant in miliseconds.
- `timeconv::Float64=sec_to_ms`: time conversation from index to simulation time for plotting (in secs, or whichever unit you use).
"""
function position_to_pattern(pos_vector::Vector{Float64},rew_area::Tuple{Float64, Float64}, sim_δt::Float64,timeconv::Float64=sec_to_ms)
    t1 = 0
    t2 = 0
    Δt = rew_area[2]-rew_area[1]
    spans = []
    for (id,ti) in enumerate(pos_vector)
        if isapprox(ti,rew_area[1],rtol=1e-3)
            t1 = id
        elseif isapprox(ti,rew_area[2],rtol=1e-3)
            t2 = id
        end
        if t2 != 0 && t1 != 0 && t2 > t1
            push!(spans,[t1*sim_δt/timeconv,t2*sim_δt/timeconv])
            t1 = 0
            t2 = 0
        end
    end
    return spans
end


function plot_pfs!(h::Plots.Plot{Plots.GRBackend},place_fields::Vector{Vector{Float64}},colors::Vector{RGB{Float64}})
    for (id,pf) in enumerate(place_fields)
        d = Normal(pf[1],pf[2])
        lo, hi = quantile.(d, [0.01, 0.99])
        x = range(lo, hi; length = 100)
        plot!(h,x,pdf.(d,x),label=string("(",pf[1],",",pf[2],")"),linewidth=3.0,color=colors[id])
    end
    plot!()
end

function plot_pfs!(h::Plots.Subplot{Plots.GRBackend},place_fields::Vector{Vector{Float64}},colors::Vector{RGB{Float64}})
    for (id,pf) in enumerate(place_fields)
        d = Normal(pf[1],pf[2])
        lo, hi = quantile.(d, [0.01, 0.99])
        x = range(lo, hi; length = 100)
        plot!(h,x,pdf.(d,x),label=string("(",pf[1],",",pf[2],")"),linewidth=3.0,color=colors[id])
    end
    plot!()
end

function plot_pfs!(h::Plots.Plot{Plots.GRBackend},place_fields::Vector{Vector{Float64}},colors::ColorPalette)
    for (id,pf) in enumerate(place_fields)
        d = Normal(pf[1],pf[2])
        lo, hi = quantile.(d, [0.01, 0.99])
        x = range(lo, hi; length = 100)
        plot!(h,x,pdf.(d,x),label=string("(",pf[1],",",pf[2],")"),linewidth=3.0,color=colors[id])
    end
    plot!()
end


function plot_pfs!(h::Plots.Subplot{Plots.GRBackend},place_fields::Vector{Vector{Float64}},colors::ColorPalette)
    for (id,pf) in enumerate(place_fields)
        cl = colors[id+1]
        d = Normal(pf[1],pf[2])
        lo, hi = quantile.(d, [0.01, 0.99])
        x = range(lo, hi; length = 100)
        plot!(h,x,pdf.(d,x),label=string("(",pf[1],",",pf[2],")"),linewidth=3.0,color=cl)
    end
    plot!()
end


"""
fractal_rate(rng::StableRNG,sim_δt::Float64,x, min_val::Float64, max_val::Float64)

	Generates a Brownian motion-based vector for input frequency variations. Used for multiplex encoding.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `min_val::Float64`: minimum frequency in Hz.
- `max_val::Float64`: maximum frequency in Hz.
"""
function fractal_rate(rng::StableRNG,sim_δt::Float64,x, min_val::Float64, max_val::Float64)
    if length(x) > 2
        n = length(x)
        mid = (n+1)÷2
        x[mid] = (x[1] + x[n]) / 2 + randn(rng) * sqrt(n) * sim_δt
        x[mid] = clamp(x[mid], min_val, max_val)
        fractal_rate(rng, sim_δt, view(x,1:mid), min_val, max_val)
        fractal_rate(rng, sim_δt, view(x,mid:n), min_val, max_val)
    end
end



"""
generate_noisy_square_wave(duration::Float64, sample_rate::Float64, sim_δt::Float64, duty_cycle::Float64, noise_amplitude::Float64)

	Generates a square wave with noise on top. Used as neurotransmitter function. 

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `sim_δt::Float64`: simulation time constant (ms).
- `min_val::Float64`: minimum frequency in Hz.
- `max_val::Float64`: maximum frequency in Hz.
"""
function generate_noisy_square_wave(sim_length::Float64, sample_rate::Float64,sim_δt::Float64, duty_cycle::Float64, signal_amplitude::Float64,noise_amplitude::Float64)
    num_samples = Int(round(sim_length /sim_δt))
    signal = zeros(Float64, num_samples)
    period_samples = Int(round(1e3/sample_rate /sim_δt))
    high_samples = Int(duty_cycle * period_samples)
    
    for i in 1:num_samples
        (mod(i-1, period_samples) < high_samples) && (signal[i] = signal_amplitude)
    end
    noise = noise_amplitude * randn(num_samples)
    
    return signal .+ noise
end







"""
generate_input_mat(n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Float64,sim_δt::Float64)

	Generates binary input matrix given sizes.

# Inputs:
- `rng::StableRNG`: random number generator from StableRNG package.
- `n_in::Int64`: input size (# of cells).
- `sim_steps::Int64`: simulation length in timesteps.
- `pats::Vector{Matrix{Bool}}`: vector of input patterns (boolean matrix).
- `pat_width::Int64`: pattern width in timesteps.
- `poiss_rate::Float64`: poisson spike rate in Hz.
- `sim_δt::Float64`: simulation time constant (ms).
"""
function generate_input_mat_multiplex(rng::StableRNG,n_in::Int64, sim_steps::Int64, pats::Vector{Any}, pat_width::Int64, poiss_rate::Vector{Float64},sim_δt::Float64)
    input_mat = zeros(Bool,(n_in, sim_steps))
    input_pat = []
    chunk = 1
    while chunk < sim_steps # creates the input time matrix and the pattern timings
        rand_width = Int(ceil(rand(rng,Uniform(pat_width,3*pat_width))))
        if rand_width > (sim_steps - chunk)
            rand_width = sim_steps - chunk
        end
        rand_mat = zeros(Bool,(n_in,rand_width))
        ti = chunk
        for i in range(1,stop=n_in)
            for j in range(1,stop=rand_width-1)
                if rand(rng) <= rand(rng)*poiss_rate[ti+j]*sim_δt*10^(-3)
                    rand_mat[i,j] = 1
                end
            end
        end
        input_mat[:,chunk:chunk+rand_width-1] =  rand_mat
        push!(input_pat,(0,chunk))
        chunk += rand_width
        if pat_width < (sim_steps - chunk)
            pat_dice = rand(rng)
            if pat_dice < 1/3
                input_mat[:,chunk:chunk+pat_width-1] = pats[1]
                push!(input_pat,(1,chunk))
            elseif pat_dice < 2/3
                input_mat[:,chunk:chunk+pat_width-1] = pats[2]
                push!(input_pat,(2,chunk))
            else
                input_mat[:,chunk:chunk+pat_width-1] = pats[3]
                push!(input_pat,(3,chunk))
            end
            chunk += pat_width
        end
    end
    return input_mat, input_pat
end
