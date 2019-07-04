using CSV

include("latinHypercube.jl")

"""

weighting(Pbs::Vector{AbstractNLPModel},solver::tunedOptimizer,runBBoptimizer::Function;
                            Nlhs::Int=20, recompute_weights::Bool=false,
                            save_dict::String="",load_dict::String="",logarithm::Bool=false)

Computes weights of training problems `Pbs` that will be used to balance them according to their typical performance while computing
the global cost to minimize. These weights are defined as the best performance index found during a preliminary search
on the given problem. This will be performed by the direct-search algorithm embedded in the function `runBBoptimizer`

# **Arguments** :

- `Pbs::Vector{AbstractNLPModel}`

The set of training problems of which weights will be computed.

- `solver::tunedOptimizer`

the tunedOptimizer depending on the parameters to optimize. `solver.run` returns the performance indexes that will be minimized.
Its attribute `weightPerf` will be filled with the computed weights.

- `runBBoptimizer::Function`

A *Function* that should take as argument an AbstractNLPModel and return a tuple (argmin,min) with argmin the input found by the direct-search algorithm that minimizes
the cost from this model with respect to its constraints, and min the corresponding objective value. Note that meta.x0, meta.lvar and meta.uvar from the NLP model should be used.
To get the objective function and constraints from the NLP model for a given input, it is suggested to use the following method :

    (f,c)=NLPModels.objcons(tpb::AbstractNLPModel, x::AbstractVector)

f is a *Float* corresponding to the objective value. c is a *Float* equal to the constraint :

    fail_nb/length(Pbs)-admitted_failure

with fail_nb the number of runs that failed for the set of parameters `x`. This quantity should remain negative in the feasible space.

It is discouraged to use the methods `NLPModels.obj(tpb::tuningProblem, x::AbstractVector)` and `NLPModels.cons(tpb::tuningProblem, x::AbstractVector)`
because the admitted failure constraint needs the calculation of the cost to be evaluated so it is a waste of time to evaluate
them separately.

# **Optional arguments** :

- `Nlhs::Int`

If strictly positive, a Latin-Hypercube search will be performed before calling the direct-search algorithm in order to find a good initialization point.
More precisely, n*Nlhs sample points will be tested with n the number of parameters to tune.
`0` by default.

- `logarithm::Bool`

If true, the Latin-Hypercube search and the direct-search process will be performed with a logarithmic scale.
`false` by default.

- `load_dict::String`

If it is a valid existing .csv file name, solver.weightPerf is replaced with the data from this file at the beginning of the process.
empty by default.

- `save_dict::String`

If it is a valid .csv file name, computed weights will be stored in a .csv file called `save_dict`.
empty by default.

- `recompute_weights::Bool`

If set to true, weights will be computed even they are already available either in the loaded .csv file or in `solver.weightPerf`.
`false` by default.

"""
function weighting(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function;
                            Nlhs::Int=0, recompute_weights::Bool=false,
                            save_dict::String="",load_dict::String="",logarithm::Bool=false) where T<:AbstractNLPModel

    #This second dictionary stores optimal parameters values
    data = Dict()

    #load existing weights dictionary
    if !(load_dict=="")
        try
            df=CSV.read(load_dict)
            solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
            @info "loaded $load_dict (weighting)"
        catch
            @warn "could not load $load_dict"
        end
    end

    min_guess = false
    no_save_warn = false
    for pb in Pbs
        if recompute_weights || !(haskey(solver.weightPerf,pb.meta.name * "_$(pb.meta.nvar)"))
            if !min_guess
                @info "start guessing weightPerf for each problem..."
                min_guess=true
            end
            tpb = tuningProblem(solver,[pb];weights=false,logarithm=logarithm,penalty=Inf)
            @info "weighting $(pb.meta.name)_$(pb.meta.nvar)"
            if Nlhs>0
                @info "lhs running..."
                (best_param,minPerf)=latinHypercube(tpb;N=Nlhs)
                logarithm && (best_param=exp.(best_param))
                @info "lhs search found good initialization point $best_param"
                tpb=tuningProblem(solver,[pb];weights=false,x0=best_param,logarithm=logarithm,penalty=Inf)
            end
            (argmin,minPerf)=runBBoptimizer(tpb)
            if minPerf<Inf
                logarithm && (argmin=exp.(argmin))
                data[pb.meta.name * "_$(pb.meta.nvar)"]=argmin
                solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"]=minPerf
                println("""weightPerf = $(solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"])""")
                if !no_save_warn
                    CSV.write("data", data)
                    try
                        CSV.write(save_dict, solver.weightPerf)
                    catch
                        @warn "computed weightPerf will not be saved in .csv file"
                        no_save_warn = true
                    end
                end
            else
                solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"]=0
                @warn "Every run from $(pb.meta.name) failed, it will be notified by 0 in dictionary"
            end
        end
    end
    min_guess ? (@info "all weightPerf guessed.\n") : (@info "all weightPerf already available.\n")

end
