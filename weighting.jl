include("latinHypercube.jl")

function weighting(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function;
                            grid::Int=20, recompute_weights::Bool=false,
                            save_dict::String="",load_dict::String="",logarithm::Bool=false,
                            lhs::Bool=false) where T<:AbstractNLPModel

    if !(load_dict=="")
        try
            df=CSV.read(load_dict)
            solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
            @info "loaded $load_dict"
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
            tpb = tuningProblem(solver,[pb];weights=false,logarithm=logarithm)
            @info "weighting $(pb.meta.name)_$(pb.meta.nvar)"
            if lhs
                @info "lhs running..."
                (best_hyperparam,minPerf)=latinHypercube(tpb;N=grid)
                logarithm && (best_hyperparam=exp.(best_hyperparam))
                @info "lhs search found good initialization point $best_hyperparam"
                tpb=tuningProblem(solver,[pb];weights=false,x0=best_hyperparam,logarithm=logarithm)
            end
            argmin=runBBoptimizer(tpb)
            logarithm && (argmin=exp.(argmin))
            minPerf = runtopt(solver,pb,argmin)
            if minPerf<Inf
                solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"]=minPerf
                println("""weightPerf = $(solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"])""")
                if !no_save_warn
                    try
                        CSV.write(save_dict, solver.weightPerf)
                    catch
                        @warn "computed weightPerf will not be saved in .csv file"
                        no_save_warn = true
                    end
                end
            else
                @warn "Every run from $(pb.meta.name) failed"
            end
        end
    end
    min_guess ? (@info "all weightPerf guessed.\n") : (@info "all weightPerf already available.\n")

end
