function weighting(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function,data_path::String;
                            grid::Int=20, log::Bool=true,recompute_weights::Bool=false,
                            save_dict::String="",load_dict::String="",
                            bb_computes_weights::Bool=false) where T<:AbstractNLPModel

    if !(load_dict=="")
        try
            df=CSV.read(load_dict)
            solver.weightCalls=Dict(String.(df[:,1]) .=> Int.(df[:,2]))
            @info "loaded $load_dict"
        catch
            @warn "could not load $load_dict"
        end
    end

    min_guess = false
    no_save_warn = false
    for pb in Pbs
        if recompute_weights || !(haskey(solver.weightCalls,pb.meta.name * "_$(pb.meta.nvar)"))
            if !min_guess
                @info "start guessing weightCalls for each problem..."
                min_guess=true
            end
            tpb = tuningProblem(solver,[pb];weights=false,admitted_failure=1)
            @info "weighting $(pb.meta.name)_$(pb.meta.nvar)"
            if bb_computes_weights
                argmin=runBBoptimizer(tpb)
                minCalls = runtopt(solver,pb,argmin)
            else
                (hyperparam_init, minCalls, guess_low_bound, guess_up_bound) = write_csv(tpb,data_path;grid=grid,log=log)
            end
            if minCalls<Inf
                solver.weightCalls[pb.meta.name * "_$(pb.meta.nvar)"]=minCalls
                println("""weightCalls = $(solver.weightCalls[pb.meta.name * "_$(pb.meta.nvar)"])""")
                if !no_save_warn
                    try
                        CSV.write(save_dict, solver.weightCalls)
                    catch
                        @warn "computed weightCalls will not be saved in .csv file"
                        no_save_warn = true
                    end
                end
            else
                @warn "Every run from $(pb.meta.name) failed"
            end
        end
    end
    min_guess ? (@info "all weightCalls guessed.\n") : (@info "all weightCalls already available.\n")

end
