"""

    benchmark_set(Pbs::Vector{AbstractNLPModels},solver::tunedOptimizer;valid_ratio=0.1,load_dict::String="",weights=true)

Pick valid problems from `Pbs` (which have a strictly positive weight in `solver.weightPerf`) and randomly separate them into two sets : training
and validation. Return a tuple (trainingProblems,validationProblems) containing two vectors of NLP models.

# **Arguments** :

- `Pbs::Vector{AbstractNLPModel}`

The set of problems to process

- `solver::tunedOptimizer`

The solver containing the weights dictionary.

# **Optional arguments** :

- `weights::Bool`

If false, information from weights dictionary is not taken into account.

- `valid_ratio::Float`

The proportion of problems used for validation among all problems

- `load_dict::String`

If it is a valid existing .csv file name, solver.weightPerf is replaced with the data from this file at the beginning of the process.
empty by default.


"""
function benchmark_set(Pbs::Vector{T},solver::tunedOptimizer;valid_ratio=0.1,load_dict::String="",weights=true) where T<:AbstractNLPModel

    if weights
        #load weights dictionary
        if !(load_dict=="")
            try
                df=CSV.read(load_dict)
                solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
                @info "loaded $load_dict (sorting)"
            catch
                @warn "could not load $load_dict"
            end
        end

        #eliminate problems of which weight is null or that are not available in weightPerf
        Pbs_avail=Vector{AbstractNLPModel}()
        for pb in Pbs
            weight=0
            try
                weight=solver.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"]
            catch
                @warn "$(pb.meta.name)_$(pb.meta.nvar) is not in dictionary, it will not make part of benchmark sets"
            end
            if weight!=0
                push!(Pbs_avail,pb)
            end
        end
        NA = length(Pbs_avail)
    else
        Pbs_avail=deepcopy(Pbs)
    end

    valid_num = Int(round(valid_ratio*NA))
    valid_num<NA || error("wrong validation ratio")
    valid_pbs=Vector{AbstractNLPModel}()

    for i=1:valid_num
        index=rand(1:length(Pbs_avail))
        push!(valid_pbs,Pbs_avail[index])
        deleteat!(Pbs_avail,index)
    end

    print("\ntest set : ")
    for pb in Pbs_avail
        print("$(pb.meta.name)_$(pb.meta.nvar) ; ")
    end
    println("\n")

    print("validation set : ")
    for pb in valid_pbs
        print("$(pb.meta.name)_$(pb.meta.nvar) ; ")
    end
    println("\n")

    return (Pbs_avail,valid_pbs)

end
