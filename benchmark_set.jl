function benchmark_set(Pbs::Vector{T},solver::tunedOptimizer;benchmark_size=0,valid_ratio=0.1,load_dict::String="") where T<:AbstractNLPModel

    if !(load_dict=="")
        try
            df=CSV.read(load_dict)
            solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
            @info "loaded $load_dict (sorting)"
        catch
            @warn "could not load $load_dict"
        end
    end

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

    if benchmark_size==0 || benchmark_size>NA
        benchmark_size=NA
    end

    test_pbs=Vector{AbstractNLPModel}()

    for i=1:benchmark_size
        index=rand(1:length(Pbs_avail))
        push!(test_pbs,Pbs_avail[index])
        deleteat!(Pbs_avail,index)
    end

    valid_num = Int(round(valid_ratio*NA))
    (valid_num==0) && (valid_num=1)
    valid_num<NA || error("wrong validation ratio")
    valid_pbs=Vector{AbstractNLPModel}()

    for i=1:valid_num
        index=rand(1:length(test_pbs))
        push!(valid_pbs,test_pbs[index])
        deleteat!(test_pbs,index)
    end

    print("\ntest set : ")
    for pb in test_pbs
        print("$(pb.meta.name)_$(pb.meta.nvar) ; ")
    end
    println("\n")

    print("validation set : ")
    for pb in valid_pbs
        print("$(pb.meta.name)_$(pb.meta.nvar) ; ")
    end
    println("\n")

    return (test_pbs,valid_pbs)

end
