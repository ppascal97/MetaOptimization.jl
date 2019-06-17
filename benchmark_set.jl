function benchmark_set(Pbs::Vector{T};benchmark_size=0,valid_ratio=0.1) where T<:AbstractNLPModel

    (benchmark_size>0) || (benchmark_size=length(Pbs))

    Pbs_copy=deepcopy(Pbs)
    test_pbs=Vector{AbstractNLPModel}()

    for i=1:benchmark_size
        index=rand(1:length(Pbs_copy))
        push!(test_pbs,Pbs_copy[index])
        deleteat!(Pbs_copy,index)
    end

    valid_num = Int(round(valid_ratio*length(Pbs)))
    (valid_num==0) && (valid_num=1)
    valid_num<length(Pbs) || error("wrong validation ratio")
    valid_pbs=Vector{AbstractNLPModel}()

    for i=1:round(valid_ratio*length(Pbs))
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
