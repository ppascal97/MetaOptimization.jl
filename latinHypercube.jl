function latinHypercube(tpb;N=200)

    LB = tpb.meta.lvar
    UB = tpb.meta.uvar

    samples=Vector{Vector{Float64}}()

    for i=1:tpb.meta.nvar
        Xi=Vector{Float64}(undef,N)
        delta=(UB[i]-LB[i])/N
        for j=1:N
            Xi[j]=LB[i]+delta*(j-1+rand())
        end
        push!(samples,Xi)
    end

    list_points=Vector{Vector{Float64}}()
    for j=1:N
        point=Vector{Float64}(undef,tpb.meta.nvar)
        for i=1:tpb.meta.nvar
            index=rand(1:length(samples[i]))
            point[i]=samples[i][index]
            deleteat!(samples[i],index)
        end
        push!(list_points,point)
    end

    minPerf=Inf
    best_hyperparam = Vector{Float64}(undef,tpb.meta.nvar)
    for k=1:N
        (obj,cstr)=objcons(tpb,list_points[k])
        if tpb.post_obj_cstr && cstr[end]>0
            obj=Inf
        end
        if obj<minPerf
            minPerf=obj
            best_hyperparam=list_points[k]
        end
    end

    return (best_hyperparam,minPerf)
end
