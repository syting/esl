module LBP

type PairTree
    setOne::Vector
    setTwo::Vector
    children::Array{PairTree}
end

function getindex(pt::PairTree, childNum::Unsigned)
    @assert childNum <= length(pt.children)
    return children[childNum]
end

function increment!(s::Vector)
    for i in 1:length(s)
        s[i] += 1
    end
end     

function incrementTree!(pt::PairTree)
    if !isempty(pt.setOne)
        increment!(pt.setOne)
    end
    increment!(pt.setTwo)
    for child in pt.children
        incrementTree!(child)
    end
end

function addOne!(pt::PairTree)
    # This is inefficient but the arrays will never be very big..
    if !isempty(pt.setOne)
        pt.setOne = [1; pt.setOne]
    end
    pt.setTwo = [1; pt.setTwo]
    for child in pt.children
        addOne!(child)
    end
end

function construct(n::Integer)
    @assert 2 <= n
    cur_tree = PairTree([], [1,2], [PairTree([1], [2], [])])
    for i in 3:n
        rt = deepcopy(cur_tree)
        incrementTree!(rt)
        lt = deepcopy(rt)
        addOne!(lt)
        rt.setOne = [1]
        push!(lt.children, rt)
        cur_tree = lt
    end
    return cur_tree
end

function getRSS(X, Y::Vector, indices::Vector)
    if isempty(indices)
        rss = (Y'*Y)[1]
    else
        @assert indices[end] <= size(X, 2)
        subX = X[:,indices]
        yHat = subX*(subX'*subX)^(-1)*subX'*Y
        rss = ((Y - yHat)'*(Y - yHat))[1]
    end
    return rss
end

function updateBest!(d::Dict, indices::Vector, rss)
    k = length(indices)
    if !(k in keys(d)) || (rss < d[k][1])
        d[k] = (rss, indices)
    end
end

function checkNode!(X, Y::Vector, pt::PairTree, d::Dict, vq)
    kOne = length(pt.setOne)
    rssOne = getRSS(X, Y, pt.setOne)
    updateBest!(d, pt.setOne, rssOne)
    kTwo = length(pt.setTwo)
    rssTwo = getRSS(X, Y, pt.setTwo)
    updateBest!(d, pt.setTwo, rssTwo)
    if d[kOne][1] <= d[kTwo][1]
        return
    else
        for k in (kTwo - kOne - 1):-1:1
            if d[kTwo - k][1] <= d[kTwo][1] <= d[kTwo - k - 1]
                push!(vq, pt.children[k+1:end]...)
                return
            end
        end
        push!(vq, pt.children...)
    end
end

function lbpRegress(X, Y::Vector)
    n = size(X, 2)
    bestSubsets = [i => (inf(1.0), zeros(i)) for i in 0:n]
    pt = construct(n)
    visit_q = [pt]
    while !isempty(visit_q)
        cur_node = shift!(visit_q)
        checkNode!(X, Y, cur_node, bestSubsets, visit_q)
    end
    return bestSubsets
end

end
