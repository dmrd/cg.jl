##################
# Representation #
##################

# Print connected component of node
function to_dot(G::Graph)
    nodeIds = Dict{Node, Int}()
    id = 0
    for node in G.nodes
        nodeIds[node] = id
        id += 1
    end

    labels = Vector{AbstractString}()
    edges = Vector{AbstractString}()
    for node in G.nodes
        thisId = nodeIds[node]
        #shape = isa(node, Operation) ? "box" : "ellipse"
        shape = "ellipse"
        labelLine = string(thisId, " [shape=\"", shape,"\", label=\"", tostring(node), "\"];")
        push!(labels, labelLine)
        for next in succ(node)
            edge = "$(nodeIds[node]) -> $(nodeIds[next]);"
            push!(edges, edge)
        end
    end

    string("digraph computation {\n",
           join(labels,"\n"),
           "\n",
           join(edges,"\n"),
           "\n}"
           )
end

function render(G::Graph, outfile::AbstractString)
    dotstr = to_dot(G)
    dotcmd = `dot -Tpng -o $(outfile)`
    run(pipeline(`echo $(dotstr)`, dotcmd))
end

function tostring(node::Node)
    # TODO - include values for constants
    if !(isnull(node.name))
        return "$(get(node.name)): $(typeof(node.op))"
    else
        return "$(typeof(node.data))"
    end
end

function tostring(nodes::Vector{Node})
    c = ", "
    "[$(join(map(tostring, nodes), c))]"
end

# Pretty print a computation
function pprint(g::Graph)

end
