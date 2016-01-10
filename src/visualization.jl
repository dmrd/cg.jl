##################
# Representation #
##################
import Base.show

function Base.show(io::IO, node::Node)
    parts = Vector{AbstractString}()
    push!(parts, "name: $(node.name)")
    push!(parts, "op: $(node.op)")
    if length(node.inputs) > 0
        push!(parts, "inputs: $(join([x.name for x = node.inputs], ", "))")
    end
    if length(node.outputs) > 0
        push!(parts, "outputs: $(join([x.name for x = node.outputs], ", "))")
    end

  print(io, "Node{$(join(parts, " || "))}")
end

function _generate_edges(node::Node, nodeIds::Dict{Node, Int}, result::Vector{AbstractString})
    for next in succ(node)
        edge = "$(nodeIds[node]) -> $(nodeIds[next]);"
        push!(result, edge)
    end
end



function _generate_subgraphs(nested::Dict{AbstractString, Union{Dict, Node}}, nodeIds::Dict{Node, Int}, result::Vector{AbstractString})
    for name in keys(nested)
        if typeof(nested[name]) <: Dict
            clean_name = replace(name,  ['#',':'], "_")
            push!(result, "subgraph cluster$(clean_name) {")
            push!(result, "label = \"$(clean_name)\";")
            _generate_subgraphs(nested[name], nodeIds, result)
            push!(result, "}")
        else
            _generate_edges(nested[name], nodeIds, result)
        end
    end
end

# Print connected component of node
function to_dot(G::Graph; group_nodes=false)
    nodeIds = Dict{Node, Int}()
    id = 0
    nested = Dict{AbstractString, Union{Dict, Node}}()
    individual = Vector{Node}()
    for node in G.nodes
        nodeIds[node] = id
        id += 1

        parts = split(node.name, "/")
        if length(parts) == 1
            push!(individual, node)
        else
            cdict = nested
            for (i, part) in enumerate(parts[1:end-1])
                if !(part in keys(cdict))
                    cdict[part] = Dict{AbstractString, Union{Dict, Node}}()
                end
                # Means there's both a namespace and a node with this path
                @assert typeof(cdict[part]) <: Dict
                cdict = cdict[part]
            end
            cdict[parts[end]] = node
        end
    end


    result = Vector{AbstractString}()
    edges = Vector{AbstractString}()
    for node in G.nodes
        thisId = nodeIds[node]
        tp = typeof(node.op)
        shape = tp == "Variable" ? "box" : "ellipse"
        color = tp == Variable ? "green" : tp == Placeholder ? "blue" : tp == Constant ? "tan" : "white"
        labelLine = string(thisId, " [shape=\"$(shape)\" fillcolor=$(color) label=\"$(tostring(node))\"];")
        push!(result, labelLine)
    end

    if (group_nodes)
        _generate_subgraphs(nested, nodeIds, result)
        for node in individual
            _generate_edges(node, nodeIds, result)
        end
    else
        for node in G.nodes
            _generate_edges(node, nodeIds, result)
        end
    end


    string("digraph computation {\n",
           join(result,"\n"),
           "\n}"
           )
end

function render(G::Graph, outfile::AbstractString; group_nodes=false)
    dotstr = to_dot(G, group_nodes=group_nodes)
    dotcmd = `dot -Tpng -o $(outfile)`
    run(pipeline(`echo $(dotstr)`, dotcmd))
end

function tostring(node::Node)
    # TODO - include values for constants
    return "$(node.name): $(typeof(node.op))"
end

function tostring(nodes::Vector{Node})
    c = ", "
    "[$(join(map(tostring, nodes), c))]"
end

# Pretty print a computation
function pprint(g::Graph)

end
