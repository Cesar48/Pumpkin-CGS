use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::time::Instant;

use flatzinc::ConstraintItem;
use louvain::modularity::Modularity;
use petgraph::graph::NodeIndex;
use petgraph::prelude::EdgeRef;
use petgraph::Graph as PetGraph;
use petgraph::Undirected;
use pumpkin_solver::options::PartitionedInstanceData;
use pumpkin_solver::variables::DomainId;
use pumpkin_solver::variables::Literal;

use crate::flatzinc::ast::FlatZincAst;
use crate::flatzinc::compiler::context::CompilationContext;
use crate::flatzinc::error::FlatZincError;

/// Data struct which encapsulates the most important data for partitioning.
pub(crate) struct PartitionerData {
    // The graph to partition.
    pub(crate) graph: PetGraph<i32, f32, Undirected, u32>,
    // The mapping from variables to their respective vertices.
    var_to_vtxid: HashMap<DomainId, NodeIndex>,
    // The mapping from literals to their respective vertices.
    lit_to_vtxid: HashMap<Literal, NodeIndex>,
    // The total edge weight; important in later calculations.
    total_edge_weight: f32,
}

pub(crate) fn create_partitioned_instance_data(
    obj: &(Vec<(i32, DomainId)>, i32),
    ast: &FlatZincAst,
    context: &mut CompilationContext,
) -> Result<PartitionedInstanceData, FlatZincError> {
    let start = Instant::now();
    let partitioner_data = create_graph_representation(ast, context)?;
    let communities = partition_graph(&partitioner_data.graph);
    let mut relevant_communities = HashMap::new();
    obj.0
        .iter()
        .map(|(_, did)| (communities[&partitioner_data.var_to_vtxid[did]], did))
        .for_each(|(com, did)| relevant_communities.entry(com).or_insert(vec![]).push(*did));
    let mut community_dist = HashMap::new();
    let rel_com_set = relevant_communities.keys().collect::<HashSet<_>>();
    let rel_com_list = rel_com_set.iter().map(|x| *x).collect::<Vec<_>>();

    // Calculate distance between communities.
    for i1 in 0..rel_com_list.len() {
        let c1 = rel_com_list[i1];
        community_dist.insert(*c1, HashMap::new());
        for i2 in 0..rel_com_list.len() {
            if i1 != i2 {
                let c2 = rel_com_list[i2];
                // Initialize to 0.
                community_dist.get_mut(c1).unwrap().insert(*c2, 0.0);
            }
        }
    }
    communities
        .iter()
        // Only consider the relevant communities.
        .filter(|&(_, comm)| rel_com_set.contains(comm))
        // For each node, consider the edges, and add their weights to the distance.
        // (outgoing is used to ensure all edges are considered once)
        .for_each(|(node, comm)| {
            partitioner_data
                .graph
                .edges_directed(*node, petgraph::Outgoing)
                .into_iter()
                .for_each(|edge| {
                    if let Some(dist) = community_dist
                        .get_mut(comm)
                        .unwrap()
                        .get_mut(&communities[&edge.target()])
                    {
                        *dist += edge.weight();
                    }
                })
        });
    Ok(PartitionedInstanceData {
        communities: relevant_communities,
        community_distances: community_dist,
        time_taken: start.elapsed().as_micros(),
    })
}

/// Creates a graph out of a provided FlatZinc instance. The [`FlatZincAst`] and
/// [`CompilationContext`] are used to achieve this. To ensure all data is fully recoverable, the
/// data struct defined above is returned.
pub(crate) fn create_graph_representation(
    ast: &FlatZincAst,
    context: &mut CompilationContext,
) -> Result<PartitionerData, FlatZincError> {
    let mut graph = PetGraph::new_undirected();
    // Keep track of all vertex ID's for every variable we've covered.
    let mut var_to_vtxid = HashMap::new();
    // Keep track of all vertex ID's for every literal we've covered separately.
    let mut lit_to_vtxid = HashMap::new();
    // The total weight is needed in later calculations; keep track of it already for efficiency.
    let mut total_weight = 0.0;

    // Iterate over the constraints; each unique type of constraint needs to be handled differently,
    // similarly to [`post_constraints::run`]. Some specific types could be merged, since they used
    // the same variable layout and similar relationship strengths.
    for constraint in &ast.constraint_decls {
        let ConstraintItem { id, exprs, .. } = constraint;
        // Add vertex for each constraint; after that, decode the variables used in this constraint,
        // ensure that all variables are present in their respective mapping (or add them to the
        // graph if needed) and add the edges to the graph with the correct weight.

        let vtx_id = graph.add_node(1);

        match id.as_str() {
            "array_int_maximum" | "array_int_minimum" => {
                let mut variables = vec![context.resolve_integer_variable(&exprs[0])?];
                variables.extend_from_slice(&context.resolve_integer_variable_array(&exprs[1])?);

                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &variables);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    variables
                        .into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "array_int_element" | "array_var_int_element" => {
                let mut vars = vec![&exprs[0], &exprs[2]]
                    .into_iter()
                    .map(|e| context.resolve_integer_variable(e).unwrap())
                    .collect::<Vec<_>>();
                vars.extend_from_slice(&context.resolve_integer_variable_array(&exprs[1])?);

                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "int_lin_ne" => {
                let vars = context.resolve_integer_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "int_lin_ne_reif" => {
                let lit = context.resolve_bool_variable(&exprs[3])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                total_weight += 1.0;

                let vars = context.resolve_integer_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "int_lin_le" | "int_lin_eq" => {
                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let rhs = context.resolve_integer_constant_from_expr(&exprs[2])?;
                let vars = context.resolve_integer_variable_array(&exprs[1])?;

                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|e| &var_to_vtxid[&e])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    rhs,
                );
            }
            "int_lin_le_reif" | "int_lin_eq_reif" => {
                let lit = context.resolve_bool_variable(&exprs[3])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                total_weight += 1.0;

                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let rhs = context.resolve_integer_constant_from_expr(&exprs[2])?;
                let vars = context.resolve_integer_variable_array(&exprs[1])?;

                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|e| &var_to_vtxid[&e])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    rhs,
                )
            }
            "int_ne" | "int_le" | "int_lt" | "int_eq" | "int_abs" | "int_plus" | "int_times"
            | "int_div" | "int_max" | "int_min" => {
                let vars = exprs
                    .iter()
                    .map(|x| context.resolve_integer_variable(x).unwrap())
                    .collect::<Vec<_>>();
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "int_ne_reif" | "int_le_reif" | "int_lt_reif" | "int_eq_reif" => {
                let lit = context.resolve_bool_variable(&exprs[2])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                total_weight += 1.0;

                let vars = exprs[0..2]
                    .iter()
                    .map(|x| context.resolve_integer_variable(x).unwrap())
                    .collect::<Vec<_>>();
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "fzn_all_different_int" => {
                let vars = context.resolve_integer_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "array_bool_and" | "array_bool_or" => {
                // Arguably, the outcome literal can be seen as a reification variable,
                // and is treated as such
                let lit = context.resolve_bool_variable(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                total_weight += 1.0;

                let lits = context.resolve_bool_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    lits.into_iter()
                        .map(|l| &lit_to_vtxid[&l])
                        .collect::<Vec<_>>(),
                );
            }
            "array_bool_element" | "array_var_bool_element" => {
                let vars = vec![context.resolve_integer_variable(&exprs[0])?];
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let mut vtxs = vars
                    .into_iter()
                    .map(|x| &var_to_vtxid[&x])
                    .collect::<Vec<_>>();

                let mut lits = vec![context.resolve_bool_variable(&exprs[2])?];
                lits.extend_from_slice(&context.resolve_bool_variable_array(&exprs[1])?);

                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                vtxs.extend_from_slice(
                    &lits
                        .into_iter()
                        .map(|x| &lit_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
                total_weight += add_weighted_edge_global_constraint(&mut graph, vtx_id, vtxs);
            }
            "pumpkin_bool_xor"
            | "pumpkin_bool_xor_reif"
            | "bool_and"
            | "bool_not"
            | "bool_eq"
            | "bool_eq_reif" => {
                let lits = exprs
                    .iter()
                    .map(|x| context.resolve_bool_variable(x).unwrap())
                    .collect::<Vec<_>>();
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    lits.into_iter()
                        .map(|lit| &lit_to_vtxid[&lit])
                        .collect::<Vec<_>>(),
                );
            }
            "bool2int" => {
                let lit = context.resolve_bool_variable(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);

                let var = context.resolve_integer_variable(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vec![var]);

                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vec![&lit_to_vtxid[&lit], &var_to_vtxid[&var]],
                );
            }
            "bool_lin_eq" | "bool_lin_le" => {
                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let var = context.resolve_integer_variable(&exprs[2])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vec![var]);
                graph.add_edge(vtx_id, var_to_vtxid[&var], 1.0);
                total_weight += 1.0;

                let lits = context.resolve_bool_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                total_weight += add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    lits.into_iter()
                        .map(|lit| &lit_to_vtxid[&lit])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    0,
                );
            }
            "bool_clause" => {
                let mut lits = vec![];
                exprs
                    .iter()
                    .map(|e| context.resolve_bool_variable_array(e).unwrap())
                    .for_each(|rc| lits.extend_from_slice(&rc));

                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                let vtxs = lits
                    .into_iter()
                    .map(|lit| &lit_to_vtxid[&lit])
                    .collect::<Vec<_>>();
                let vtxs_len = vtxs.len();
                total_weight +=
                    add_weighted_edge_linear(&mut graph, vtx_id, vtxs, &vec![1; vtxs_len], 0);
            }
            "set_in_reif" => {
                let var = context.resolve_integer_variable(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vec![var]);

                let lit = context.resolve_bool_variable(&exprs[2])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                let vars = vec![&var_to_vtxid[&var], &lit_to_vtxid[&lit]];
                total_weight += add_weighted_edge_global_constraint(&mut graph, vtx_id, vars);
            }
            "pumpkin_cumulative" => {
                let vars = context.resolve_integer_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);

                total_weight += add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|did| &var_to_vtxid[&did])
                        .collect::<Vec<_>>(),
                );
            }
            _ => (),
        }
    }

    // Remove all constant nodes from the graph; since they do not change, they do not influence
    // their neighbours, and two of their neighbours do not influence one another through them.
    // As such, these nodes interfere with correct community finding.
    if let Some(x) = lit_to_vtxid.get(&context.constant_bool_false) {
        graph.remove_node(*x);
    }
    if let Some(x) = lit_to_vtxid.get(&context.constant_bool_true) {
        graph.remove_node(*x);
    }
    for v in context.constant_domain_ids.values() {
        if let Some(x) = var_to_vtxid.get(v) {
            graph.remove_node(*x);
        }
    }

    Ok(PartitionerData {
        graph,
        var_to_vtxid,
        lit_to_vtxid,
        total_edge_weight: total_weight,
    })
}

/// Ensures that every provided T (where T is either DomainId or Literal) is present in a given
/// graph, where the mapping from T to VertexId is given. If a T is not present in the mapping, and
/// thus not in the graph, it is added to both, meaning both the mapping and the graph may be
/// modified by this function.
fn ensure_vars_in_vtx_map<T>(
    graph: &mut PetGraph<i32, f32, Undirected>,
    var_to_vtx: &mut HashMap<T, NodeIndex>,
    vars: &[T],
) where
    T: Clone,
    T: Hash,
    T: Eq,
{
    for var in vars {
        if !var_to_vtx.contains_key(var) {
            var_to_vtx.insert(var.clone(), graph.add_node(1));
        }
    }
}

/// Adds weighted edges to the provided graph, based on the provided global constraint relationship.
/// The first provided node is the vertex representing the constraint, which is incident to all
/// edges created by this call. The provided list of other nodes is a list of vertexes representing
/// the variables associated with the constraint, all incident to exactly 1 edge created by this
/// call. All edges have the same weight.
fn add_weighted_edge_global_constraint(
    graph: &mut PetGraph<i32, f32, Undirected>,
    vertex: NodeIndex,
    variables: Vec<&NodeIndex>,
) -> f32 {
    let var_len = variables.len() as i32;
    let w = 0.9_f32.powi(var_len);
    for v in variables {
        graph.add_edge(vertex, *v, w);
    }
    w * var_len as f32
}

/// Adds weighted edges to the provided graph, based on the provided linear inequality relationship.
/// The first provided node is the vertex representing the constraint, which is incident to all
/// edges created by this call. The provided list of other nodes is a list of vertexes representing
/// the variables associated with the constraint, associated with the provided weights. The right
/// hand side of the inequality is passed for normalisation purposes.
fn add_weighted_edge_linear(
    graph: &mut PetGraph<i32, f32, Undirected>,
    vertex: NodeIndex,
    variables: Vec<&NodeIndex>,
    weights: &[i32],
    rhs: i32,
) -> f32 {
    // If no right hand side is passed, calculate it as the sum of all absolute weights.
    let div = if rhs == 0 {
        weights.iter().map(|x| x.abs()).sum()
    } else {
        rhs
    };
    let rel_weights = weights
        .iter()
        // Make sure all weights are positive.
        .map(|x| (*x as f32 / div as f32).abs())
        .collect::<Vec<_>>();
    for (&v, w) in variables.iter().zip(rel_weights.iter()) {
        graph.add_edge(vertex, *v, *w);
    }
    rel_weights.iter().sum::<f32>()
}

/// Use the Louvain library to partition a given undirected graph. Returns a HashMap mapping each
/// node to a partition number.
pub(crate) fn partition_graph<T>(
    graph: &PetGraph<T, f32, Undirected, u32>,
) -> HashMap<NodeIndex, i32> {
    let mut modular = Modularity::new(1.0, 1);
    modular.execute(graph);
    modular
        .communityByNode
        .iter()
        .zip(graph.node_indices())
        .map(|(comm, node)| (node, *comm))
        .collect::<HashMap<NodeIndex, i32>>()
}
