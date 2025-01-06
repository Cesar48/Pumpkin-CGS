use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::RandomState;
use std::time::Instant;

use flatzinc::ConstraintItem;
use log::info;
use louvain::modularity::Modularity;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Graph as PetGraph;
use petgraph::Outgoing;
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
    pub(crate) graph: PetGraph<i32, f32, Undirected>,
    // The mapping from variables to their respective vertices.
    var_to_vtxid: HashMap<DomainId, NodeIndex>,
    // The mapping from literals to their respective vertices.
    lit_to_vtxid: HashMap<Literal, NodeIndex>,
}

/// Shorthand function that provides a simple API to the features in this file. Accepts an objective
/// function, the problem it was taken from, and the compilation context; these are needed to create
/// the partitioning. A [`PartitionedInstanceData`] is returned, containing all information needed
/// to further use and process the partitions - including the time taken.
pub(crate) fn create_partition_and_process_instance(
    obj: &[(i32, DomainId)],
    ast: &FlatZincAst,
    context: &mut CompilationContext,
) -> Result<PartitionedInstanceData, FlatZincError> {
    let start = Instant::now();
    let PartitionerData {
        graph,
        var_to_vtxid,
        ..
    } = create_graph_representation(ast, context)?;
    let communities = partition_graph(&graph);

    // Convert the `communities` map to a map from community ID (i32) to nodes (NodeIndex) for more
    // convenient access to elements of a single community.
    let mut relevant_communities = HashMap::default();
    obj.iter()
        .map(|(_, did)| (communities[&var_to_vtxid[did]], did))
        .for_each(|(com, did)| relevant_communities.entry(com).or_insert(vec![]).push(*did));

    // Calculate distance between communities
    let mut community_dist = HashMap::default();
    let rel_com_list = relevant_communities.keys().map(|x| *x).collect::<Vec<_>>();
    let rel_com_set: HashSet<i32, RandomState> =
        HashSet::from_iter(rel_com_list.iter().map(|x| *x));

    for i1 in 0..rel_com_list.len() {
        let c1 = rel_com_list[i1];
        let mut distmap = HashMap::default();
        for i2 in 0..rel_com_list.len() {
            let c2 = rel_com_list[i2];
            if c1 != c2 {
                assert!(distmap.insert(c2, 0.0).is_none());
            }
        }
        assert!(community_dist.insert(c1, distmap).is_none());
    }
    communities
        .iter()
        .filter(|&(_, comm)| rel_com_set.contains(comm))
        .for_each(|(node, comm)| {
            graph
                .edges_directed(*node, Outgoing)
                .into_iter()
                .for_each(|edge| {
                    if let Some(map) = community_dist.get_mut(comm) {
                        if let Some(dist) = map.get_mut(&communities[&edge.target()]) {
                            *dist += edge.weight();
                        }
                    }
                })
        });

    info!("Created {} communities", relevant_communities.len());
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
    // Keep track of all vertex ID's for every variable and literal we've covered.
    let mut var_to_vtxid = HashMap::default();
    let mut lit_to_vtxid = HashMap::default();

    // Iterate over the constraints; most constraints need to be handled differently, depending on
    // variable layout and relationship strength - this is similar on only some constraints.
    for constraint in &ast.constraint_decls {
        let ConstraintItem { id, exprs, .. } = constraint;
        // Add vertex for each constraint and add the edges with the correct weight to the graph
        // between this vertex and the (potentially new) variable/literal vertices.
        let vtx_id = graph.add_node(1);

        match id.as_str() {
            "array_int_maximum" | "array_int_minimum" => {
                let var = context.resolve_integer_variable(&exprs[0])?;
                let mut variables = vec![var];
                variables.extend_from_slice(&context.resolve_integer_variable_array(&exprs[1])?);

                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &variables);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            // Due to the inherently different behaviour of != compared to <=, > and =, this is
            // considered a global constraint rather than a linear one.
            "int_lin_ne" => {
                let vars = context.resolve_integer_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                let vars = context.resolve_integer_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = add_weighted_edge_linear(
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
                let _ = graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let rhs = context.resolve_integer_constant_from_expr(&exprs[2])?;

                let vars = context.resolve_integer_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|e| &var_to_vtxid[&e])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    rhs,
                );
            }
            "int_ne" | "int_le" | "int_lt" | "int_eq" | "int_abs" | "int_plus" | "int_times"
            | "int_div" | "int_max" | "int_min" => {
                let vars = exprs
                    .iter()
                    .map(|x| context.resolve_integer_variable(x).unwrap())
                    .collect::<Vec<_>>();
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                let vars = exprs[0..2]
                    .iter()
                    .map(|x| context.resolve_integer_variable(x).unwrap())
                    .collect::<Vec<_>>();
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vars.into_iter()
                        .map(|x| &var_to_vtxid[&x])
                        .collect::<Vec<_>>(),
                );
            }
            "pumpkin_all_different" => {
                let vars = context.resolve_integer_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = graph.add_edge(vtx_id, lit_to_vtxid[&lit], 1.0);
                let lits = context.resolve_bool_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                let _ = add_weighted_edge_global_constraint(
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
                let _ = add_weighted_edge_global_constraint(&mut graph, vtx_id, vtxs);
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
                let _ = add_weighted_edge_global_constraint(
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
                let _ = add_weighted_edge_global_constraint(
                    &mut graph,
                    vtx_id,
                    vec![&lit_to_vtxid[&lit], &var_to_vtxid[&var]],
                );
            }
            "bool_lin_eq" => {
                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let var = context.resolve_integer_variable(&exprs[2])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vec![var]);
                let edge_w = *graph.node_weight(var_to_vtxid[&var]).unwrap() as f32;
                let _ = graph.add_edge(vtx_id, var_to_vtxid[&var], edge_w);

                let lits = context.resolve_bool_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                let _ = add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    lits.into_iter()
                        .map(|lit| &lit_to_vtxid[&lit])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    0,
                );
            }
            "bool_lin_le" => {
                let orig_weights = context.resolve_array_integer_constants(&exprs[0])?;
                let rhs = context.resolve_integer_constant_from_expr(&exprs[2])?;

                let lits = context.resolve_bool_variable_array(&exprs[1])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &lits);
                let _ = add_weighted_edge_linear(
                    &mut graph,
                    vtx_id,
                    lits.into_iter()
                        .map(|lit| &lit_to_vtxid[&lit])
                        .collect::<Vec<_>>(),
                    &orig_weights,
                    rhs,
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
                let _ =
                    add_weighted_edge_linear(&mut graph, vtx_id, vtxs, &vec![1; vtxs_len], 0);
            }
            "set_in_reif" => {
                let var = context.resolve_integer_variable(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vec![var]);

                let lit = context.resolve_bool_variable(&exprs[2])?;
                ensure_vars_in_vtx_map(&mut graph, &mut lit_to_vtxid, &vec![lit]);
                let vars = vec![&var_to_vtxid[&var], &lit_to_vtxid[&lit]];
                let _ = add_weighted_edge_global_constraint(&mut graph, vtx_id, vars);
            }
            "pumpkin_cumulative" => {
                let vars = context.resolve_integer_variable_array(&exprs[0])?;
                ensure_vars_in_vtx_map(&mut graph, &mut var_to_vtxid, &vars);

                let _ = add_weighted_edge_global_constraint(
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

    // Remove any and all constant nodes from the graph; since they do not change, they do not
    // influence their neighbours, and two of their neighbours do not influence one another
    // through them. As such, these nodes interfere with correct community finding.
    if let Some(x) = lit_to_vtxid.get(&context.constant_bool_false) {
        let _ = graph.remove_node(*x);
    }
    if let Some(x) = lit_to_vtxid.get(&context.constant_bool_true) {
        let _ = graph.remove_node(*x);
    }
    for v in context.constant_domain_ids.values() {
        if let Some(x) = var_to_vtxid.get(v) {
            let _ = graph.remove_node(*x);
        }
    }

    Ok(PartitionerData {
        graph,
        var_to_vtxid,
        lit_to_vtxid,
    })
}

/// Ensures that given values `vars` are present in the vertex map `var_to_vtx`. If a given value
/// from `vars` is not present in `var_to_vtx`, a new node is added to the graph, and the key-value
/// pair of said value and the new node is added to `var_to_vtx`.
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
            assert!(var_to_vtx.insert(var.clone(), graph.add_node(1)).is_none());
        }
    }
}

/// Adds a weight to `graph` according to the rule crafted for global constraints: raise 0.9 to the
/// power `n`, where `n` is the number of arguments to the global constraint. The remaining number
/// is used as the weight to all `n` edges between the constraint node and variable nodes. The sum
/// of these new edge weights (i.e. `n*0.9^n`) is returned.
fn add_weighted_edge_global_constraint(
    graph: &mut PetGraph<i32, f32, Undirected>,
    vertex: NodeIndex,
    variables: Vec<&NodeIndex>,
) -> f32 {
    let var_len = variables.len() as i32;
    let w = 0.9_f32.powi(var_len);
    for v in variables {
        let mult = graph.node_weight(*v).unwrap();
        let _ = graph.add_edge(vertex, *v, w * (*mult as f32));
    }
    w * var_len as f32
}

/// Adds a weight to `graph` according to the rule crafted for linear constraints: normalize all
/// weights by the right hand side (if any) or the sum of all absolute weights (otherwise). The
/// magnitudes of these normalized weights are used as the edge weights between the constraint node
/// and the respective variable (or literal) node. The sum of these new edge weights is returned.
fn add_weighted_edge_linear(
    graph: &mut PetGraph<i32, f32, Undirected>,
    vertex: NodeIndex,
    variables: Vec<&NodeIndex>,
    weights: &[i32],
    rhs: i32,
) -> f32 {
    // If no RHS is present, use the sum of all (absolute) weights to normalize.
    let div = if rhs == 0 {
        weights.iter().map(|x| x.abs()).sum()
    } else {
        rhs
    };
    let rel_weights = weights
        .iter()
        .map(|x| (*x as f32 / div as f32).abs())
        .collect::<Vec<_>>();
    for (&v, w) in variables.iter().zip(rel_weights.iter()) {
        let mult = graph.node_weight(*v).unwrap();
        let _ = graph.add_edge(vertex, *v, *w * (*mult as f32));
    }
    rel_weights.iter().sum::<f32>()
}

/// Accepts a graph and uses the `louvain` crate to partition it. The partition information is
/// returned in a HashMap which maps a NodeIndex to a partition number (i32).
pub(crate) fn partition_graph(graph: &PetGraph<i32, f32, Undirected>) -> HashMap<NodeIndex, i32> {
    let mut modular = Modularity::new(1.0, 1);
    let _ = modular.execute(graph);
    graph
        .node_indices()
        .zip(modular.communityByNode.into_iter())
        .collect::<HashMap<NodeIndex, i32>>()
}
