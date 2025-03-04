mod collect_domains;
mod context;
mod create_objective;
mod create_search_strategy;
mod define_constants;
mod define_variable_arrays;
mod handle_set_in;
mod merge_equivalences;
mod partitioner;
mod post_constraints;
mod prepare_variables;

use context::CompilationContext;
use pumpkin_solver::Solver;

use super::ast::FlatZincAst;
use super::instance::FlatZincInstance;
use super::FlatZincError;
use super::FlatZincOptions;
use crate::flatzinc::compiler::partitioner::create_partition_and_process_instance;

pub(crate) fn compile(
    ast: FlatZincAst,
    solver: &mut Solver,
    options: FlatZincOptions,
    apply_partitioning: bool,
) -> Result<FlatZincInstance, FlatZincError> {
    let mut context = CompilationContext::new(solver);

    define_constants::run(&ast, &mut context)?;
    prepare_variables::run(&ast, &mut context)?;
    merge_equivalences::run(&ast, &mut context)?;
    handle_set_in::run(&ast, &mut context)?;
    collect_domains::run(&ast, &mut context)?;
    define_variable_arrays::run(&ast, &mut context)?;
    let objective_function_with_name = create_objective::run(&ast, &mut context)?;
    let objective_definition =
        post_constraints::run(&ast, &mut context, options, &objective_function_with_name)?;
    let search = create_search_strategy::run(&ast, &mut context)?;

    let objective_function = objective_function_with_name.unzip().0;

    // If we should apply partitioning and there is an objective function, perform partitioning.
    let partitioned = match (&objective_definition, apply_partitioning) {
        (Some(obj), true) => Some(create_partition_and_process_instance(
            &obj.0,
            &ast,
            &mut context,
        )?),
        _ => None,
    };

    Ok(FlatZincInstance {
        outputs: context.outputs,
        objective_function,
        search: Some(search),
        objective_definition,
        partitioned_instance: partitioned,
    })
}
