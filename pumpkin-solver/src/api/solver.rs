use std::num::NonZero;

use log::debug;
use log::info;
use log::warn;

use super::results::OptimisationResult;
use super::results::SatisfactionResult;
use super::results::SatisfactionResultUnderAssumptions;
use crate::api::monitors::CoreExhaustionMonitor;
use crate::api::monitors::CoreSizeMonitor;
use crate::api::monitors::HardeningDomainLimitationMonitor;
use crate::api::monitors::LowerBoundEvolutionMonitor;
use crate::api::monitors::MonitoredTasks;
use crate::api::monitors::TimePerTaskMonitor;
use crate::api::monitors::WCECoreAmountMonitor;
use crate::basic_types::CSPSolverExecutionFlag;
use crate::basic_types::ConstraintOperationError;
use crate::basic_types::HashMap;
use crate::basic_types::HashSet;
use crate::basic_types::Solution;
use crate::branching::branchers::independent_variable_value_brancher::IndependentVariableValueBrancher;
#[cfg(doc)]
use crate::branching::value_selection::ValueSelector;
#[cfg(doc)]
use crate::branching::variable_selection::VariableSelector;
use crate::branching::Brancher;
use crate::branching::PhaseSaving;
use crate::branching::SolutionGuidedValueSelector;
use crate::branching::Vsids;
use crate::constraints::boolean_equals;
use crate::constraints::less_than_or_equals;
use crate::constraints::Constraint;
use crate::constraints::ConstraintPoster;
use crate::engine::predicates::predicate::Predicate;
use crate::engine::propagation::Propagator;
use crate::engine::propagation::ReadDomains;
use crate::engine::termination::TerminationCondition;
use crate::engine::variables::DomainId;
use crate::engine::variables::IntegerVariable;
use crate::engine::variables::Literal;
use crate::engine::ConstraintSatisfactionSolver;
use crate::options::LearningOptions;
use crate::options::SolverOptions;
use crate::predicate;
use crate::predicates::IntegerPredicate;
use crate::pumpkin_assert_simple;
use crate::results::solution_iterator::SolutionIterator;
use crate::results::unsatisfiable::UnsatisfiableUnderAssumptions;
use crate::results::SolutionCallbackArguments;
use crate::statistics::statistic_logging::log_statistic;
use crate::statistics::statistic_logging::log_statistic_postfix;
use crate::variables::AffineView;
use crate::variables::PropositionalVariable;
use crate::variables::TransformableVariable;

/// The main interaction point which allows the creation of variables, the addition of constraints,
/// and solving problems.
///
///
/// # Creating Variables
/// As stated in [`crate::variables`], we can create two types of variables: propositional variables
/// and integer variables.
///
/// ```rust
/// # use pumpkin_solver::Solver;
/// # use crate::pumpkin_solver::variables::TransformableVariable;
/// let mut solver = Solver::default();
///
/// // Integer Variables
///
/// // We can create an integer variable with a domain in the range [0, 10]
/// let integer_between_bounds = solver.new_bounded_integer(0, 10);
///
/// // We can also create such a variable with a name
/// let named_integer_between_bounds = solver.new_named_bounded_integer(0, 10, "x");
///
/// // We can also create an integer variable with a non-continuous domain in the follow way
/// let mut sparse_integer = solver.new_sparse_integer(vec![0, 3, 5]);
///
/// // We can also create such a variable with a name
/// let named_sparse_integer = solver.new_named_sparse_integer(vec![0, 3, 5], "y");
///
/// // Additionally, we can also create an affine view over a variable with both a scale and an offset (or either)
/// let view_over_integer = integer_between_bounds.scaled(-1).offset(15);
///
///
/// // Propositional Variable
///
/// // We can create a literal
/// let literal = solver.new_literal();
///
/// // We can also create such a variable with a name
/// let named_literal = solver.new_named_literal("z");
///
/// // We can also get the propositional variable from the literal
/// let propositional_variable = literal.get_propositional_variable();
///
/// // We can also create an iterator of new literals and get a number of them at once
/// let list_of_5_literals = solver.new_literals().take(5).collect::<Vec<_>>();
/// assert_eq!(list_of_5_literals.len(), 5);
/// ```
///
/// # Using the Solver
/// For examples on how to use the solver, see the [root-level crate documentation](crate) or [one of these examples](https://github.com/ConSol-Lab/Pumpkin/tree/master/pumpkin-lib/examples).
pub struct Solver {
    /// The internal [`ConstraintSatisfactionSolver`] which is used to solve the problems.
    satisfaction_solver: ConstraintSatisfactionSolver,
    /// The function is called whenever an optimisation function finds a solution; see
    /// [`Solver::with_solution_callback`].
    solution_callback: Box<dyn Fn(SolutionCallbackArguments)>,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            satisfaction_solver: Default::default(),
            solution_callback: create_empty_function(),
        }
    }
}

/// Creates a place-holder empty function which does not do anything when a solution is found.
fn create_empty_function() -> Box<dyn Fn(SolutionCallbackArguments)> {
    Box::new(|_| {})
}

impl std::fmt::Debug for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("satisfaction_solver", &self.satisfaction_solver)
            .finish()
    }
}

impl Solver {
    /// Creates a solver with the provided [`LearningOptions`] and [`SolverOptions`].
    pub fn with_options(learning_options: LearningOptions, solver_options: SolverOptions) -> Self {
        Solver {
            satisfaction_solver: ConstraintSatisfactionSolver::new(
                learning_options,
                solver_options,
            ),
            solution_callback: create_empty_function(),
        }
    }

    /// Adds a call-back to the [`Solver`] which is called every time that a solution is found when
    /// optimising using [`Solver::maximise`] or [`Solver::minimise`].
    ///
    /// Note that this will also
    /// perform the call-back on the optimal solution which is returned in
    /// [`OptimisationResult::Optimal`].
    pub fn with_solution_callback(
        &mut self,
        solution_callback: impl Fn(SolutionCallbackArguments) + 'static,
    ) {
        self.solution_callback = Box::new(solution_callback);
    }

    /// Logs the statistics currently present in the solver with the provided objective value.
    pub fn log_statistics_with_objective(&self, objective_value: i64) {
        log_statistic("objective", objective_value);
        self.log_statistics();
    }

    /// Logs the statistics currently present in the solver.
    pub fn log_statistics(&self) {
        self.satisfaction_solver.log_statistics();
        log_statistic_postfix();
    }

    pub(crate) fn get_satisfaction_solver_mut(&mut self) -> &mut ConstraintSatisfactionSolver {
        &mut self.satisfaction_solver
    }
}

/// Methods to retrieve information about variables
impl Solver {
    /// Get the literal corresponding to the given predicate. As the literal may need to be
    /// created, this possibly mutates the solver.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// # use pumpkin_solver::predicate;
    /// let mut solver = Solver::default();
    ///
    /// let x = solver.new_bounded_integer(0, 10);
    ///
    /// // We can get the literal representing the predicate `[x >= 3]` via the Solver
    /// let literal = solver.get_literal(predicate!(x >= 3));
    ///
    /// // Note that we can also get a literal which is always true
    /// let true_lower_bound_literal = solver.get_literal(predicate!(x >= 0));
    /// assert_eq!(true_lower_bound_literal, solver.get_true_literal());
    /// ```
    pub fn get_literal(&self, predicate: Predicate) -> Literal {
        self.satisfaction_solver.get_literal(predicate)
    }

    /// Get the value of the given [`Literal`] at the root level (after propagation), which could be
    /// unassigned.
    pub fn get_literal_value(&self, literal: Literal) -> Option<bool> {
        self.satisfaction_solver.get_literal_value(literal)
    }

    /// Get a literal which is globally true.
    pub fn get_true_literal(&self) -> Literal {
        self.satisfaction_solver.get_true_literal()
    }

    /// Get a literal which is globally false.
    pub fn get_false_literal(&self) -> Literal {
        self.satisfaction_solver.get_false_literal()
    }

    /// Get the lower-bound of the given [`IntegerVariable`] at the root level (after propagation).
    pub fn lower_bound(&self, variable: &impl IntegerVariable) -> i32 {
        self.satisfaction_solver.get_lower_bound(variable)
    }

    /// Get the upper-bound of the given [`IntegerVariable`] at the root level (after propagation).
    pub fn upper_bound(&self, variable: &impl IntegerVariable) -> i32 {
        self.satisfaction_solver.get_upper_bound(variable)
    }
}

/// Functions to create and retrieve integer and propositional variables.
impl Solver {
    /// Returns an infinite iterator of positive literals of new variables. The new variables will
    /// be unnamed.
    ///
    /// # Example
    /// ```
    /// # use pumpkin_solver::Solver;
    /// # use pumpkin_solver::variables::Literal;
    /// let mut solver = Solver::default();
    /// let literals: Vec<Literal> = solver.new_literals().take(5).collect();
    ///
    /// // `literals` contains 5 positive literals of newly created propositional variables.
    /// assert_eq!(literals.len(), 5);
    /// ```
    ///
    /// Note that this method captures the lifetime of the immutable reference to `self`.
    pub fn new_literals(&mut self) -> impl Iterator<Item = Literal> + '_ {
        std::iter::from_fn(|| Some(self.new_literal()))
    }

    /// Create a fresh propositional variable and return the literal with positive polarity.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can create a literal
    /// let literal = solver.new_literal();
    /// ```
    pub fn new_literal(&mut self) -> Literal {
        Literal::new(
            self.satisfaction_solver
                .create_new_propositional_variable(None),
            true,
        )
    }

    /// Create a fresh propositional variable with a given name and return the literal with positive
    /// polarity.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can also create such a variable with a name
    /// let named_literal = solver.new_named_literal("z");
    /// ```
    pub fn new_named_literal(&mut self, name: impl Into<String>) -> Literal {
        Literal::new(
            self.satisfaction_solver
                .create_new_propositional_variable(Some(name.into())),
            true,
        )
    }

    /// Create a new integer variable with the given bounds.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can create an integer variable with a domain in the range [0, 10]
    /// let integer_between_bounds = solver.new_bounded_integer(0, 10);
    /// ```
    pub fn new_bounded_integer(&mut self, lower_bound: i32, upper_bound: i32) -> DomainId {
        self.satisfaction_solver
            .create_new_integer_variable(lower_bound, upper_bound, None)
    }

    /// Create a new named integer variable with the given bounds.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can also create such a variable with a name
    /// let named_integer_between_bounds = solver.new_named_bounded_integer(0, 10, "x");
    /// ```
    pub fn new_named_bounded_integer(
        &mut self,
        lower_bound: i32,
        upper_bound: i32,
        name: impl Into<String>,
    ) -> DomainId {
        self.satisfaction_solver.create_new_integer_variable(
            lower_bound,
            upper_bound,
            Some(name.into()),
        )
    }

    /// Create a new integer variable which has a domain of predefined values. We remove duplicates
    /// by converting to a hash set
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can also create an integer variable with a non-continuous domain in the follow way
    /// let mut sparse_integer = solver.new_sparse_integer(vec![0, 3, 5]);
    /// ```
    pub fn new_sparse_integer(&mut self, values: impl Into<Vec<i32>>) -> DomainId {
        let values: HashSet<i32> = values.into().into_iter().collect();

        self.satisfaction_solver
            .create_new_integer_variable_sparse(values.into_iter().collect(), None)
    }

    /// Create a new named integer variable which has a domain of predefined values.
    ///
    /// # Example
    /// ```rust
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// // We can also create such a variable with a name
    /// let named_sparse_integer = solver.new_named_sparse_integer(vec![0, 3, 5], "y");
    /// ```
    pub fn new_named_sparse_integer(
        &mut self,
        values: impl Into<Vec<i32>>,
        name: impl Into<String>,
    ) -> DomainId {
        self.satisfaction_solver
            .create_new_integer_variable_sparse(values.into(), Some(name.into()))
    }
}

/// Functions for solving with the constraints that have been added to the [`Solver`].
impl Solver {
    /// Solves the current model in the [`Solver`] until it finds a solution (or is indicated to
    /// terminate by the provided [`TerminationCondition`]) and returns a [`SatisfactionResult`]
    /// which can be used to obtain the found solution or find other solutions.
    pub fn satisfy<B: Brancher, T: TerminationCondition>(
        &mut self,
        brancher: &mut B,
        termination: &mut T,
    ) -> SatisfactionResult {
        match self.satisfaction_solver.solve(termination, brancher) {
            CSPSolverExecutionFlag::Feasible => {
                let solution: Solution = self.satisfaction_solver.get_solution_reference().into();
                self.satisfaction_solver.restore_state_at_root(brancher);
                self.process_solution(&solution, brancher);
                SatisfactionResult::Satisfiable(solution)
            }
            CSPSolverExecutionFlag::Infeasible => {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                let _ = self.satisfaction_solver.conclude_proof_unsat();

                SatisfactionResult::Unsatisfiable
            }
            CSPSolverExecutionFlag::Timeout => {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                SatisfactionResult::Unknown
            }
        }
    }

    pub fn get_solution_iterator<
        'this,
        'brancher,
        'termination,
        B: Brancher,
        T: TerminationCondition,
    >(
        &'this mut self,
        brancher: &'brancher mut B,
        termination: &'termination mut T,
    ) -> SolutionIterator<'this, 'brancher, 'termination, B, T> {
        SolutionIterator::new(self, brancher, termination)
    }

    /// Solves the current model in the [`Solver`] until it finds a solution (or is indicated to
    /// terminate by the provided [`TerminationCondition`]) and returns a [`SatisfactionResult`]
    /// which can be used to obtain the found solution or find other solutions.
    ///
    /// This method takes as input a list of [`Literal`]s which represent so-called assumptions (see
    /// \[1\] for a more detailed explanation). The [`Literal`]s corresponding to [`Predicate`]s
    /// over [`IntegerVariable`]s (e.g. lower-bound predicates) can be retrieved from the [`Solver`]
    /// using [`Solver::get_literal`].
    ///
    /// # Bibliography
    /// \[1\] N. Eén and N. Sörensson, ‘Temporal induction by incremental SAT solving’, Electronic
    /// Notes in Theoretical Computer Science, vol. 89, no. 4, pp. 543–560, 2003.
    pub fn satisfy_under_assumptions<'this, 'brancher, B: Brancher, T: TerminationCondition>(
        &'this mut self,
        brancher: &'brancher mut B,
        termination: &mut T,
        assumptions: &[Literal],
    ) -> SatisfactionResultUnderAssumptions<'this, 'brancher, B> {
        match self
            .satisfaction_solver
            .solve_under_assumptions(assumptions, termination, brancher)
        {
            CSPSolverExecutionFlag::Feasible => {
                let solution: Solution = self.satisfaction_solver.get_solution_reference().into();
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                brancher.on_solution(solution.as_reference());
                SatisfactionResultUnderAssumptions::Satisfiable(solution)
            }
            CSPSolverExecutionFlag::Infeasible => {
                if self
                    .satisfaction_solver
                    .state
                    .is_infeasible_under_assumptions()
                {
                    // The state is automatically reset when we return this result
                    SatisfactionResultUnderAssumptions::UnsatisfiableUnderAssumptions(
                        UnsatisfiableUnderAssumptions::new(&mut self.satisfaction_solver, brancher),
                    )
                } else {
                    // Reset the state whenever we return a result
                    self.satisfaction_solver.restore_state_at_root(brancher);
                    SatisfactionResultUnderAssumptions::Unsatisfiable
                }
            }
            CSPSolverExecutionFlag::Timeout => {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                SatisfactionResultUnderAssumptions::Unknown
            }
        }
    }

    /// Solves the model currently in the [`Solver`] to optimality where the provided
    /// `objective_variable` is minimised (or is indicated to terminate by the provided
    /// [`TerminationCondition`]).
    ///
    /// It returns an [`OptimisationResult`] which can be used to retrieve the optimal solution if
    /// it exists.
    pub fn minimise(
        &mut self,
        brancher: &mut impl Brancher,
        termination: &mut impl TerminationCondition,
        objective_variable: DomainId,
        core_guided_options: Option<CoreGuidedArgs>,
        objective_definition: Option<Vec<(i32, DomainId)>>,
    ) -> OptimisationResult {
        match core_guided_options {
            Some(x) => {
                let true_objective = match objective_definition {
                    Some(y) => y,
                    None => vec![(1_i32, objective_variable)],
                };
                self.minimise_internal_cgs(
                    brancher,
                    termination,
                    x,
                    true_objective,
                    objective_variable,
                    false,
                )
            }
            _ => self.minimise_internal(brancher, termination, objective_variable, false),
        }
    }

    /// Solves the model currently in the [`Solver`] to optimality where the provided
    /// `objective_variable` is maximised (or is indicated to terminate by the provided
    /// [`TerminationCondition`]).
    ///
    /// It returns an [`OptimisationResult`] which can be used to retrieve the optimal solution if
    /// it exists.
    pub fn maximise(
        &mut self,
        brancher: &mut impl Brancher,
        termination: &mut impl TerminationCondition,
        objective_variable: DomainId,
        core_guided_options: Option<CoreGuidedArgs>,
        objective_definition: Option<Vec<(i32, DomainId)>>,
    ) -> OptimisationResult {
        match core_guided_options {
            Some(x) => {
                let true_objective = match objective_definition {
                    Some(y) => y.into_iter().map(|(w, v)| (-w, v)).collect(),
                    None => vec![(-1_i32, objective_variable)],
                };
                self.minimise_internal_cgs(
                    brancher,
                    termination,
                    x,
                    true_objective,
                    objective_variable,
                    true,
                )
            }
            _ => self.minimise_internal(brancher, termination, objective_variable.scaled(-1), true),
        }
    }

    /// The internal method which optimizes the objective function, this function takes an extra
    /// argument (`is_maximising`) as compared to [`Solver::maximise`] and [`Solver::minimise`]
    /// which determines whether the logged objective value should be scaled by `-1` or not.
    ///
    /// This is necessary due to the fact that [`Solver::maximise`] simply calls minimise with
    /// the objective variable scaled with `-1` which would lead to incorrect statistic if not
    /// scaled back.
    fn minimise_internal(
        &mut self,
        brancher: &mut impl Brancher,
        termination: &mut impl TerminationCondition,
        objective_variable: impl IntegerVariable,
        is_maximising: bool,
    ) -> OptimisationResult {
        // If we are maximising then when we simply scale the variable by -1, however, this will
        // lead to the printed objective value in the statistics to be multiplied by -1; this
        // objective_multiplier ensures that the objective is correctly logged.
        let objective_multiplier = if is_maximising { -1 } else { 1 };

        let initial_solve = self.satisfaction_solver.solve(termination, brancher);
        match initial_solve {
            CSPSolverExecutionFlag::Feasible => {}
            CSPSolverExecutionFlag::Infeasible => {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                let _ = self.satisfaction_solver.conclude_proof_unsat();
                return OptimisationResult::Unsatisfiable;
            }
            CSPSolverExecutionFlag::Timeout => {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                return OptimisationResult::Unknown;
            }
        }
        let mut best_objective_value = Default::default();
        let mut best_solution = Solution::default();

        self.update_best_solution_and_process(
            objective_multiplier,
            &objective_variable,
            &mut best_objective_value,
            &mut best_solution,
            brancher,
        );

        loop {
            self.satisfaction_solver.restore_state_at_root(brancher);

            let objective_bound_predicate = if is_maximising {
                predicate![objective_variable <= best_objective_value as i32]
            } else {
                predicate![objective_variable >= best_objective_value as i32]
            };

            let objective_bound_literal = self
                .satisfaction_solver
                .get_literal(objective_bound_predicate);

            if self
                .strengthen(
                    &objective_variable,
                    best_objective_value * objective_multiplier as i64,
                )
                .is_err()
            {
                // Reset the state whenever we return a result
                self.satisfaction_solver.restore_state_at_root(brancher);
                let _ = self
                    .satisfaction_solver
                    .conclude_proof_optimal(objective_bound_literal);

                return OptimisationResult::Optimal(best_solution);
            }

            let solve_result = self.satisfaction_solver.solve(termination, brancher);
            match solve_result {
                CSPSolverExecutionFlag::Feasible => {
                    self.debug_bound_change(
                        &objective_variable,
                        best_objective_value * objective_multiplier as i64,
                    );
                    self.update_best_solution_and_process(
                        objective_multiplier,
                        &objective_variable,
                        &mut best_objective_value,
                        &mut best_solution,
                        brancher,
                    );
                }
                CSPSolverExecutionFlag::Infeasible => {
                    {
                        // Reset the state whenever we return a result
                        self.satisfaction_solver.restore_state_at_root(brancher);
                        let _ = self
                            .satisfaction_solver
                            .conclude_proof_optimal(objective_bound_literal);
                        return OptimisationResult::Optimal(best_solution);
                    }
                }
                CSPSolverExecutionFlag::Timeout => {
                    // Reset the state whenever we return a result
                    self.satisfaction_solver.restore_state_at_root(brancher);
                    return OptimisationResult::Satisfiable(best_solution);
                }
            }
        }
    }

    /// Processes a solution when it is found, it consists of the following procedure:
    /// - Assigning `best_objective_value` the value assigned to `objective_variable` (multiplied by
    ///   `objective_multiplier`).
    /// - Storing the new best solution in `best_solution`.
    /// - Calling [`Brancher::on_solution`] on the provided `brancher`.
    /// - Logging the statistics using [`Solver::log_statistics_with_objective`].
    /// - Calling the solution callback stored in [`Solver::solution_callback`].
    fn update_best_solution_and_process(
        &self,
        objective_multiplier: i32,
        objective_variable: &impl IntegerVariable,
        best_objective_value: &mut i64,
        best_solution: &mut Solution,
        brancher: &mut impl Brancher,
    ) {
        *best_objective_value = (objective_multiplier
            * self
                .satisfaction_solver
                .get_assigned_integer_value(objective_variable)
                .expect("expected variable to be assigned")) as i64;
        *best_solution = self.satisfaction_solver.get_solution_reference().into();

        self.internal_process_solution(best_solution, brancher, Some(*best_objective_value))
    }

    pub(crate) fn process_solution(&self, solution: &Solution, brancher: &mut impl Brancher) {
        self.internal_process_solution(solution, brancher, None)
    }

    fn internal_process_solution(
        &self,
        solution: &Solution,
        brancher: &mut impl Brancher,
        objective_value: Option<i64>,
    ) {
        brancher.on_solution(self.satisfaction_solver.get_solution_reference());

        (self.solution_callback)(SolutionCallbackArguments::new(
            self,
            solution,
            objective_value,
        ));
    }

    /// Given the current objective value `best_objective_value`, it adds a constraint specifying
    /// that the objective value should be at most `best_objective_value - 1`. Note that it is
    /// assumed that we are always minimising the variable.
    fn strengthen(
        &mut self,
        objective_variable: &impl IntegerVariable,
        best_objective_value: i64,
    ) -> Result<(), ConstraintOperationError> {
        self.satisfaction_solver
            .add_clause([self.satisfaction_solver.get_literal(
                objective_variable.upper_bound_predicate((best_objective_value - 1) as i32),
            )])
    }

    fn debug_bound_change(
        &self,
        objective_variable: &impl IntegerVariable,
        best_objective_value: i64,
    ) {
        pumpkin_assert_simple!(
            (self
                .satisfaction_solver
                .get_assigned_integer_value(objective_variable)
                .expect("expected variable to be assigned") as i64)
                < best_objective_value,
            "{}",
            format!(
                "The current bound {} should be smaller than the previous bound {}",
                self.satisfaction_solver
                    .get_assigned_integer_value(objective_variable)
                    .expect("expected variable to be assigned"),
                best_objective_value
            )
        );
    }
}

/// Functions for adding new constraints to the solver.
impl Solver {
    /// Add a constraint to the solver. This returns a [`ConstraintPoster`] which enables control
    /// on whether to add the constraint as-is, or whether to (half) reify it.
    ///
    /// If none of the methods on [`ConstraintPoster`] are used, the constraint _is not_ actually
    /// added to the solver. In this case, a warning is emitted.
    ///
    /// # Example
    /// ```
    /// # use pumpkin_solver::constraints;
    /// # use pumpkin_solver::Solver;
    /// let mut solver = Solver::default();
    ///
    /// let a = solver.new_bounded_integer(0, 3);
    /// let b = solver.new_bounded_integer(0, 3);
    ///
    /// solver.add_constraint(constraints::equals([a, b], 0)).post();
    /// ```
    pub fn add_constraint<Constraint>(
        &mut self,
        constraint: Constraint,
    ) -> ConstraintPoster<'_, Constraint> {
        ConstraintPoster::new(self, constraint)
    }

    /// Creates a clause from `literals` and adds it to the current formula.
    ///
    /// If the formula becomes trivially unsatisfiable, a [`ConstraintOperationError`] will be
    /// returned. Subsequent calls to this method will always return an error, and no
    /// modification of the solver will take place.
    pub fn add_clause(
        &mut self,
        clause: impl IntoIterator<Item = Literal>,
    ) -> Result<(), ConstraintOperationError> {
        self.satisfaction_solver.add_clause(clause)
    }

    /// Adds a propagator with a tag, which is used to identify inferences made by this propagator
    /// in the proof log.
    pub(crate) fn add_tagged_propagator(
        &mut self,
        propagator: impl Propagator + 'static,
        tag: NonZero<u32>,
    ) -> Result<(), ConstraintOperationError> {
        self.satisfaction_solver
            .add_propagator(propagator, Some(tag))
    }

    /// Post a new propagator to the solver. If unsatisfiability can be immediately determined
    /// through propagation, this will return a [`ConstraintOperationError`].
    ///
    /// The caller should ensure the solver is in the root state before calling this, either
    /// because no call to [`Self::solve()`] has been made, or because
    /// [`Self::restore_state_at_root()`] was called.
    ///
    /// If the solver is already in a conflicting state, i.e. a previous call to this method
    /// already returned `false`, calling this again will not alter the solver in any way, and
    /// `false` will be returned again.
    pub(crate) fn add_propagator(
        &mut self,
        propagator: impl Propagator + 'static,
    ) -> Result<(), ConstraintOperationError> {
        self.satisfaction_solver.add_propagator(propagator, None)
    }
}

/// Default brancher implementation
impl Solver {
    /// Creates a default [`IndependentVariableValueBrancher`] which uses [`Vsids`] as
    /// [`VariableSelector`] and [`SolutionGuidedValueSelector`] (with [`PhaseSaving`] as its
    /// back-up selector) as its [`ValueSelector`]; it searches over all
    /// [`PropositionalVariable`]s defined in the provided `solver`.
    pub fn default_brancher_over_all_propositional_variables(&self) -> DefaultBrancher {
        self.satisfaction_solver
            .default_brancher_over_all_propositional_variables()
    }
}

/// Proof logging methods
impl Solver {
    #[doc(hidden)]
    /// Conclude the proof with the unsatisfiable claim.
    ///
    /// This method will finish the proof. Any new operation will not be logged to the proof.
    pub fn conclude_proof_unsat(&mut self) -> std::io::Result<()> {
        self.satisfaction_solver.conclude_proof_unsat()
    }

    #[doc(hidden)]
    /// Conclude the proof with the optimality claim.
    ///
    /// This method will finish the proof. Any new operation will not be logged to the proof.
    pub fn conclude_proof_optimal(&mut self, bound: Literal) -> std::io::Result<()> {
        self.satisfaction_solver.conclude_proof_optimal(bound)
    }

    pub(crate) fn into_satisfaction_solver(self) -> ConstraintSatisfactionSolver {
        self.satisfaction_solver
    }
}

/// The type of [`Brancher`] which is created by
/// [`Solver::default_brancher_over_all_propositional_variables`].
///
/// It consists of the value selector
/// [`Vsids`] in combination with a [`SolutionGuidedValueSelector`] with as backup [`PhaseSaving`].
pub type DefaultBrancher = IndependentVariableValueBrancher<
    PropositionalVariable,
    Vsids<PropositionalVariable>,
    SolutionGuidedValueSelector<
        PropositionalVariable,
        bool,
        PhaseSaving<PropositionalVariable, bool>,
    >,
>;

/// When using core-guided search, certain [`IntegerPredicate`]s may need to be analysed more
/// in-depth. This struct allows the information from [`IntegerPredicate`]s to be stored in a more
/// convenient format, to allow its elements to be accessed more easily.
#[derive(Debug, Clone)]
pub(crate) struct DecomposedPredicate {
    variable: DomainId,
    is_greater: bool,
    bound: i32,
}

/// In core-guided search, stratification can be used to find intermediate (increasingly good)
/// solutions. When used, stratification needs to divide the objective function into strata,
/// the steps of which are conveniently bundled into this object. It can be used as a
/// (pseudo-)iterator by iteratively calling [`StratificationPartitioner::next_stratum`], or all
/// strata can be extracted at once by calling [`StratificationPartitioner::all_strata`].
pub(crate) struct StratificationPartitioner {
    weights: Vec<i32>,
    vars: Vec<AffineView<DomainId>>,
    lower_bounds: Vec<i32>,
}

type VarWithBound = (AffineView<DomainId>, i32);
type VarsWithSeparatedBounds = (Vec<AffineView<DomainId>>, Vec<i32>);
impl StratificationPartitioner {
    pub(crate) fn new(
        objective_function: impl Iterator<Item = (i32, VarWithBound)>,
    ) -> StratificationPartitioner {
        // Clone the input and sort by weights.
        let mut tmp = objective_function.collect::<Vec<_>>();
        // Sort increasing, this way [`Vec::pop`] returns the highest-weighing variable.
        tmp.sort_by(|a, b| a.0.cmp(&b.0));
        // Split into components
        let (weights, (vars, lower_bounds)): (Vec<i32>, VarsWithSeparatedBounds) =
            tmp.into_iter().unzip();
        StratificationPartitioner {
            weights,
            vars,
            lower_bounds,
        }
    }

    /// Returns a vector of the highest-weighing elements left in the objective function.
    /// These are returned as tuples of a (possibly scaled) [`DomainId`] and an i32 representing
    /// the original lower bound; this to calculate any incurred cost. Furthermore, an i32
    /// is returned which represents the weight of all these elements.
    pub(crate) fn next_stratum(&mut self) -> (Vec<VarWithBound>, i32) {
        assert_eq!(self.weights.len(), self.vars.len());
        assert_eq!(self.lower_bounds.len(), self.vars.len());
        if self.weights.is_empty() {
            (vec![], 0)
        } else {
            let mut res = vec![];
            // Take the first weight; this is the weight of this stratum.
            let w = self.weights.pop().unwrap();
            // Take corresponding variable and lower bound as well.
            res.push((self.vars.pop().unwrap(), self.lower_bounds.pop().unwrap()));

            loop {
                match self.weights.pop() {
                    // While the weight is the same, we're still in this stratum.
                    Some(v) if v == w => {
                        res.push((self.vars.pop().unwrap(), self.lower_bounds.pop().unwrap()))
                    }
                    // If the weight changes, re-add the weight, and stop collecting. The stratum
                    // is exhausted.
                    Some(x) => {
                        self.weights.push(x);
                        break;
                    }
                    // If the list is exhausted, the stratum is as well.
                    None => break,
                }
            }

            (res, w)
        }
    }

    /// Returns a vector of all strata, in order of ascending weights. This way, [`Vec::pop`]
    /// will return the highest-weighing stratum.
    pub(crate) fn all_strata(&mut self) -> Vec<(Vec<VarWithBound>, i32)> {
        let mut res = vec![];
        loop {
            // Collect strata until the object is exhausted.
            let strat = self.next_stratum();
            match strat.0.is_empty() {
                true => break,
                false => res.insert(0, strat),
            }
        }

        res
    }
}

/// Core-guided search implementation of the solver
impl Solver {
    /// Helper function to get the [`DomainId`] corresponding to a given literal.
    pub fn get_domain_literal(&self, lit: Literal) -> Option<DomainId> {
        self.satisfaction_solver
            .variable_literal_mappings
            .get_domain_literal(lit)
    }

    /// Helper function to retrieve all [`IntegerPredicate`]s corresponding to a given literal.
    /// Used to retrieve the (single) [`LowerBound`] or  [`UpperBound`] predicate used in the
    /// assumptions.
    pub fn get_integer_predicates_from_literal(
        &self,
        lit: Literal,
    ) -> impl Iterator<Item = IntegerPredicate> + '_ {
        self.satisfaction_solver
            .variable_literal_mappings
            .get_predicates(lit)
    }

    /// Removes the inverses of a given slice of [`Literal`]s from a vector of [`Literal`]s.
    /// This is used to remove the assumptions belonging to a core from the assumptions currently
    /// used when solving.
    fn remove_core_from_assumptions(core: &[Literal], assumptions: &mut Vec<Literal>) {
        let core_set: HashSet<&Literal> = HashSet::from_iter((*core).iter());
        // Remove assumptions occurring in core.
        assumptions.retain(|a| !core_set.contains(a));
    }

    /// Internal minimisation algorithm using core-guided search. Takes as input several of the same
    /// inputs as the linear minimisation algorithm, in addition to the definition of the
    /// objective function, and two flags determining the core-guided approach.
    pub fn minimise_internal_cgs(
        &mut self,
        brancher: &mut impl Brancher,
        termination: &mut impl TerminationCondition,
        core_guided_options: CoreGuidedArgs,
        objective_function: Vec<(i32, DomainId)>,
        objective_variable: DomainId,
        _is_maximising: bool,
    ) -> OptimisationResult {
        let mut lb_stat = LowerBoundEvolutionMonitor::new();
        let mut core_stat = CoreSizeMonitor::new();
        let mut task_stat = TimePerTaskMonitor::new();
        let mut wce_stat = WCECoreAmountMonitor::new();
        let mut hard_stat = HardeningDomainLimitationMonitor::new();
        let mut exh_stat = CoreExhaustionMonitor::new();

        task_stat.start_task(MonitoredTasks::Overhead);
        debug!(
            "Start of CGS minimisation, with following options: {:?}",
            core_guided_options
        );
        let mut unadded_parts_of_objective_function;
        if core_guided_options.stratification {
            task_stat.start_task(MonitoredTasks::StrataCreation);
            unadded_parts_of_objective_function = StratificationPartitioner::new(
                // Add lower bounds and pass to function.
                objective_function
                    .iter()
                    .map(|(w, did)| (w, did.scaled(w.signum())))
                    .map(|(w, did)| {
                        let lb = self.lower_bound(&did);
                        (*w, (did, lb))
                    }),
            )
            .all_strata(); // Extract relevant data
            task_stat.end_task(MonitoredTasks::StrataCreation);
            debug!(
                "Stratification active, number of strata: {}",
                unadded_parts_of_objective_function.len(),
            );
        } else {
            unadded_parts_of_objective_function = vec![(
                objective_function
                    .iter()
                    .cloned()
                    .map(|(w, did)| did.scaled(w.signum()))
                    .map(|did| {
                        let lb = self.lower_bound(&did);
                        (did, lb)
                    })
                    .collect(),
                0,
            )]
        }

        // Add assumptions which fix all objective variables to their lower bounds;
        // create literals for these conditions.
        let (initial_stratum, _) = unadded_parts_of_objective_function
            .pop()
            .expect("Expected at least 1 objective function term");
        let (mut assumptions, _) = self.create_assumptions_with_lb_diff(initial_stratum);

        let mut solution: Option<Solution> = None;
        // This boolean signals whether the result of the solver is 'final'; if a solution is found
        // with no information left to be added, or if unconditional unsatisfiability is found.
        let mut proven = false;
        // Keep track of the lower bound on the objective, as governed by the assumptions.
        // NOTE: this assumes there is no bias in the objective variable. If the domain is
        // explicitly limited, or the definition of the objective variable includes a bias term,
        // the value calculated here is incorrect.
        let signed_objective = objective_variable.scaled(match _is_maximising {
            true => -1,
            false => 1,
        });
        let mut current_lower_bound = self.lower_bound(&signed_objective);
        debug!("Initializing lower bound as {}", current_lower_bound);

        let mut weights_per_var: HashMap<u32, (i32, i32)> = HashMap::from(
            objective_function
                .iter()
                .map(|(w, did)| (did.id, (*w, *w)))
                .collect(),
        );

        // In the case of weight-aware core extraction (WCE), we fully process a core but don't yet
        // add the new assumptions. Keep track of the assumptions to be added.
        // Note: all reformulation variables have weight >=0
        let mut delayed_assumptions: Vec<(DomainId, i32)> = vec![];

        // Repeat until satisfiable
        loop {
            // Check if the current assumed objective value is still below the (true) upper bound
            if current_lower_bound < self.lower_bound(&signed_objective) {
                warn!(
                    "Warn! LB estimate under-estimates: {} instead of {}; delayed info? {}",
                    current_lower_bound,
                    self.lower_bound(&signed_objective),
                    !unadded_parts_of_objective_function.is_empty()
                        || !delayed_assumptions.is_empty(),
                );
                if !unadded_parts_of_objective_function.is_empty()
                    || !delayed_assumptions.is_empty()
                {
                    debug!("Not updating LB estimate due to delayed info");
                } else {
                    debug!("Updating LB estimate to current lower bound");
                    current_lower_bound = self.lower_bound(&signed_objective);
                }
            }

            let (actual_lb, actual_ub) = (
                self.lower_bound(&signed_objective),
                self.upper_bound(&signed_objective),
            );
            // Update lower bound every iteration
            lb_stat.update(current_lower_bound);

            if current_lower_bound > actual_ub
                || actual_lb > actual_ub
                || assumptions.contains(&Literal::u32_to_literal(0))
            {
                info!("Stopping solving process due to violated bounds");
                debug!(
                    "est. lb > ub: {}; act. lb > ub: {}; assume false: {}",
                    current_lower_bound > actual_ub,
                    actual_lb > actual_ub,
                    assumptions.contains(&Literal::u32_to_literal(0)),
                );
                break;
            }

            let core = {
                task_stat.start_task(MonitoredTasks::Solving);
                let solv_res = self.satisfy_under_assumptions(brancher, termination, &assumptions);
                task_stat.end_task(MonitoredTasks::Solving);

                match solv_res {
                    SatisfactionResultUnderAssumptions::Satisfiable(sol) => {
                        debug!("New solution found");
                        // Extract solution.
                        solution = match solution {
                            Some(old_sol)
                                if old_sol.lower_bound(&signed_objective)
                                    < sol.lower_bound(&signed_objective) =>
                            {
                                // If the old solution is better than the new one, don't replace.
                                // Note: minimisation, so better means that (old_obj < new_obj)
                                debug!("Keeping old solution");
                                Some(old_sol)
                            }
                            _ => {
                                debug!("Using new solution as best so far");
                                Some(sol)
                            }
                        };

                        if delayed_assumptions.is_empty()
                            && unadded_parts_of_objective_function.is_empty()
                        {
                            info!("Newly found solution is proven to be optimal; exiting...");
                            // If no information is held back, this is the final solution, and thus
                            // optimal. The cores are the proof of this. In other words, the
                            // current solution is proven to be optimal.
                            proven = true;
                            break;
                        } else {
                            debug!("Information remaining; preparing to add info");
                            None
                        }
                    }
                    SatisfactionResultUnderAssumptions::Unknown => {
                        info!("Result Unknown returned; presumably timeout");
                        break;
                    }
                    SatisfactionResultUnderAssumptions::Unsatisfiable => {
                        // If the solver reports (unconditional) unsatisfiability, this must mean
                        // the base problem is unsatisfiable. If this is
                        // indeed the case (and a previous solution does not
                        // exist), we have proven usatisfiability.
                        info!("UNSAT proven; exiting...");
                        proven = solution.is_none();
                        if !proven {
                            warn!(
                                "A solution has been found before UNSAT; closer inspection \
                                required. Note: hardening was {}",
                                core_guided_options.harden,
                            );
                        }
                        break;
                    }
                    SatisfactionResultUnderAssumptions::UnsatisfiableUnderAssumptions(
                        mut core_extractor,
                    ) => {
                        debug!("Core found");
                        // If a core is encountered, we process the core and continue solving.
                        Some(core_extractor.extract_core())
                    }
                }
            };

            if let Some(c) = core {
                info!("Processing core of size {}", c.len());
                core_stat.core_found(c.len());

                task_stat.start_task(MonitoredTasks::CoreProcessing);
                let (add_cost, add_assum) = self.process_core(
                    &c,
                    &mut assumptions,
                    core_guided_options.variable_reformulation,
                    core_guided_options.coefficient_elimination,
                    &mut weights_per_var,
                );
                task_stat.end_task(MonitoredTasks::CoreProcessing);

                current_lower_bound += add_cost;

                if let Some(assum) = add_assum {
                    debug!("Assumption was returned: {:?}", assum);
                    let (did, lb) = assum;
                    // TODO perform core exhaustion
                    let new_lb = lb;
                    exh_stat.record_exhaustion(new_lb as f32 / lb as f32);
                    if core_guided_options.weight_aware_cores {
                        delayed_assumptions.push((did, new_lb));
                    } else {
                        debug!("Added to assumptions");
                        assumptions.push(self.get_literal(predicate!(did <= new_lb)));
                    }
                }
            } else if let Some(sol) = solution {
                // There was a solution found; since no `break` has been encountered, there is
                // information which was held back. Add this information where needed.

                // WCE: check if previously extracted cores still need to be added.
                if !delayed_assumptions.is_empty() {
                    debug!("Adding delayed assumptions: {}", delayed_assumptions.len());

                    task_stat.start_task(MonitoredTasks::WCEAdditions);
                    wce_stat.record_reformulation(delayed_assumptions.len());
                    let (additional_assum, additional_cost) =
                        self.process_wce(&mut delayed_assumptions, &mut weights_per_var);
                    assumptions.extend(additional_assum);
                    debug!(
                        "LB's of delayed assumptions incurred additional cost: {}",
                        additional_cost,
                    );
                    current_lower_bound += additional_cost;
                    task_stat.end_task(MonitoredTasks::WCEAdditions);
                }

                // Stratification: check if strata still need to be added.
                if !unadded_parts_of_objective_function.is_empty() {
                    debug!(
                        "{} strata remaining",
                        unadded_parts_of_objective_function.len()
                    );
                    // Unpack stratum.
                    task_stat.start_task(MonitoredTasks::StrataCreation);
                    let (stratum, weight) = unadded_parts_of_objective_function.pop().unwrap();
                    debug!("Adding stratum of length {} (w: {})", stratum.len(), weight);
                    // Make assumptions from stratum, and calculate lower bound difference.
                    let (new_assum, diff) = self.create_assumptions_with_lb_diff(stratum);
                    assumptions.extend(new_assum);

                    let cost = weight * diff.iter().sum::<i32>();
                    if cost > 0 {
                        debug!("LB's of stratum incurred additional cost: {}", cost);
                    }
                    current_lower_bound += cost;
                    task_stat.end_task(MonitoredTasks::StrataCreation);
                }

                // Hardening: add the upper bounds explicitly.
                if core_guided_options.harden {
                    let mapped_obj_val = sol.lower_bound(&signed_objective);
                    debug!("Hardening obj to {}", mapped_obj_val);
                    // Harden the full objective; useful for inference.
                    less_than_or_equals(vec![signed_objective.clone()], mapped_obj_val)
                        .post(self, None)
                        .expect("Could not harden");

                    // Calculate the domain gap and use this to calculate the new domain sizes
                    // of each variable (for statistics, as well as through constraints)
                    // Note: we CANNOT use the lower bound estimate for this, as parts of its cost
                    // are not reflected in the objective variables.
                    let inferred_lb = objective_function
                        .iter()
                        .map(|(w, did)| self.lower_bound(did) * w)
                        .sum::<i32>();
                    let gap = mapped_obj_val - inferred_lb;
                    let fraction: f32 = objective_function
                        .iter()
                        .map(|(w, did)| {
                            (self.upper_bound(did) - self.lower_bound(did) + 1) as f32
                                / (gap / w.abs() + 1) as f32
                        })
                        .filter(|frac| *frac < 1.0)
                        .product();
                    hard_stat.hardened(fraction);
                    // Add aforementioned constraints.
                    for (w, did) in &objective_function {
                        let scaled = did.scaled(*w);
                        let bound = self.lower_bound(&scaled) + gap;
                        if self.upper_bound(&scaled) > bound {
                            less_than_or_equals(vec![scaled], bound)
                                .post(self, None)
                                .expect("Could not add hardening element");
                        }
                    }
                }
                solution = Some(sol);
            }
        }

        debug!("Exited loop, returning answer...");
        task_stat.end_task(MonitoredTasks::Overhead);

        let lb_res = lb_stat.get_result();
        let core_res = core_stat.get_result();
        let task_res = task_stat.get_result();
        let wce_res = wce_stat.get_result();
        let hard_res = hard_stat.get_result();
        let exh_res = exh_stat.get_result();
        println!(
            "The following custom statistics were collected:\n\
            Lower Bounds: {:?}\n\
            Core Size: {:?}\n\
            Time per Task: {:?}\n\
            Number of Disjoint Cores between Reformulations: {:?}\n\
            Remaining Domain Fractions after Hardening: {:?}\n\
            Relative Increase in Lower Bound after Exhaustion: {:?}",
            lb_res, core_res, task_res, wce_res, hard_res, exh_res,
        );

        let lb_last_result = lb_res.last().expect("Expected lower bound");
        let lb_len = lb_stat.get_result().len();
        let core_len = core_res.len();

        // Calculate optimal slope for linear approximation (using Gauss-Markov theorem)
        // for both core weight (= lb increase) and core size
        // https://doi.org/10.1016/0378-3758(91)90016-8
        let lb_incs: Vec<i32> = lb_res
            .iter()
            .zip(lb_res.iter().skip(1))
            .map(|(&(first_lb, _), &(second_lb, _))| *second_lb - *first_lb)
            .collect();
        let lb_step_size = *lb_last_result.0 as f32 / lb_len as f32;

        // xs are the number of cores, ys are the lower bound increases
        let (xy, x2): (Vec<i32>, Vec<i32>) = lb_incs
            .iter()
            .enumerate()
            .map(|(a, &b)| (a as i32 * b, (a * a) as i32))
            .unzip();
        let sum_x = ((lb_len * (lb_len + 1)) / 2) as i32;
        let slope_lb = (lb_len as i32 * xy.iter().sum::<i32>()
            - sum_x * lb_incs.iter().sum::<i32>()) as f64
            / (lb_len as i32 * x2.iter().sum::<i32>() - sum_x * sum_x) as f64;

        let core_size = core_res.iter().sum::<usize>() as f32 / core_len as f32;
        // xs are the number of cores, ys are the core sizes
        let (xy, x2): (Vec<i32>, Vec<i32>) = core_res
            .iter()
            .enumerate()
            .map(|(a, &b)| ((a * b) as i32, (a * a) as i32))
            .unzip();
        let sum_x = ((core_len * (core_len + 1)) / 2) as i32;
        let sum_y = core_res.iter().map(|a| *a as i32).sum::<i32>();
        let slope_core = (core_len as i32 * xy.iter().sum::<i32>() - sum_x * sum_y) as f64
            / (core_len as i32 * x2.iter().sum::<i32>() - sum_x * sum_x) as f64;

        let time_solv = **task_res.get(&MonitoredTasks::Solving).unwrap_or(&&0);
        let time_core = **task_res.get(&MonitoredTasks::CoreProcessing).unwrap_or(&&0);
        let time_spec = **task_res.get(&MonitoredTasks::WCEAdditions).unwrap_or(&&0)
            + **task_res.get(&MonitoredTasks::StrataCreation).unwrap_or(&&0);

        let core_per_reform = wce_res.iter().sum::<usize>() as f32 / wce_res.len() as f32;

        let unhard_fraction = hard_res.iter().product::<f32>();
        let exh_fraction = exh_res.iter().product::<f32>();

        println!(
            "This is summarised in the following data points:\n\
            Lower Bound Steps: {:?}\n\
            Lower Bound Step Size: {:?}\n\
            Lower Bound Update Frequency: {:?}\n\
            Lower Bound Slope: {:?}\n\
            Average Core Size: {:?}\n\
            Core Size Slope: {:?}\n\
            Time Spent in Solver: {:?}\n\
            Time Spent Processing Cores: {:?}\n\
            Time Spent on Special Operations: {:?}\n\
            Average Number of Reformulation Cores: {:?}\n\
            Total Unhardened Fraction: {:?}\n\
            Total Relative Increase of Exhaustion: {:?}",
            lb_len,
            lb_step_size,
            *lb_last_result.1 as f32 / lb_len as f32, // final value / #steps = average step
            slope_lb,
            core_size,
            slope_core,
            time_solv,
            time_core,
            time_spec,
            core_per_reform,
            unhard_fraction,
            exh_fraction,
        );

        let obj = if let Some(s) = &solution {
            s.lower_bound(&objective_variable) as f32
        } else {
            f32::NAN
        };
        let is_optimal = proven && (!obj.is_nan());
        let obj_of_optimal = if is_optimal { obj } else { f32::NAN };

        let total_time = time_solv + time_core + time_spec;
        let nnodes = self.satisfaction_solver.get_number_of_decisions();

        println!("-=- CSV version -=-");
        println!("lb_nstep;lb_size;lb_slope;core_size;core_slope;time_solv;time_core;time_special;ncore_reform;unhard_frac;exh_frac;obj;is_optimal;optimal_found_value;nnodes;total_time");
        println!("{lb_len};{lb_step_size};{slope_lb};{core_size};{slope_core};{time_solv};{time_core};{time_spec};{core_per_reform};{unhard_fraction};{exh_fraction};{obj};{is_optimal};{obj_of_optimal};{nnodes};{total_time}");
        println!("-=- End of CSV -=-");

        match (solution, proven) {
            // If a solution has been found, and was proven to be optimal, return it as the
            // optimal solution.
            (Some(s), true) => OptimisationResult::Optimal(s),
            // If a solution has been found for which optimality could *not* be proven,
            // report satisfiability instead of optimality, and return solution.
            (Some(s), _) => OptimisationResult::Satisfiable(s),
            // If no solution was found at any point, and we have proven that we cannot find one,
            // report that the problem is unsatisfiable.
            (None, true) => OptimisationResult::Unsatisfiable,
            // If we have found no solutions yet, but have also been unable to prove
            // unsatisfiability, the result of the problem is unknown.
            _ => OptimisationResult::Unknown,
        }
    }

    /// Changes the assumptions composing a core to resolve the conflict. Removes the [`Literal`]s
    /// present in the core from the list of assumptions, depending on the weight handling approach
    /// (i.e. lower weight accordingly in the case of weight splitting); adds a new [`Literal`]
    /// corresponding to the reformulation variable. Returns an integer, which represents the
    /// additional cost induced by resolving this core.
    fn process_core(
        &mut self,
        core: &[Literal],
        assumptions: &mut Vec<Literal>,
        variable_reformulation: bool,
        coefficient_elimination: bool,
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
    ) -> (i32, Option<(DomainId, i32)>) {
        // Keep track of the cost incurred by resolving the core.
        let (cost, assum);

        if core.len() == 1 {
            // Unit cores are handled separately, with relatively small differences between the
            // different approaches
            cost =
                self.process_unit_core(core, assumptions, coefficient_elimination, weights_per_var);
            assum = None;
        } else {
            debug!("Core has a length of {}", core.len());
            // Larger cores
            if !variable_reformulation {
                // slice-based
                let (new_cost, new_assum) = self.process_core_slice(
                    assumptions,
                    core,
                    weights_per_var,
                    coefficient_elimination,
                );
                cost = new_cost;
                assum = Some(new_assum);
            } else {
                // variable-based
                let (new_cost, new_assum) = self.process_core_var(
                    assumptions,
                    core,
                    weights_per_var,
                    coefficient_elimination,
                );
                cost = new_cost;
                assum = Some(new_assum);
            }
        }
        assumptions.retain(|l| l.to_u32() != 1);
        debug!("Core incurred cost of {}", cost);
        (cost, assum)
    }

    /// Processes a core of which the length is 1. Requires the core and the list of assumptions,
    /// alongside the associated weights and a flag marking coefficient elimination. Changes the
    /// assumptions to loosen the bound on the violated assumption, and updates the weight
    /// accordingly (if needed). Returns the cost induced by this operation.
    fn process_unit_core(
        &mut self,
        core: &[Literal],
        assumptions: &mut Vec<Literal>,
        coefficient_elimination: bool,
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
    ) -> i32 {
        // For unit cores, bounds are updated through inference.
        // The processing of this core is thus to remove the old assumption with the old
        // (impossible) bound, and add one with the new bound.
        let did = self
            .get_domain_literal(core[0])
            .expect("Must have integer variable");

        debug!("Removing core from assumptions");
        // Unit cores contain a non-negated assumption (i.e. the version present in `assumptions`).
        let _ = assumptions.remove(assumptions.iter().position(|a| a == &core[0]).unwrap());

        // Decompose literal and associated predicate.
        let old_assumption = self.decompose_literal(core[0]).unwrap();
        debug!("Assumption was: {:?}", old_assumption);

        // Keep track how much the bound has increased (can be more than 1, in the case of
        // unit cores), this is needed to calculate the cost increase.
        let diff_in_bound;
        if old_assumption.is_greater {
            // If it was a >= relation, create a new >= relation, with the lowered upper bound
            diff_in_bound = old_assumption.bound - self.upper_bound(&did);
            assumptions.push(self.get_literal(predicate!(did >= self.upper_bound(&did))));
        } else {
            // If it was a <= relation, create <= relation with the higher lower bound
            diff_in_bound = self.lower_bound(&did) - old_assumption.bound;
            assumptions.push(self.get_literal(predicate!(did <= self.lower_bound(&did))));
        }
        debug!(
            "New assumption was accompanied by bound increase of {}",
            diff_in_bound
        );

        self.calculate_cost_of_increased_lb(
            did,
            diff_in_bound,
            weights_per_var,
            !coefficient_elimination,
        )
    }

    /// After a core has been extracted, certain processing steps have to be taken to remove the
    /// unsatisfiability captured in the core. This method performs the steps required to process a
    /// core through slice-based reformulation, and returns the cost and reformulation assumption.
    fn process_core_slice(
        &mut self,
        assumptions: &mut Vec<Literal>,
        core: &[Literal],
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
        weight_elimination: bool,
    ) -> (i32, (DomainId, i32)) {
        let decomposed_core = self.decompose_with_weights(core, weights_per_var);
        debug!(
            "Core consists of the following assumptions: {:?}",
            decomposed_core
        );
        // Note: we take the absolute value of the weights, because the sign of the weight has
        // already been incorporated into the polarity of the predicate corresponding to the
        // provided literal. As such, incorporating the sign of the weight would counteract this
        // effect and yield incorrect results.
        let only_weights: Vec<i32> = decomposed_core.iter().map(|(_, w)| w.0.abs()).collect();

        // Calculate several useful constants.
        let core_len: i32 = core.len() as i32;
        let min_weight: i32 = *only_weights.iter().min().unwrap();
        let sum_weight: i32 = only_weights.iter().sum::<i32>();

        // Instantiate variables that are set inside the conditional, but used outside of it.
        let (var_weights, d, d_weight);
        if weight_elimination {
            // In weight elimination, d is defined using the weighted sum of the elements of the
            // core. This sum itself thus has weight 1, and the relation is defined by the list
            // of residual weights.
            d_weight = 1;
            var_weights = only_weights;

            debug!("Removing core from assumptions");
            // All original assumptions are included in the new variable, and are thus removed.
            Solver::remove_core_from_assumptions(core, assumptions);

            // "Slice off" lowest value, i.e. increase assumption bound.
            debug!("Adding all new slice assumptions");
            for (
                DecomposedPredicate {
                    variable,
                    is_greater,
                    bound,
                },
                _,
            ) in decomposed_core
            {
                assumptions.push(self.get_literal(if is_greater {
                    predicate!(variable >= bound - 1)
                } else {
                    predicate!(variable <= bound + 1)
                }))
            }

            // Since we want to remove the unsatisfiability, we know that at least one of the
            // elements of d needs to be satisfied. As such, a lower bound of 0 is impossible.
            // The lowest feasible option corresponds to the lowest weight of a single element.
            debug!(
                "Adding new variable with domain [{},{}] (w: {})",
                min_weight, sum_weight, d_weight,
            );
            d = self.new_bounded_integer(min_weight, sum_weight);
        } else {
            // In weight splitting, d is defined as the sum of the elements in the core, and an
            // equal fraction of weight is transferred from every element to d. As such, it takes
            // over this weight - corresponding to the lowest among the residual weights - and uses
            // unit weights in its relation to the core.
            d_weight = min_weight;
            var_weights = vec![1_i32; core_len as usize];

            // Update the weights of the elements in the core, and return what assumptions should
            // be removed and added
            let (to_remove, to_add) =
                self.process_weight_split(&decomposed_core, weights_per_var, min_weight, true);
            // Remove the elements marked for removal.
            debug!("Removing marked assumptions from assumptions");
            Solver::remove_core_from_assumptions(&to_remove, assumptions);
            // Add elements marked for addition.
            debug!("Adding {} new (slice) assumptions", to_add.len());
            assumptions.extend_from_slice(&to_add);

            // Note that, to remove the unsatisfiability, at least 1 of the elements of the core
            // needs to be satisfied, and thus d has a lower bound of 1.
            debug!(
                "Adding new variable with domain [1,{}] (w: {})",
                core_len, d_weight
            );
            d = self.new_bounded_integer(1, core_len);
        }

        // Encode relation.
        boolean_equals(var_weights, core.iter().map(|l| !*l).collect::<Vec<_>>(), d)
            .post(self, None)
            .expect("Could not add boolean cardinality constraint");
        // Set weight.
        assert!(weights_per_var.insert(d.id, (d_weight, d_weight)).is_none());

        // The current cost assumes all elements in the core to be false. The lower bound on d
        // corresponds to the number of relaxations needed to remove unsatisfiability, all of which
        // incur a cost of d_weight. This results in the total cost: lb_d * w_d
        // Also return the information needed for the new assumption d <= lb_d
        let lb = self.lower_bound(&d);
        (lb * d_weight, (d, lb))
    }

    /// Similar to the method above, this method performs the steps required to process a
    /// core and returns the cost. However, this method applies variable-based reformulation.
    fn process_core_var(
        &mut self,
        assumptions: &mut Vec<Literal>,
        core: &[Literal],
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
        weight_elimination: bool,
    ) -> (i32, (DomainId, i32)) {
        let decomposed_core = self.decompose_with_weights(core, weights_per_var);
        debug!(
            "Core consists of the following assumptions: {:?}",
            decomposed_core
        );
        // Determine the components of the reformulation variable.
        let mut vars: Vec<AffineView<DomainId>> = decomposed_core
            .iter()
            .map(|(d_pred, (w, _))| {
                d_pred.variable.scaled(match weight_elimination {
                    // For coefficient elimination, we "absorb" all variables and their full
                    // weights into the reformulation variable d.
                    true => *w,
                    // For weight splitting, we want the polarity of all weights to be the same,
                    // to make sure we handle them properly:
                    // 5a - 4b ==> (1+4)a + 4(-b) ==> 1a + 4d; d = a-b
                    _ => w.signum(),
                })
            })
            .collect();

        // Determine the upper and lower bounds of each variable (with adjusted polarity or
        // weight, depending on the weight handling approach).
        let bounds: (Vec<i32>, Vec<i32>) = vars
            .iter()
            .map(|var| (self.lower_bound(var), self.upper_bound(var)))
            .unzip();

        // Summing the respective upper and lower bounds of the variables as determined before,
        // yields the upper and lower bound of the reformulation variable. Note that, to remove
        // infeasibility, the lower bound will need to be increased further.
        let (lb, ub): (i32, i32) = (bounds.0.iter().sum(), bounds.1.iter().sum());

        // The minimum weight is needed to either decrease the weights of core elements, or as an
        // increase for the lower bound on the reformulation variable.
        let min_weight = decomposed_core
            .iter()
            .map(|(_, (w1, _))| w1.abs())
            .min()
            .unwrap();

        // These variables depend on the weight handling approach
        let (d, d_weight);
        if weight_elimination {
            // As with slice-based.
            d_weight = 1;

            // The entire core is "absorbed" by the reformulation, as such they can all be removed
            // from the assumptions
            debug!("Removing core from assumptions");
            Solver::remove_core_from_assumptions(core, assumptions);

            // As with slice-based; since we need to remove the unsatisfiability, the
            // lowest weight of a single element needs to be added to the previously calculated
            // lower bound for a feasible value.
            debug!(
                "Adding new variable with domain: [{}, {}] (w: {})",
                lb + min_weight,
                ub,
                d_weight
            );
            d = self.new_bounded_integer(lb + min_weight, ub);
        } else {
            // As with slice-based.
            d_weight = min_weight;

            // As with slice-based; however, since we work on variables, no new sliced need to be
            // added to the assumptions. These values are also immediately incorporated into the
            // reformulation variable.
            let (to_remove, _) =
                self.process_weight_split(&decomposed_core, weights_per_var, min_weight, false);
            // Remove marked elements.
            debug!("Removing marked elements from assumptions");
            Solver::remove_core_from_assumptions(&to_remove, assumptions);

            // As with slice-based; 1 needs to be added to ensure unsatisfiability is removed.
            debug!(
                "Adding new variable with domain: [{}, {}] (w: {})",
                lb + 1,
                ub,
                d_weight
            );
            d = self.new_bounded_integer(lb + 1, ub);
        }

        // Assign coefficient -1 to reformulation variable:
        // d = a + b + c <=> 0 = a + b + c - d
        vars.push(d.scaled(-1));
        less_than_or_equals(vars, 0)
            .post(self, None)
            .expect("Could not add cardinality constraint");

        // Insert new weight
        assert!(weights_per_var.insert(d.id, (d_weight, d_weight)).is_none());

        // The current total cost caused by the variables considered here is summarised in lb; it
        // contains the weighted sum of all elements from the core. The actual lower bound on d will
        // exceed this; as is needed to remove unsatisfiability. Any increase incurs a cost,
        // of d_weight. As such, the total cost is: (lb_d - lb) * w_d
        // Also return the information needed for the new assumption d <= lb_d
        let lb_d = self.lower_bound(&d);
        ((self.lower_bound(&d) - lb) * d_weight, (d, lb_d))
    }

    /// For both reformulation techniques, the weight splitting procedure is quite similar. This
    /// method performs the common steps and returns the literals to be added to or removed from
    /// the assumptions.
    fn process_weight_split(
        &mut self,
        decomposed_core: &[(DecomposedPredicate, (i32, i32))],
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
        min_weight: i32,
        add_next_slice: bool,
    ) -> (Vec<Literal>, Vec<Literal>) {
        let mut to_remove = vec![];
        let mut to_add = vec![];
        for (dec_lit, w) in decomposed_core {
            // Decrease the (magnitude of the) weight by `min_weight`.
            // Note: for positive weights, this is w - w_min, for negative weights it is w + w_min.
            let mut new_w_res = w.0 - (w.0.signum() * min_weight);
            if new_w_res == 0 {
                // This variable is now fully absorbed into the reformulation
                // and can thus be removed from the assumptions.
                let DecomposedPredicate {
                    variable,
                    is_greater,
                    bound,
                } = *dec_lit;
                debug!("Variable {} now has 0 weight", variable);
                // Mark old assumption for removal.
                to_remove.push(self.get_literal(if is_greater {
                    predicate!(variable >= bound)
                } else {
                    predicate!(variable <= bound)
                }));

                if add_next_slice {
                    debug!("Adding next slice and resetting weight to {}", w.1);
                    // If we are instructed to add the next slice upon weight exhaustion,
                    // generate the corresponding literal and reset the weight.
                    to_add.push(self.get_literal(if is_greater {
                        predicate!(variable >= bound - 1)
                    } else {
                        predicate!(variable <= bound + 1)
                    }));
                    new_w_res = w.1;
                }
            }
            // Update the weight in the weight storage
            let _ = weights_per_var
                .entry(dec_lit.variable.id)
                .and_modify(|(w1, _)| *w1 = new_w_res);
        }
        (to_remove, to_add)
    }

    /// Takes a list of delayed assumptions, and - along with their corresponding weights -
    /// calculates the additional incurred cost.
    fn process_wce(
        &mut self,
        delayed_assumptions: &mut Vec<(DomainId, i32)>,
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
    ) -> (Vec<Literal>, i32) {
        // Note: add the assumption with the up-to-date lower bound
        debug!("Converting delayed assumptions");
        let res_assum = delayed_assumptions
            .iter()
            .map(|(did, _)| self.get_literal(predicate!(did <= self.lower_bound(did))))
            .collect();

        // All assumptions which have an altered bound incur an additional
        // cost. Calculate this cost.
        let res_cost = delayed_assumptions
            .iter()
            .map(|(did, bound)| {
                self.calculate_cost_of_increased_lb(
                    *did,
                    self.lower_bound(did) - bound,
                    weights_per_var,
                    false,
                )
            })
            .sum::<i32>();

        // All delayed assumptions have been added, and thus the list can be reset.
        debug!("Clearing delayed assumptions");
        delayed_assumptions.clear();
        (res_assum, res_cost)
    }

    /// Calculates the cost of increased cost for a given variable. The boolean
    /// `use_up_residual_weight` determines whether the residual weight should be considered and
    /// immediately reset for the given variable.
    fn calculate_cost_of_increased_lb(
        &self,
        did: DomainId,
        diff: i32,
        weights_per_var: &mut HashMap<u32, (i32, i32)>,
        use_up_residual_weight: bool,
    ) -> i32 {
        let mut cost = 0;

        // Multiply the bound increase with associated weight.
        let (res_weight, orig_weight) = *weights_per_var
            .get(&did.id)
            .expect("Variable must have weights");
        cost += diff * orig_weight;

        if use_up_residual_weight && diff > 0 {
            // NOTE: when using weight splitting, the first increase step uses the residual weight,
            // not the full weight. Correct the cost for this single step.
            cost += res_weight - orig_weight;
            // When "used up", reset the residual weight.

            debug!("Resetting residual weight for variable {}", did);
            let _ = weights_per_var
                .entry(did.id)
                .and_modify(|(w_res, w_orig)| *w_res = *w_orig);
        }
        cost
    }

    /// Takes as input a list of variables and their bounds, and returns two lists. The first is the
    /// list of literals corresponding to the assumptions `x <= l_x` for variables `x` with lower
    /// bounds `l_x`. The second is the list of differences between actual and provided lower
    /// bounds, used to calculate incurred cost.
    fn create_assumptions_with_lb_diff(
        &mut self,
        stratum: Vec<VarWithBound>,
    ) -> (Vec<Literal>, Vec<i32>) {
        stratum
            .iter()
            .map(|(var, lb)| {
                let new_lb = self.lower_bound(var);
                (self.get_literal(predicate!(var <= new_lb)), new_lb - lb)
            })
            .unzip()
    }

    /// Takes a literal and maps it to the corresponding  [`DecomposedPredicate`], allowing
    /// easy access to the elements making up the predicate.
    fn decompose_literal(&mut self, lit: Literal) -> Option<DecomposedPredicate> {
        self.get_integer_predicates_from_literal(lit)
            .find_map(|pred| match pred {
                IntegerPredicate::LowerBound {
                    domain_id: variable,
                    lower_bound: bound,
                } => Some(DecomposedPredicate {
                    variable,
                    is_greater: true,
                    bound,
                }),
                // remember variable, type, and used bound
                IntegerPredicate::UpperBound {
                    domain_id: variable,
                    upper_bound: bound,
                } => Some(DecomposedPredicate {
                    variable,
                    is_greater: false,
                    bound,
                }),
                _ => None,
            })
    }

    /// Takes a core (or any other vector of literals) and maps its literals to the
    /// [`DecomposedPredicate`]s corresponding to their assumptions.
    fn decompose_core(&mut self, core: &[Literal]) -> Vec<DecomposedPredicate> {
        core.iter()
            .filter_map(|c| {
                // get predicate from assumption
                self.decompose_literal(*c)
            })
            .collect::<Vec<DecomposedPredicate>>()
    }

    /// Takes the result of [`Solver::decompose_core`] and matches it with the weights, as provided
    /// by the third argument. This allows easy access to (nearly) all relevant data for the
    /// predicates present in a core.
    fn decompose_with_weights(
        &mut self,
        core: &[Literal],
        weights_per_var: &HashMap<u32, (i32, i32)>,
    ) -> Vec<(DecomposedPredicate, (i32, i32))> {
        let decomposed_core = self.decompose_core(core);
        assert_eq!(decomposed_core.len(), core.len());
        decomposed_core
            .into_iter()
            .map(|d_pred| {
                // match with weights
                (
                    *weights_per_var
                        .get(&d_pred.variable.id)
                        .expect("Weight needed for every domain id"),
                    d_pred,
                )
            })
            .map(|(a, b)| (b, a))
            .collect()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CoreGuidedArgs {
    // Reformulation approach
    pub variable_reformulation: bool,
    pub coefficient_elimination: bool,

    // Optional features
    pub weight_aware_cores: bool,
    pub stratification: bool,
    pub harden: bool,
}
