mod file_format;
mod flatzinc;
mod maxsat;
mod parsers;
mod result;

use std::fmt::Debug;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use clap::ValueEnum;
use convert_case::Case;
use file_format::FileFormat;
use log::error;
use log::info;
use log::warn;
use log::Level;
use log::LevelFilter;
use parsers::dimacs::parse_cnf;
use parsers::dimacs::SolverArgs;
use parsers::dimacs::SolverDimacsSink;
use pumpkin_solver::encodings::PseudoBooleanEncoding;
use pumpkin_solver::options::*;
use pumpkin_solver::proof::Format;
use pumpkin_solver::proof::ProofLog;
use pumpkin_solver::results::ProblemSolution;
use pumpkin_solver::results::SatisfactionResult;
use pumpkin_solver::results::Solution;
use pumpkin_solver::statistics::configure_statistic_logging;
use pumpkin_solver::termination::TimeBudget;
use pumpkin_solver::variables::PropositionalVariable;
use pumpkin_solver::Solver;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use result::PumpkinError;
use result::PumpkinResult;

use crate::flatzinc::FlatZincOptions;
use crate::maxsat::wcnf_problem;

#[derive(Debug, Parser)]
#[command(
    help_template = "\
{before-help}{name} {version}
Authors: {author}
About: {about}

{usage-heading}\n{tab}{usage}

{all-args}{after-help}
",
    author,
    version,
    about,
    arg_required_else_help = true
)]
struct Args {
    /// The instance to solve. The file should have one of the following extensions:
    ///  - '*.cnf' for SAT instances, given in the DIMACS format,
    ///  - '*.wcnf' for MaxSAT instances, given in the WDIMACS format.
    ///  - '*.fzn' for FlatZinc instances (see <https://docs.minizinc.dev/en/stable/flattening.html>).
    #[clap(verbatim_doc_comment)]
    instance_path: PathBuf,

    /// The output path for the proof file.
    ///
    /// When solving a DIMACS instance, a DRAT proof is logged. In case of a FlatZinc model, a DRCP
    /// proof is logged.
    #[arg(long, verbatim_doc_comment)]
    proof_path: Option<PathBuf>,

    /// What type of proof to log.
    ///
    /// If the `proof_path` option is not provided, this is ignored.
    #[arg(long, default_value_t = ProofType::Scaffold)]
    proof_type: ProofType,

    /// The number of high lbd learned clauses that are kept in the database.
    /// Learned clauses are kept based on the tiered system introduced in "Improving
    /// SAT Solvers by Exploiting Empirical Characteristics of CDCL - Chanseok Oh (2016)".
    ///
    /// Possible values: u64
    #[arg(
        long = "learning-max-num-clauses",
        default_value_t = 4000,
        verbatim_doc_comment
    )]
    learning_max_num_clauses: u64,

    /// Learned clauses with this threshold LBD or lower are kept permanently
    /// Learned clauses are kept based on the tiered system introduced "Improving
    /// SAT Solvers by Exploiting Empirical Characteristics of CDCL - Chanseok Oh (2016)".
    ///
    /// Possible values: u32
    #[arg(
        long = "learning-lbd-threshold",
        default_value_t = 5,
        verbatim_doc_comment
    )]
    learning_lbd_threshold: u32,

    /// Decides which clauses will be removed when cleaning up the learned clauses. Can either be
    /// based on the LBD of a clause (the number of different decision levels) or on the activity
    /// of a clause (how often it is used in conflict analysis).
    #[arg(
        short = 'l',
        long = "learning-sorting-strategy",
        default_value_t = LearnedClauseSortingStrategy::Activity, verbatim_doc_comment
    )]
    learning_sorting_strategy: LearnedClauseSortingStrategy,

    /// Decides whether learned clauses are minimised as a post-processing step after computing the
    /// 1-UIP Minimisation is done; according to the idea proposed in "Generalized Conflict-Clause
    /// Strengthening for Satisfiability Solvers - Allen van Gelder (2011)".
    ///
    /// If this flag is present then the minimisation is turned off.
    ///
    /// Possible values: bool
    #[arg(long = "no-learning-minimise", verbatim_doc_comment)]
    no_learning_clause_minimisation: bool,

    /// Decides the sequence based on which the restarts are performed.
    /// - The "constant" approach uses a constant number of conflicts before another restart is
    ///   triggered
    /// - The "geometric" approach uses a geometrically increasing sequence
    /// - The "luby" approach uses a recursive sequence of the form 1, 1, 2, 1, 1, 2, 4, 1, 1, 2,
    ///   1, 1, 2, 4, 8, 1, 1, 2.... (see "Optimal speedup of Las Vegas algorithms - Luby et al.
    ///   (1993)")
    ///
    /// To be used in combination with "--restarts-base-interval".
    #[arg(
        long = "restart-sequence",
        default_value_t = SequenceGeneratorType::Constant, verbatim_doc_comment
    )]
    restart_sequence_generator_type: SequenceGeneratorType,

    /// The base interval length is used as a multiplier to the restart sequence.
    /// - In the case of the "constant" restart sequence this argument indicates the constant which
    ///   is used to determine when a restart occurs
    /// - For the "geometric" approach this argument indicates the starting value of the sequence
    /// - For the "luby" approach, the sequence is multiplied by this value
    ///
    /// For example, constant restarts with base interval 50 means a restart is triggered every 50
    /// conflicts.
    ///
    /// Possible values: u64
    #[arg(
        long = "restart-base-interval",
        default_value_t = 50,
        verbatim_doc_comment
    )]
    restart_base_interval: u64,

    /// Indicates the minimum number of initial conflicts before the first restart can occur. This
    /// allows the solver to learn some things about the problem before a restart is allowed to
    /// occur.
    ///
    /// Possible values: u64
    #[arg(
        long = "restart-min-initial-conflicts",
        default_value_t = 10000,
        verbatim_doc_comment
    )]
    restart_min_num_conflicts_before_first_restart: u64,

    /// Used to determine if a restart should be forced (see "Refining Restarts Strategies for SAT
    /// and UNSAT - Audemard and Simon (2012)").
    ///
    /// The state is "bad" if the current LBD value is much greater than
    /// the global LBD average. A greater (lower) value for lbd-coef means a less (more) frequent
    /// restart policy. If the long-term average LBD multiplied by this coefficient is lower
    /// than the short-term average LBD then a restart is performed.
    ///
    /// Possible values: f64
    #[arg(
        long = "restart-lbd-coef",
        default_value_t = 1.25,
        verbatim_doc_comment
    )]
    restart_lbd_coef: f64,

    /// Used to determine if a restart should be blocked (see "Refining Restarts Strategies for SAT
    /// and UNSAT - Audemard and Simon (2012)").
    ///
    /// To be used in combination with "--restarts-num-assigned-window".
    ///
    /// A restart is blocked if the number of assigned propositional variables is much greater than
    /// the average number of assigned variables in the recent past. A greater (lower) value for
    /// "--restart-num-assigned-coef" means fewer (more) blocked restarts.
    ///
    /// Possible values: f64
    #[arg(
        long = "restart-num-assigned-coef",
        default_value_t = 1.4,
        verbatim_doc_comment
    )]
    restart_num_assigned_coef: f64,

    /// Used to determine the length of the recent past that should be considered when deciding on
    /// blocking restarts (see "Refining Restarts Strategies for SAT
    /// and UNSAT - Audemard and Simon (2012)").
    ///
    /// The solver considers the last "--restart_num_assigned_window" conflicts as the reference
    /// point for the number of assigned variables.
    ///
    /// Possible values: u64
    #[arg(
        long = "restart-num-assigned-window",
        default_value_t = 5000,
        verbatim_doc_comment
    )]
    restart_num_assigned_window: u64,

    /// The coefficient in the geometric sequence `x_i = x_{i-1} * "--restart-geometric-coef"`
    /// where `x_1 = "--restarts-base-interval"`. Used only if "--restarts-sequence-generator"
    /// is assigned to "geometric".
    ///
    /// Possible values: f64 (Optional)
    #[arg(long = "restart-geometric-coef", verbatim_doc_comment)]
    restart_geometric_coef: Option<f64>,

    /// The time budget for the solver, given in milliseconds.
    ///
    /// Possible values: u64 (Optional)
    #[arg(short = 't', long = "time-limit", verbatim_doc_comment)]
    time_limit: Option<u64>,

    /// The random seed to use for the Pseudo Random Number Generator.
    ///
    /// Randomisation can be used for aspects such as the variable/value generator or initial
    /// ordering of variables.
    ///
    /// Possible values: u64
    #[arg(
        short = 'r',
        long = "random-seed",
        default_value_t = 42,
        verbatim_doc_comment
    )]
    random_seed: u64,

    /// Enables log message output from the solver.
    ///
    /// For printing statistics see the option "--log-statistics", and for printing all solutions
    /// (in case of a satisfaction problem) or printing solutions of increasing quality (in case of
    /// an optimization problem) see the option "--all-solutions".
    ///
    /// Possible values: bool
    #[arg(short = 'v', long = "verbose", verbatim_doc_comment)]
    verbose: bool,

    /// Enables logging of statistics from the solver.
    ///
    /// Possible values: bool
    #[arg(short = 's', long = "log-statistics", verbatim_doc_comment)]
    log_statistics: bool,

    /// Instructs the solver to perform free search when solving a MiniZinc model; this flag
    /// indicates that it is allowed to ignore the search annotations specified in the model.
    ///
    /// See the MiniZinc specification (<https://docs.minizinc.dev/en/stable/fzn-spec.html#cmdoption-f>)
    /// for more information.
    ///
    /// Possible values: bool
    #[arg(short = 'f', long = "free-search", verbatim_doc_comment)]
    free_search: bool,

    /// Instructs the solver to report all solutions in the case of satisfaction problems,
    /// or print intermediate solutions of increasing quality in the case of optimisation
    /// problems.
    ///
    /// See the MiniZinc specification (<https://docs.minizinc.dev/en/stable/fzn-spec.html#cmdoption-a>)
    /// for more information.
    ///
    /// Possible values: bool
    #[arg(short = 'a', long = "all-solutions", verbatim_doc_comment)]
    all_solutions: bool,

    /// If `--verbose` is enabled then this option removes the timestamp information from the log
    /// messages. Note that this option will only take affect in the case of a (W)CNF instance.
    ///
    /// Possible values: bool
    #[arg(long = "omit-timestamp", verbatim_doc_comment)]
    omit_timestamp: bool,

    /// If `--verbose` is enabled then this option removes the call site information from the log
    /// messages. The call site is the file and line from which the message
    /// originated. Note that this option will only take affect in the case of a (W)CNF instance.
    ///
    /// Possible values: bool
    #[arg(long = "omit-call-site", default_value_t = false, verbatim_doc_comment)]
    omit_call_site: bool,

    /// The encoding to use for the upper bound constraint in a MaxSAT optimisation problem.
    ///
    /// The "gte" value specifies that the solver should use the Generalized Totalizer Encoding
    /// (see "Generalized totalizer encoding for pseudo-boolean constraints - Saurabh et al.
    /// (2015)"), and the "cne" value specifies that the solver should use the Cardinality Network
    /// Encoding (see "Cardinality networks: a theoretical and empirical study - Asín et al.
    /// (2011)").
    #[arg(
        long = "upper-bound-encoding",
        default_value_t = PseudoBooleanEncoding::GeneralizedTotalizer, verbatim_doc_comment
    )]
    upper_bound_encoding: PseudoBooleanEncoding,

    /// Determines that the cumulative propagator(s) are allowed to create holes in the domain.
    ///
    /// Possible values: bool
    #[arg(long = "cumulative-allow-holes", verbatim_doc_comment)]
    cumulative_allow_holes: bool,

    /// Determines that no restarts are allowed by the solver.
    ///
    /// Possible values: bool
    #[arg(long = "no-restarts", verbatim_doc_comment)]
    no_restarts: bool,
    /// Determines the type of explanation used by the cumulative propagator(s) to explain
    /// propagations/conflicts.
    #[arg(long = "cumulative-explanation-type", default_value_t = CumulativeExplanationType::default())]
    cumulative_explanation_type: CumulativeExplanationType,

    /// Determines the type of propagator which is used by the cumulative propagator(s) to
    /// propagate the constraint.
    ///
    /// Currently, the solver only supports variations on time-tabling methods.
    #[arg(long = "cumulative-propagation-method",  default_value_t = CumulativePropagationMethod::default())]
    cumulative_propagation_method: CumulativePropagationMethod,

    /// Determines whether a sequence of profiles is generated when explaining a propagation for
    /// the cumulative constraint.
    ///
    /// Possible values: bool
    #[arg(long = "cumulative-generate-sequence")]
    cumulative_generate_sequence: bool,

    /// Determines whether incremental backtracking is applied or whether the cumulative
    /// propagators compute the time-table from scratch upon backtracking
    ///
    /// Possible values: bool
    #[arg(long = "cumulative-incremental-backtracking")]
    cumulative_incremental_backtracking: bool,

    /// The following arguments are specifically core-guided

    /// Specify whether to use core-guided search, instead of linear search.
    #[arg(long = "core-guided", default_value_t = false)]
    core_guided: bool,

    /// When using core-guided search, specifies to use variable-based reformulations
    /// rather than slice-based reformulations
    #[arg(long = "reform-var", default_value_t = false)]
    variable_reformulation: bool,

    /// When using core-guided search, specifies to use coefficient elimination
    /// rather than weight splitting
    #[arg(long = "coef-elim", default_value_t = false)]
    coefficient_elimination: bool,

    /// In core-guided search, we can apply Disjoint Core Extraction.
    /// This causes several cores to be extracted in between (completed) reformulations.
    /// This argument determines whether WCE is applied.
    #[arg(long = "wce", default_value_t = false)]
    weight_aware_cores: bool,

    /// In core-guided search, we can apply stratification, to consider high-weighing strata
    /// (i.e. groups of objective variables) before low-weighing strata.
    /// This argument specifies whether strata are used.
    #[arg(long = "strat", default_value_t = false)]
    stratification: bool,

    /// In core-guided search, we can apply hardening if intermediate solutions are found.
    /// This argument specifies whether to do so.
    #[arg(long = "harden", default_value_t = false)]
    harden: bool,

    /// In core-guided search, we can apply partitioning, which divides the objective into several
    /// smaller problems which are iteratively solved individually, then merged, until the original
    /// problem is reached after merging the last two partitions.
    /// This argument specifies whether to do so.
    #[arg(long = "partition", default_value_t = false)]
    partition: bool,
}

fn configure_logging(
    file_format: FileFormat,
    verbose: bool,
    log_statistics: bool,
    omit_timestamp: bool,
    omit_call_site: bool,
) -> std::io::Result<()> {
    match file_format {
        FileFormat::CnfDimacsPLine | FileFormat::WcnfDimacsPLine => {
            configure_logging_sat(verbose, log_statistics, omit_timestamp, omit_call_site)
        }
        FileFormat::FlatZinc => configure_logging_minizinc(verbose, log_statistics),
    }
}

fn configure_logging_unknown() -> std::io::Result<()> {
    env_logger::Builder::new()
        .format(move |buf, record| writeln!(buf, "{}", record.args()))
        .filter_level(LevelFilter::Trace)
        .target(env_logger::Target::Stdout)
        .init();
    Ok(())
}

fn configure_logging_minizinc(verbose: bool, log_statistics: bool) -> std::io::Result<()> {
    if log_statistics {
        configure_statistic_logging("%%%mzn-stat:", Some("%%%mzn-stat-end"), Some(Case::Camel));
    }
    let level_filter = if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Warn
    };

    env_logger::Builder::new()
        .format(move |buf, record| {
            write!(buf, "% ")?;

            writeln!(buf, "{}", record.args())
        })
        .filter_level(level_filter)
        .target(env_logger::Target::Stdout)
        .init();
    info!("Logging successfully configured");
    Ok(())
}

fn configure_logging_sat(
    verbose: bool,
    log_statistics: bool,
    omit_timestamp: bool,
    omit_call_site: bool,
) -> std::io::Result<()> {
    if log_statistics {
        configure_statistic_logging("c STAT", None, None);
    }
    let level_filter = if verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Warn
    };

    env_logger::Builder::new()
        .format(move |buf, record| {
            write!(buf, "c ")?;
            if record.level() != Level::Info && !omit_timestamp {
                write!(buf, "{} ", buf.timestamp())?;
            }
            write!(buf, "{} ", record.level())?;
            if record.level() != Level::Info && !omit_call_site {
                write!(
                    buf,
                    "[{}:{}] ",
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0)
                )?;
            }
            writeln!(buf, "{}", record.args())
        })
        .filter_level(level_filter)
        .target(env_logger::Target::Stdout)
        .init();
    info!("Logging successfully configured");
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(e) => {
            error!("Execution failed, error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run() -> PumpkinResult<()> {
    let args = Args::parse();

    let file_format = match args.instance_path.extension().and_then(|ext| ext.to_str()) {
        Some("cnf") => FileFormat::CnfDimacsPLine,
        Some("wcnf") => FileFormat::WcnfDimacsPLine,
        Some("fzn") => FileFormat::FlatZinc,
        _ => {
            configure_logging_unknown()?;
            return Err(PumpkinError::invalid_instance(args.instance_path.display()));
        }
    };

    configure_logging(
        file_format,
        args.verbose,
        args.log_statistics,
        args.omit_timestamp,
        args.omit_call_site,
    )?;

    if pumpkin_solver::asserts::PUMPKIN_ASSERT_LEVEL_DEFINITION
        >= pumpkin_solver::asserts::PUMPKIN_ASSERT_MODERATE
    {
        warn!("Potential performance degradation: the Pumpkin assert level is set to {}, meaning many debug asserts are active which may result in performance degradation.", pumpkin_solver::asserts::PUMPKIN_ASSERT_LEVEL_DEFINITION);
    };

    let learning_options = LearningOptions {
        num_high_lbd_learned_clauses_max: args.learning_max_num_clauses,
        high_lbd_learned_clause_sorting_strategy: args.learning_sorting_strategy,
        lbd_threshold: args.learning_lbd_threshold,
        ..Default::default()
    };

    let proof_log = if let Some(path_buf) = args.proof_path {
        match file_format {
            FileFormat::CnfDimacsPLine => ProofLog::dimacs(&path_buf)?,
            FileFormat::WcnfDimacsPLine => {
                return Err(PumpkinError::ProofGenerationNotSupported("wcnf".to_owned()))
            }
            FileFormat::FlatZinc => {
                let log_inferences =
                    matches!(args.proof_type, ProofType::Full | ProofType::WithHints);
                let log_hints = matches!(args.proof_type, ProofType::WithHints);
                ProofLog::cp(&path_buf, Format::Text, log_inferences, log_hints)?
            }
        }
    } else {
        ProofLog::default()
    };

    let solver_options = SolverOptions {
        restart_options: RestartOptions {
            sequence_generator_type: args.restart_sequence_generator_type,
            base_interval: args.restart_base_interval,
            min_num_conflicts_before_first_restart: args
                .restart_min_num_conflicts_before_first_restart,
            lbd_coef: args.restart_lbd_coef,
            num_assigned_coef: args.restart_num_assigned_coef,
            num_assigned_window: args.restart_num_assigned_window,
            geometric_coef: args.restart_geometric_coef,
            no_restarts: args.no_restarts,
        },
        proof_log,
        learning_clause_minimisation: !args.no_learning_clause_minimisation,
        random_generator: SmallRng::seed_from_u64(args.random_seed),
    };

    let time_limit = args.time_limit.map(Duration::from_millis);
    let instance_path = args
        .instance_path
        .to_str()
        .ok_or(PumpkinError::invalid_instance(args.instance_path.display()))?;

    let cg_args = if args.core_guided {
        Some(CoreGuidedArgs {
            variable_reformulation: args.variable_reformulation,
            coefficient_elimination: args.coefficient_elimination,
            weight_aware_cores: args.weight_aware_cores,
            stratification: args.stratification,
            harden: args.harden,
            partition: args.partition,
        })
    } else {
        None
    };

    match file_format {
        FileFormat::CnfDimacsPLine => {
            cnf_problem(learning_options, solver_options, time_limit, instance_path)?
        }
        FileFormat::WcnfDimacsPLine => wcnf_problem(
            learning_options,
            solver_options,
            time_limit,
            instance_path,
            args.upper_bound_encoding,
        )?,
        FileFormat::FlatZinc => flatzinc::solve(
            Solver::with_options(learning_options, solver_options),
            instance_path,
            time_limit,
            FlatZincOptions {
                free_search: args.free_search,
                all_solutions: args.all_solutions,
                cumulative_options: CumulativeOptions::new(
                    args.cumulative_allow_holes,
                    args.cumulative_explanation_type,
                    args.cumulative_generate_sequence,
                    args.cumulative_propagation_method,
                    args.cumulative_incremental_backtracking,
                ),
            },
            cg_args,
        )?,
    }

    Ok(())
}

fn cnf_problem(
    learning_options: LearningOptions,
    solver_options: SolverOptions,
    time_limit: Option<Duration>,
    instance_path: impl AsRef<Path>,
) -> Result<(), PumpkinError> {
    let instance_file = File::open(instance_path)?;
    let mut solver = parse_cnf::<SolverDimacsSink>(
        instance_file,
        SolverArgs::new(learning_options, solver_options),
    )?;

    let mut termination =
        TimeBudget::starting_now(time_limit.unwrap_or(Duration::from_secs(u64::MAX)));
    let mut brancher = solver.default_brancher_over_all_propositional_variables();
    match solver.satisfy(&mut brancher, &mut termination) {
        SatisfactionResult::Satisfiable(solution) => {
            solver.log_statistics();
            println!("s SATISFIABLE");
            let num_propositional_variables = solution.num_propositional_variables();
            println!(
                "v {}",
                stringify_solution(&solution, num_propositional_variables, true)
            );
        }
        SatisfactionResult::Unsatisfiable => {
            solver.log_statistics();
            if solver.conclude_proof_unsat().is_err() {
                warn!("Failed to log solver conclusion");
            };

            println!("s UNSATISFIABLE");
        }
        SatisfactionResult::Unknown => {
            solver.log_statistics();
            println!("s UNKNOWN");
        }
    };

    Ok(())
}

fn stringify_solution(
    solution: &Solution,
    num_variables: usize,
    terminate_with_zero: bool,
) -> String {
    (1..num_variables)
        .map(|index| PropositionalVariable::new(index.try_into().unwrap()))
        .map(|var| {
            if solution.get_propositional_variable_value(var) {
                format!("{} ", var.get_index())
            } else {
                format!("-{} ", var.get_index())
            }
        })
        .chain(if terminate_with_zero {
            std::iter::once(String::from("0"))
        } else {
            std::iter::once(String::new())
        })
        .collect::<String>()
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProofType {
    /// Log only the proof scaffold.
    Scaffold,
    /// Log the full proof without hints.
    Full,
    /// Log the full proof with hints.
    WithHints,
}

impl Display for ProofType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofType::Scaffold => write!(f, "scaffold"),
            ProofType::Full => write!(f, "full"),
            ProofType::WithHints => write!(f, "with-hints"),
        }
    }
}
