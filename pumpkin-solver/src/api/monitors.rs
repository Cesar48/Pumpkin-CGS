use std::time::Instant;

use log::warn;

use crate::basic_types::HashMap;

pub(crate) struct MonitorGroup {
    pub(crate) lb_stat: LowerBoundEvolutionMonitor,
    pub(crate) core_stat: CoreSizeMonitor,
    pub(crate) task_stat: TimePerTaskMonitor,
    pub(crate) wce_stat: WCECoreAmountMonitor,
    pub(crate) hard_stat: HardeningDomainLimitationMonitor,
    pub(crate) exh_stat: CoreExhaustionMonitor,
}

impl MonitorGroup {
    pub(crate) fn new() -> Self {
        MonitorGroup {
            lb_stat: LowerBoundEvolutionMonitor::new(),
            core_stat: CoreSizeMonitor::new(),
            task_stat: TimePerTaskMonitor::new(),
            wce_stat: WCECoreAmountMonitor::new(),
            hard_stat: HardeningDomainLimitationMonitor::new(),
            exh_stat: CoreExhaustionMonitor::new(),
        }
    }

    pub(crate) fn get_results(
        self,
    ) -> (
        Vec<(i32, u128)>,
        Vec<usize>,
        HashMap<MonitoredTasks, u128>,
        Vec<usize>,
        Vec<f32>,
        Vec<f32>,
    ) {
        (
            self.lb_stat.get_result(),
            self.core_stat.get_result(),
            self.task_stat.get_result(),
            self.wce_stat.get_result(),
            self.hard_stat.get_result(),
            self.exh_stat.get_result(),
        )
    }
}

/// A struct for measuring the evolution of lower bounds, by measuring both the lower bound and the
/// time between two updated lower bounds.
pub(crate) struct LowerBoundEvolutionMonitor {
    start: Instant,
    lower_bounds: Vec<i32>,
    timestamps: Vec<u128>,
    // Note: u128 is the default elapsed time unit. However, when using microseconds, this will
    // only have interesting effects when the timeout is order of magnitude 2^128 ~= 3e38
    // micros, or 9e28 hours (~= 1e25 years). As such, it can safely be converted to (slightly)
    // smaller types.
}

impl LowerBoundEvolutionMonitor {
    /// Initialises a new [`LowerBoundEvolutionMonitor`].
    pub(crate) fn new() -> Self {
        LowerBoundEvolutionMonitor {
            start: Instant::now(),
            lower_bounds: vec![],
            timestamps: vec![],
        }
    }

    /// Called when a tighter lower bound is reported; saves the corresponding lower bound,
    /// as well as the time it took to find it - measured from initialisation.
    pub(crate) fn update(&mut self, lb: i32) {
        self.timestamps.push(self.start.elapsed().as_micros());
        self.lower_bounds.push(lb);
    }

    /// Combines the relevant data from this object into a single vector
    /// to remove the need for the object itself.
    /// Note: consumes self.
    pub(crate) fn get_result(self) -> Vec<(i32, u128)> {
        let LowerBoundEvolutionMonitor {
            lower_bounds,
            timestamps,
            ..
        } = self;
        lower_bounds.into_iter().zip(timestamps).collect()
    }
}

/// A struct for keeping track of the sizes of cores found during the minimisation process.
pub(crate) struct CoreSizeMonitor {
    cores: Vec<usize>,
}

impl CoreSizeMonitor {
    /// Initialises a new [`CoreSizeMonitor`].
    pub(crate) fn new() -> Self {
        CoreSizeMonitor { cores: vec![] }
    }

    /// Called when a new core is found; saves its size.
    pub(crate) fn core_found(&mut self, size: usize) {
        self.cores.push(size);
    }

    /// Returns the list of core sizes, in order of discovery.
    /// Note: consumes self.
    pub(crate) fn get_result(self) -> Vec<usize> {
        let CoreSizeMonitor { cores } = self;
        cores
    }
}

/// Defines the tasks which are monitored by the [`TimePerTaskMonitor`] defined below.
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub(crate) enum MonitoredTasks {
    Overhead,
    Solving,
    CoreProcessing,
    WCEAdditions,
    StrataCreation,
    PartitionMerging,
}

/// A struct for keeping track of the time spent on every notable task.
pub(crate) struct TimePerTaskMonitor {
    tasks_with_times: HashMap<MonitoredTasks, u128>,
    // Note: as with the lower bound monitor, u128 is the default elapsed time unit,
    // but can be easily converted to smaller types if needed.
    active_tasks: Vec<MonitoredTasks>,
    start_of_task: Instant,
}

impl TimePerTaskMonitor {
    /// Initialises a new [`TimePerTaskMonitor`].
    pub(crate) fn new() -> Self {
        TimePerTaskMonitor {
            tasks_with_times: HashMap::default(),
            active_tasks: vec![],
            start_of_task: Instant::now(),
        }
    }

    /// Called when a given task is started. This function (temporarily) closes any potential
    /// active tasks and starts measuring the time from this function call.
    pub(crate) fn start_task(&mut self, task: MonitoredTasks) {
        if !self.active_tasks.is_empty() {
            self.save_time(self.active_tasks[self.active_tasks.len() - 1].clone());
        }
        self.active_tasks.push(task);
        self.start_of_task = Instant::now();
    }

    /// Called when a given task ends. Calculates the elapsed time (in microseconds) since the
    /// task was started, and records this. If any paused, active tasks remain, it will re-open
    /// the most recently paused one.
    pub(crate) fn end_task(&mut self, task: MonitoredTasks) {
        let old_task = self.active_tasks.pop();
        if old_task.is_none() || old_task != Some(task.clone()) {
            warn!(
                "Incorrect active task: {:?} instead of {:?}. Results may be incorrect.",
                old_task.clone(),
                task.clone(),
            );

            let idx_to_remove = self.active_tasks.iter().rposition(|x| x == &task);
            if let Some(idx) = idx_to_remove {
                let old_task = self.active_tasks.remove(idx);
                let old_time = self.tasks_with_times.entry(old_task).or_insert(0_u128);
                *old_time += self.start_of_task.elapsed().as_micros();
                self.start_of_task = Instant::now();
            } else {
                warn!("Incorrect task was not in stack and could not be stopped");
            }
            if let Some(t) = old_task {
                // Note: if the old task was not used, put it back
                self.active_tasks.push(t);
            }
        } else {
            self.save_time(task);
            self.start_of_task = Instant::now();
        }
    }

    /// Helper function to record the (current) running time of an active task.
    fn save_time(&mut self, current: MonitoredTasks) {
        let old_time = self.tasks_with_times.entry(current).or_insert(0_u128);
        *old_time += self.start_of_task.elapsed().as_micros();
    }

    /// Returns a [`HashMap`] of [`MonitoredTasks`]s and their total running time from the
    /// minimisation process.
    /// Note: consumes self.
    pub(crate) fn get_result(self) -> HashMap<MonitoredTasks, u128> {
        let TimePerTaskMonitor {
            tasks_with_times, ..
        } = self;
        tasks_with_times
    }
}

/// A struct for keeping track of the amount of distinct cores discovered between two
/// reformulations.
pub(crate) struct WCECoreAmountMonitor {
    num_cores: Vec<usize>,
}

impl WCECoreAmountMonitor {
    /// Initialises a new [`WCECoreAmountMonitor`].
    pub(crate) fn new() -> Self {
        WCECoreAmountMonitor { num_cores: vec![] }
    }

    /// Called when a reformulation is initiated. Receives the number of cores found since the
    /// last reformulation, and records this number.
    pub(crate) fn record_reformulation(&mut self, num_cores: usize) {
        self.num_cores.push(num_cores);
    }

    /// Returns the list containing the number of cores between two reformulation, for every
    /// interval between two reformulations.
    /// Note: consumes self.
    pub(crate) fn get_result(self) -> Vec<usize> {
        let WCECoreAmountMonitor { num_cores } = self;
        num_cores
    }
}

/// A struct for keeping track of the efficiency of hardening. This is done by measuring the
/// cartesian product of objective variables, both before and after hardening, and dividing the
/// two numbers to find the fraction of remaining domain sizes.
pub(crate) struct HardeningDomainLimitationMonitor {
    domain_fractions: Vec<f32>,
}

impl HardeningDomainLimitationMonitor {
    /// Initialises a new [`HardeningDomainLimitationMonitor`].
    pub(crate) fn new() -> Self {
        HardeningDomainLimitationMonitor {
            domain_fractions: vec![],
        }
    }

    /// Called after a hardening step is completed. Receives the quotient of the old and new
    /// cartesian products of the domains of objective variables, divided by one another.
    pub(crate) fn hardened(&mut self, fraction: f32) {
        self.domain_fractions.push(fraction)
    }

    /// Returns a [`Vec`] of the fractions of domains remaining after each hardening step.
    /// Note: consumes self.
    pub(crate) fn get_result(self) -> Vec<f32> {
        let HardeningDomainLimitationMonitor { domain_fractions } = self;
        domain_fractions
    }
}

/// A struct for keeping track of the relative lower bound increase achieved by core exhaustion.
/// This metric is calculated by dividing the exhausted lower bound by the non-exhausted lower
/// bound. Note that some very basic inference, which might increase the bound upon creation of the
/// variable, is NOT counted in this metric, and thus actual core exhaustion needs to be performed
/// for this metric to record different numbers from 1.
pub(crate) struct CoreExhaustionMonitor {
    bound_ratio: Vec<f32>,
}

impl CoreExhaustionMonitor {
    /// Initialises a new [`CoreExhaustionMonitor`]
    pub(crate) fn new() -> Self {
        CoreExhaustionMonitor {
            bound_ratio: vec![],
        }
    }

    /// Called after exhaustion. Receives the quotient of the new bound and the old bound on the
    /// newly introduced reformulation variable. This value is at least 1.
    pub(crate) fn record_exhaustion(&mut self, quotient: f32) {
        self.bound_ratio.push(quotient);
    }

    /// Returns a [`Vec`] of the recorded values corresponding to each exhaustion.
    /// Note: consumes self
    pub(crate) fn get_result(self) -> Vec<f32> {
        let CoreExhaustionMonitor { bound_ratio } = self;
        bound_ratio
    }
}
