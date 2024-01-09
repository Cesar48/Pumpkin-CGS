use crate::basic_types::{
    ConflictInfo, ConstraintReference, Literal, PropositionalVariable,
    PropositionalVariableGeneratorIterator, Trail,
};
use crate::engine::LocalId;
use crate::{pumpkin_assert_moderate, pumpkin_assert_simple};

#[derive(Clone)]
pub struct AssignmentsPropositional {
    assignment_info: Vec<PropositionalAssignmentInfo>,
    trail: Trail<Literal>,
    pub true_literal: Literal,
    pub false_literal: Literal,
}

impl Default for AssignmentsPropositional {
    fn default() -> Self {
        let dummy_literal = Literal::new(PropositionalVariable::new(0), true);
        AssignmentsPropositional {
            assignment_info: Default::default(),
            trail: Default::default(),
            true_literal: dummy_literal,
            false_literal: !dummy_literal,
        }
    }
}

impl AssignmentsPropositional {
    pub fn increase_decision_level(&mut self) {
        self.trail.increase_decision_level()
    }

    pub fn get_decision_level(&self) -> usize {
        self.trail.get_decision_level()
    }

    pub fn num_trail_entries(&self) -> usize {
        self.trail.len()
    }

    pub fn get_trail_entry(&self, index: usize) -> Literal {
        self.trail[index]
    }

    pub fn grow(&mut self) {
        self.assignment_info
            .push(PropositionalAssignmentInfo::Unassigned);
    }

    pub fn num_propositional_variables(&self) -> u32 {
        self.assignment_info.len() as u32
    }

    pub fn get_propositional_variables(&self) -> PropositionalVariableGeneratorIterator {
        //we start from 1 to ignore the special variable with index zero, which is always assigned at the root to true
        PropositionalVariableGeneratorIterator::new(1, self.num_propositional_variables())
    }

    pub fn is_variable_assigned_true(&self, variable: PropositionalVariable) -> bool {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Assigned {
                truth_value,
                decision_level: _,
                constraint_reference: _,
                local_id: _,
            } => truth_value,
            PropositionalAssignmentInfo::Unassigned => false,
        }
    }

    pub fn is_variable_assigned_false(&self, variable: PropositionalVariable) -> bool {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Assigned {
                truth_value,
                decision_level: _,
                constraint_reference: _,
                local_id: _,
            } => !truth_value,
            PropositionalAssignmentInfo::Unassigned => false,
        }
    }

    pub fn get_local_id_of_literal(&self, literal: Literal) -> Option<LocalId> {
        match self.assignment_info[literal.get_propositional_variable()] {
            PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level: _,
                constraint_reference: _,
                local_id,
            } => local_id,
            PropositionalAssignmentInfo::Unassigned => None,
        }
    }

    pub fn is_literal_assigned_true(&self, literal: Literal) -> bool {
        if literal.is_positive() {
            self.is_variable_assigned_true(literal.get_propositional_variable())
        } else {
            self.is_variable_assigned_false(literal.get_propositional_variable())
        }
    }

    pub fn is_literal_assigned_false(&self, literal: Literal) -> bool {
        self.is_literal_assigned(literal) && !self.is_literal_assigned_true(literal)
    }

    pub fn is_literal_assigned(&self, literal: Literal) -> bool {
        self.is_variable_assigned(literal.get_propositional_variable())
    }

    pub fn is_literal_unassigned(&self, literal: Literal) -> bool {
        self.is_variable_unassigned(literal.get_propositional_variable())
    }

    pub fn is_variable_unassigned(&self, variable: PropositionalVariable) -> bool {
        self.assignment_info[variable] == PropositionalAssignmentInfo::Unassigned
    }

    pub fn is_variable_assigned(&self, variable: PropositionalVariable) -> bool {
        self.assignment_info[variable] != PropositionalAssignmentInfo::Unassigned
    }

    pub fn is_literal_root_assignment(&self, literal: Literal) -> bool {
        if self.is_literal_unassigned(literal) {
            false
        } else {
            self.get_variable_assignment_level(literal.get_propositional_variable()) == 0
        }
    }

    pub fn is_variable_decision(&self, variable: PropositionalVariable) -> bool {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Unassigned => false,
            PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level: _,
                constraint_reference,
                local_id: _,
            } => constraint_reference.is_null(),
        }
    }

    pub fn is_literal_decision(&self, literal: Literal) -> bool {
        self.is_variable_decision(literal.get_propositional_variable())
    }

    pub fn is_variable_propagated(&self, variable: PropositionalVariable) -> bool {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Unassigned => false,
            PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level: _,
                constraint_reference,
                local_id: _,
            } => !constraint_reference.is_null(),
        }
    }

    pub fn is_literal_propagated(&self, literal: Literal) -> bool {
        self.is_variable_propagated(literal.get_propositional_variable())
    }

    pub fn get_variable_assignment_level(&self, variable: PropositionalVariable) -> usize {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Unassigned => {
                panic!("Unassigned variables do not have assignment levels");
            }
            PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level,
                constraint_reference: _,
                local_id: _,
            } => decision_level,
        }
    }

    pub fn get_literal_assignment_level(&self, literal: Literal) -> usize {
        self.get_variable_assignment_level(literal.get_propositional_variable())
    }

    pub fn get_variable_reason_constraint(
        &self,
        variable: PropositionalVariable,
    ) -> ConstraintReference {
        match self.assignment_info[variable] {
            PropositionalAssignmentInfo::Unassigned => {
                panic!("Unassigned variables do not have reason codes levels");
            }
            PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level: _,
                constraint_reference,
                local_id: _,
            } => constraint_reference,
        }
    }

    pub fn get_literal_reason_constraint(&self, literal: Literal) -> ConstraintReference {
        self.get_variable_reason_constraint(literal.get_propositional_variable())
    }

    fn make_assignment(
        &mut self,
        true_literal: Literal,
        constraint_reference: ConstraintReference,
        local_id: Option<LocalId>,
    ) -> Option<ConflictInfo> {
        if self.is_literal_assigned_false(true_literal) {
            return Some(ConflictInfo::Propagation {
                literal: true_literal,
                reference: constraint_reference,
            });
        } else if self.is_literal_assigned_true(true_literal) {
            // This can happen if a CP propagation triggers this, when the corresponding literal
            // was already assigned.
            return None;
        }

        self.assignment_info[true_literal.get_propositional_variable()] =
            PropositionalAssignmentInfo::Assigned {
                truth_value: true_literal.is_positive(),
                decision_level: self.get_decision_level(),
                constraint_reference,
                local_id,
            };

        self.trail.push(true_literal);

        None
    }

    pub fn undo_assignment(&mut self, variable: PropositionalVariable) {
        pumpkin_assert_moderate!(self.is_variable_assigned(variable));

        self.assignment_info[variable] = PropositionalAssignmentInfo::Unassigned;
    }

    pub fn enqueue_decision_literal(&mut self, decision_literal: Literal) {
        pumpkin_assert_simple!(!self.is_literal_assigned(decision_literal));

        self.make_assignment(
            decision_literal,
            ConstraintReference::create_null_reference(),
            None,
        );
    }

    pub fn enqueue_propagated_literal(
        &mut self,
        propagated_literal: Literal,
        constraint_reference: ConstraintReference,
        local_id: Option<LocalId>,
    ) -> Option<ConflictInfo> {
        pumpkin_assert_simple!(!constraint_reference.is_null());
        self.make_assignment(propagated_literal, constraint_reference, local_id)
    }

    /// This iterator returns the literals on the trail in _reverse_ order (LIFO)
    pub fn synchronise(&mut self, new_decision_level: usize) -> impl Iterator<Item = Literal> + '_ {
        self.trail.synchronise(new_decision_level).inspect(|entry| {
            let variable = entry.get_propositional_variable();

            self.assignment_info[variable] = PropositionalAssignmentInfo::Unassigned;
        })
    }

    pub fn is_at_the_root_level(&self) -> bool {
        self.get_decision_level() == 0
    }

    pub fn debug_create_empty_clone(&self) -> Self {
        AssignmentsPropositional {
            assignment_info: vec![Default::default(); self.assignment_info.len()],
            trail: Default::default(),
            true_literal: self.true_literal,
            false_literal: self.false_literal,
        }
    }
}

#[derive(PartialEq, Clone, Default)]
enum PropositionalAssignmentInfo {
    Assigned {
        truth_value: bool,
        decision_level: usize,

        // TODO: There is a dependency between these two fields that we ideally
        // want to encode at a type level. A `constraint_reference` points
        // to a propagator if and only if `local_id` is `Some(_)`.
        //
        // Since a reference to a propagator will in most cases be accompanied
        // by a local id, it would make sense to encode the local id within the
        // reference. However, that impacts the layout of the type possibly
        // hurting cache performance. This is pure speculation, and should be
        // verified with benchmarks.
        constraint_reference: ConstraintReference,
        local_id: Option<LocalId>,
    },
    #[default]
    Unassigned,
}

#[cfg(test)]
mod tests {
    use crate::{
        basic_types::{ConstraintReference, Literal, PropositionalVariable},
        engine::{sat::assignments_propositional::PropositionalAssignmentInfo, PropagatorId},
    };

    use super::AssignmentsPropositional;

    #[test]
    pub fn already_assigned_literal_does_not_override_assignment_info() {
        let mut assignments_propositional = AssignmentsPropositional::default();
        let literal = Literal::new(
            PropositionalVariable::new(assignments_propositional.num_propositional_variables()),
            true,
        );
        assignments_propositional.grow();

        let result = assignments_propositional.make_assignment(
            literal,
            ConstraintReference::create_propagator_reference(PropagatorId(0)),
            None,
        );
        assert!(result.is_none());
        assert_eq!(assignments_propositional.trail.len(), 1);
        //Re-assigning a literal which is already true does not result in the info being overwritten
        let result_reassignment = assignments_propositional.make_assignment(
            literal,
            ConstraintReference::create_propagator_reference(PropagatorId(1)),
            None,
        );
        assert!(result_reassignment.is_none());
        //Nor does it result in anything being added to the trail
        assert_eq!(assignments_propositional.trail.len(), 1);
        assert!({
            if let PropositionalAssignmentInfo::Assigned {
                truth_value: _,
                decision_level: _,
                constraint_reference,
                local_id: _,
            } = assignments_propositional.assignment_info[literal.get_propositional_variable()]
            {
                constraint_reference.get_propagator_id() == PropagatorId(0)
            } else {
                false
            }
        })
    }
}
