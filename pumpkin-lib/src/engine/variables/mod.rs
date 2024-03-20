//! A variable, in the context of the solver, is a view onto a domain. It may forward domain
//! information unaltered, or apply transformations which can be performed without the need of
//! constraints.

mod affine_view;
mod domain_generator_iterator;
mod domain_id;
mod integer_variable;
mod literal;
mod propositional_variable;
mod propositional_variable_generator_iterator;

pub use affine_view::AffineView;
pub use domain_generator_iterator::DomainGeneratorIterator;
pub use domain_id::DomainId;
pub use integer_variable::IntegerVariable;
pub use literal::Literal;
pub use propositional_variable::PropositionalVariable;
pub use propositional_variable_generator_iterator::PropositionalVariableGeneratorIterator;