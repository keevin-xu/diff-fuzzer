//! Strategies for deciding whether results are wrong.
//!
//! Each strategy is a type implementing [`Oracle`](crate::traits::Oracle), so the
//! driver treats them interchangeably and gaining a new one is an addition rather than
//! a rewrite.
//!
//! [`differential`] is the one built here: compare several implementations of the same
//! thing and flag disagreement. Its blind spot is worth stating plainly — if every
//! implementation is wrong in the same way, they all agree and nothing is reported.
//! Covering that needs a different strategy, one that checks a single implementation's
//! results against a relationship that must hold regardless of what the correct answer
//! is. This module is arranged so that arrives as a sibling of `differential`.

pub mod differential;

pub use differential::DifferentialOracle;
