//! Shrinking a failing case to the smallest one that still fails.
//!
//! A generated divergence arrives at whatever size the generator happened to produce —
//! possibly a rank-4 tensor of several thousand values, most of which have nothing to do
//! with the failure. Nobody can act on that. A maintainer receiving it has to first
//! work out which part matters, which is work we are better placed to do automatically.
//!
//! The technique is **delta debugging**: given a failing input and a predicate that says
//! whether a candidate still fails, repeatedly try simpler candidates and keep any that
//! preserve the failure. Repeat until nothing simpler works — a local minimum.
//!
//! This module holds the [`Shrink`] capability, which asks a domain "what simpler
//! versions of this are there?". The search that uses it lives alongside, and the moves
//! themselves are necessarily domain knowledge: only the tensor adapter knows that
//! halving a matrix multiplication's inner dimension means changing *both* operands.

/// A value that can propose simpler versions of itself.
///
/// Two obligations, and both matter for the search to terminate and to be trustworthy.
///
/// **Every candidate must be valid.** A shrunk case still has to be something the
/// systems under test will accept — halving one operand of an elementwise operation
/// without halving the other produces a case that cannot run, which wastes a step and
/// teaches nothing. Constraints that held for the generated case must hold for every
/// candidate.
///
/// **Every candidate must be strictly simpler.** If a candidate could be as complex as
/// its parent, the search could cycle forever. "Simpler" here means fewer elements, or
/// values closer to zero — never more of either.
pub trait Shrink: Sized {
    /// Simpler versions of this value, **most aggressive first**.
    ///
    /// Order matters for speed rather than correctness. A greedy search takes the first
    /// candidate that still fails, so offering the biggest reduction first means fewer
    /// rounds to reach the same place: halving a dimension gets there faster than
    /// removing one element at a time.
    fn candidates(&self) -> Vec<Self>;
}
