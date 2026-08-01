//! What we hand back when implementations disagree.
//!
//! Minimal for now — enough to see *that* something diverged and reproduce it. The
//! full artifact (serialised to disk, carrying the shrunk input and every backend
//! version) comes later, once there is something real to put in it.

/// A record of implementations disagreeing on one input.
///
/// The fields are deliberately plain strings at this stage. Making this generic over
/// the output type, or serialisable, would be guessing at requirements that the
/// minimisation and reporting work has not produced yet — and a guess baked into the
/// core is harder to remove than to add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DivergenceReport {
    /// The seed that produced the input. Given this, anyone can replay the run.
    pub seed: u64,
    /// The offending input, as its `Debug` representation.
    pub input: String,
    /// One entry per implementation: its name, and what it produced.
    pub outputs: Vec<(String, String)>,
    /// A human-readable statement of what disagreed.
    pub summary: String,
}

impl std::fmt::Display for DivergenceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "divergence (seed {}): {}", self.seed, self.summary)?;
        writeln!(f, "  input: {}", self.input)?;
        for (name, output) in &self.outputs {
            writeln!(f, "  {name}: {output}")?;
        }
        Ok(())
    }
}
