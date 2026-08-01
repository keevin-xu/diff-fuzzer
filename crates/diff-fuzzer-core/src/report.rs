//! What we hand back when implementations disagree.
//!
//! Note what is *not* here: the seed. A [`Divergence`] describes what disagreed, which
//! is all an oracle can know — an oracle is handed some results and judges them; it
//! has no idea which run produced them or how. The seed, the versions of the systems
//! involved, and the tolerance in force are properties of the *run*, and get attached
//! by the driver that owns those facts.
//!
//! The full artifact — a saved, replayable file combining a divergence with all of
//! that run context — is built later, once there is something real to put in it.

/// A record of implementations disagreeing on one input.
///
/// The fields are deliberately plain strings at this stage. Making this generic over
/// the output type, or serialisable, would be guessing at requirements that the
/// minimisation and reporting work has not produced yet — and a guess baked into the
/// core is harder to remove than to add.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// The offending input, as its `Debug` representation.
    pub input: String,
    /// One entry per implementation: its name, and what it produced.
    pub outputs: Vec<(String, String)>,
    /// A human-readable statement of what disagreed.
    pub summary: String,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "divergence: {}", self.summary)?;
        writeln!(f, "  input: {}", self.input)?;
        for (name, output) in &self.outputs {
            writeln!(f, "  {name}: {output}")?;
        }
        Ok(())
    }
}
