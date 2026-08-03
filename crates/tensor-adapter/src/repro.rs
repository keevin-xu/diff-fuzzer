//! Re-running a saved report to confirm it still holds.
//!
//! **A finding that cannot be replayed is not a finding.** It is a story about something
//! that happened once, and nobody can act on it — least of all a maintainer who has to
//! decide whether to spend an afternoon on it. So a report is only worth sending if it
//! can be loaded back and shown to still diverge.
//!
//! This is also the check that catches defects in *this tool*. If a saved report fails
//! to reproduce, the first suspect is not the library under test — it is a stray source
//! of randomness, a normalisation that is not deterministic, or a report that failed to
//! record something it depended on.

use crate::input::TensorOp;
use crate::normalize::CanonicalTensor;
use diff_fuzzer_core::{Agreement, ApproxEq, DivergenceReport, Runner, Tolerance};

/// What happened when a report was replayed.
#[derive(Debug, Clone, PartialEq)]
pub struct Reproduction {
    /// Whether the divergence occurred again.
    pub reproduced: bool,
    /// What each implementation produced *this time*, so a changed result can be
    /// compared against what the report recorded.
    pub outputs: Vec<(String, String)>,
    /// A human-readable account of the outcome.
    pub detail: String,
}

/// Re-run a report's case and report whether it still diverges.
///
/// **Compares under the tolerance the report recorded, not the current policy.** That
/// distinction matters. Replaying under the recorded threshold answers "does this claim
/// still hold as it was made?", which is the question a maintainer cares about. Replaying
/// under today's policy answers a different question — "would we still flag this?" — and
/// conflating them would let a policy change silently look like a fixed bug, or a fixed
/// bug look like a policy change.
pub fn reproduce(
    report: &DivergenceReport<TensorOp>,
    implementations: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
) -> Reproduction {
    let mut results: Vec<(String, CanonicalTensor)> = Vec::new();
    let mut failures = Vec::new();

    for implementation in implementations {
        match implementation.run_and_normalize(&report.input) {
            Ok(output) => results.push((implementation.name().to_string(), output)),
            Err(error) => failures.push(format!("{}: {error}", implementation.name())),
        }
    }

    let outputs: Vec<(String, String)> = results
        .iter()
        .map(|(name, output)| (name.clone(), format!("{output:?}")))
        .collect();

    if results.len() < 2 {
        return Reproduction {
            reproduced: false,
            outputs,
            detail: format!(
                "could not re-run on at least two implementations: {}",
                failures.join("; ")
            ),
        };
    }

    let tolerance = report.tolerance;
    let (reference_name, reference) = &results[0];

    for (name, candidate) in &results[1..] {
        match reference.approx_compare(candidate, tolerance) {
            Agreement::Agree(_) => {}
            Agreement::Structural { reason } => {
                return Reproduction {
                    reproduced: true,
                    outputs,
                    detail: format!("{reference_name} vs {name}: {reason}"),
                };
            }
            Agreement::Disagree(comparison) => {
                return Reproduction {
                    reproduced: true,
                    outputs,
                    detail: format!(
                        "{reference_name} vs {name}: {} of {} elements differ, \
                         max relative error {:.3e}",
                        comparison.mismatches, comparison.total, comparison.max_relative_error
                    ),
                };
            }
        }
    }

    Reproduction {
        reproduced: false,
        outputs,
        detail: describe_disappearance(tolerance),
    }
}

/// What it means when a recorded divergence no longer occurs.
///
/// Deliberately does *not* say "fixed". Four things could have changed — the library, our
/// tool, the environment, or the report's own fidelity — and only one of them is good
/// news. Asserting the cheerful interpretation would be exactly the kind of unearned
/// conclusion this project tries to avoid.
fn describe_disappearance(tolerance: Tolerance) -> String {
    format!(
        "no longer diverges under the recorded tolerance (rtol {:e}, atol {:e}). \
         Possible causes, in order of likelihood worth checking: the report omitted \
         something it depended on; the environment differs (library versions, platform); \
         this tool changed; or the underlying behaviour genuinely changed.",
        tolerance.rtol, tolerance.atol
    )
}
