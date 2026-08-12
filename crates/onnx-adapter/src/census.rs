//! The capability census — which runtime supports which operator, at which element type.
//!
//! # Why this is measured rather than read
//!
//! `onnx-mlir` publishes a support table and ONNX Runtime documents its coverage. Those are
//! claims about *intent*. `08-RISKS.md` §3 is explicit that a runtime may claim an operator
//! and fail on it, or support one it does not document — so every cell in this matrix comes
//! from building a minimal valid model and attempting it.
//!
//! # What it is for
//!
//! Two things, and the second is the one that matters most:
//!
//! 1. **Sizing the domain.** `08-RISKS.md` §1 names a small operator intersection as the
//!    risk most likely to sink this domain, which is why PHASE-N2 is a genuine go/no-go.
//! 2. **Making the crash thesis possible at all.** You cannot tell *"does not implement this
//!    operator"* from *"implements it and crashed"* without knowing what each runtime
//!    claims to support. Until this matrix exists, every returned error has to be recorded
//!    conservatively as `Rejected`, because guessing would manufacture findings.
//!
//! # The probe must be valid, or the measurement is meaningless
//!
//! A probe that fails because *our* model is malformed would be recorded as the runtime not
//! supporting the operator. Two gates stand in front of that: our own `validate`, and
//! `onnx.checker` reached through the reference implementation. Both are asserted over the
//! whole candidate set in `ops.rs`, and the census refuses to run if the reference declines
//! a probe.

use std::collections::BTreeMap;

use diff_fuzzer_core::traits::Implementation;
use serde::{Deserialize, Serialize};

use crate::case::{ElemType, OpKind};
use crate::ops::{self, Tier};
use crate::outcome::OnnxOutcome;

/// What one runtime did with one probe.
///
/// Deliberately **not** collapsed to a boolean. "Supported" and "unsupported" would lose the
/// distinction the whole domain rests on: a runtime that *crashes* on a valid minimal model
/// is not the same as one that declines the operator, and flattening them here would discard
/// the finding before it was ever looked at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Support {
    /// Produced a result. The operator is implemented for this element type.
    Supported,
    /// Declared it does not implement this. Legitimate, and never a bug.
    Unsupported,
    /// Returned a typed error on a model the specification's own checker accepted.
    ///
    /// Ambiguous by nature, which is why it is its own cell rather than being folded into
    /// either neighbour: it may be a polite refusal, or it may be a defect. At N5 the
    /// capability model turns some of these into crashes.
    Rejected,
    /// Panicked on a valid minimal model with ordinary values. **A finding.**
    Crashed,
}

impl Support {
    fn from(outcome: &OnnxOutcome) -> Self {
        match outcome {
            OnnxOutcome::Ok(_) => Support::Supported,
            OnnxOutcome::Unsupported { .. } => Support::Unsupported,
            OnnxOutcome::Rejected { .. } => Support::Rejected,
            OnnxOutcome::Crashed { .. } | OnnxOutcome::TimedOut { .. } => Support::Crashed,
        }
    }

    pub fn symbol(self) -> char {
        match self {
            Support::Supported => '+',
            Support::Unsupported => '-',
            Support::Rejected => '?',
            Support::Crashed => '!',
        }
    }
}

/// One measured cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    pub op: String,
    pub elem_type: ElemType,
    pub runtime: String,
    pub support: Support,
    /// The runtime's own words, when it declined. Kept because *why* a runtime refused is
    /// what turns a cell into an investigation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// The whole matrix, plus what produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Census {
    /// When it was taken. A capability matrix is a claim about *these* versions on *this*
    /// day; a stale one is worse than none, because it looks current.
    pub taken: String,
    pub opset: i64,
    pub environment: Vec<(String, String)>,
    pub runtimes: Vec<String>,
    pub cells: Vec<Cell>,
}

impl Census {
    /// Which runtimes produced a result for this operator and element type.
    pub fn supporting(&self, op: OpKind, elem: ElemType) -> Vec<&str> {
        self.cells
            .iter()
            .filter(|c| {
                c.op == op.onnx_name() && c.elem_type == elem && c.support == Support::Supported
            })
            .map(|c| c.runtime.as_str())
            .collect()
    }

    /// Operators supported by at least `n` participants at **at least one** element type.
    ///
    /// "At least one" rather than "all": an operator two runtimes agree on for `f32` is a
    /// testable operator, even if one of them declines it for `i32`. The census records the
    /// per-type detail; this is the coarser question the go/no-go bar asks.
    pub fn operators_supported_by(&self, n: usize) -> Vec<OpKind> {
        OpKind::ALL
            .into_iter()
            .filter(|op| {
                ElemType::ALL
                    .into_iter()
                    .any(|elem| self.supporting(*op, elem).len() >= n)
            })
            .collect()
    }

    /// Every cell where a runtime crashed. **Findings, not statistics.**
    pub fn crashes(&self) -> Vec<&Cell> {
        self.cells
            .iter()
            .filter(|c| c.support == Support::Crashed)
            .collect()
    }

    /// Count of each outcome, per runtime.
    pub fn tally(&self) -> BTreeMap<&str, BTreeMap<Support, usize>> {
        let mut counts: BTreeMap<&str, BTreeMap<Support, usize>> = BTreeMap::new();
        for cell in &self.cells {
            *counts
                .entry(cell.runtime.as_str())
                .or_default()
                .entry(cell.support)
                .or_default() += 1;
        }
        counts
    }
}

/// The go/no-go verdict, measured against the bar agreed **before** the census ran.
///
/// Agreed at G-N0 and recorded in `crates/onnx-adapter/DECISIONS.md`, precisely so it could
/// not be rationalized after the numbers were seen — `08-RISKS.md` §1's stated risk for this
/// phase.
#[derive(Debug, Clone)]
pub struct GoNoGo {
    pub operators_3plus: usize,
    pub tier_a: usize,
    pub value_dependent: usize,
}

/// The agreed minimum: ≥20 operators across ≥3 participants, ≥10 Tier A, ≥8 value-dependent.
pub const MIN_OPERATORS: usize = 20;
pub const MIN_TIER_A: usize = 10;
pub const MIN_VALUE_DEPENDENT: usize = 8;
/// Participants required to count an operator toward the bar.
pub const MIN_PARTICIPANTS: usize = 3;

impl GoNoGo {
    pub fn measure(census: &Census) -> Self {
        let qualifying = census.operators_supported_by(MIN_PARTICIPANTS);
        Self {
            operators_3plus: qualifying.len(),
            tier_a: qualifying
                .iter()
                .filter(|op| ops::spec(**op).tier == Tier::A)
                .count(),
            // The clause added during review. Without it the bar is clearable by operators
            // whose output does not depend on the values at all — `Shape`, `Size`,
            // `Identity`, `Reshape` — none of which can exercise this domain's thesis.
            value_dependent: qualifying
                .iter()
                .filter(|op| ops::spec(**op).value_dependent)
                .count(),
        }
    }

    pub fn passes(&self) -> bool {
        self.operators_3plus >= MIN_OPERATORS
            && self.tier_a >= MIN_TIER_A
            && self.value_dependent >= MIN_VALUE_DEPENDENT
    }

    /// The smallest margin by which any clause clears its bar, as a fraction.
    ///
    /// Reported because a bar met at 21-of-20 is not a mandate. A long autonomous run treats
    /// anything under 20% as a stop rather than a pass.
    pub fn tightest_margin(&self) -> f64 {
        [
            (self.operators_3plus, MIN_OPERATORS),
            (self.tier_a, MIN_TIER_A),
            (self.value_dependent, MIN_VALUE_DEPENDENT),
        ]
        .into_iter()
        .map(|(got, need)| (got as f64 - need as f64) / need as f64)
        .fold(f64::INFINITY, f64::min)
    }
}

/// Run the census across the given participants.
///
/// Takes the runtimes as a slice so the caller decides who takes part — a census run with a
/// participant missing would silently understate the intersection, and which participants
/// were present is recorded in the result.
pub fn take(
    runtimes: &[&dyn Implementation<In = crate::case::OnnxCase, Out = OnnxOutcome>],
    opset: i64,
) -> Census {
    let mut cells = Vec::new();

    for (op, elem) in ops::candidates(opset) {
        let case = ops::probe(op, elem, opset).expect("candidates only yields buildable pairs");
        for runtime in runtimes {
            let outcome = runtime.run(&case).expect("failures are values, never Err");
            let detail = match &outcome {
                OnnxOutcome::Ok(_) => None,
                OnnxOutcome::Rejected { detail } | OnnxOutcome::Crashed { detail } => {
                    Some(first_line(detail))
                }
                OnnxOutcome::Unsupported { reason } => Some(first_line(reason)),
                OnnxOutcome::TimedOut { after_ms } => Some(format!("{after_ms}ms")),
            };
            cells.push(Cell {
                op: op.onnx_name().to_string(),
                elem_type: elem,
                runtime: runtime.name().to_string(),
                support: Support::from(&outcome),
                detail,
            });
        }
    }

    Census {
        taken: env_date(),
        opset,
        environment: crate::environment::environment().components,
        runtimes: runtimes.iter().map(|r| r.name().to_string()).collect(),
        cells,
    }
}

fn first_line(text: &str) -> String {
    text.lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .chars()
        .take(160)
        .collect()
}

/// The date, for the census record.
///
/// Read from the environment rather than invented: a matrix that misreports when it was
/// taken is worse than one with no date, because it looks current.
fn env_date() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%MZ"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtimes::{OrtRuntime, TractRuntime};

    const OPSET: i64 = 22;

    /// The bar must be the one agreed in advance. A test on the constants themselves,
    /// because the whole point of pre-committing a number is that it cannot drift once the
    /// data is in view.
    #[test]
    fn the_go_no_go_bar_is_the_one_agreed_before_the_census() {
        assert_eq!(MIN_OPERATORS, 20);
        assert_eq!(MIN_TIER_A, 10);
        assert_eq!(MIN_VALUE_DEPENDENT, 8);
        assert_eq!(MIN_PARTICIPANTS, 3);
    }

    /// A census over two runtimes must produce one cell per candidate pair per runtime —
    /// no silent drops.
    #[test]
    fn the_census_covers_every_candidate() {
        let census = take(&[&OrtRuntime, &TractRuntime], OPSET);
        assert_eq!(census.cells.len(), ops::candidates(OPSET).len() * 2);
        assert_eq!(census.runtimes, vec!["onnxruntime", "tract"]);
        assert!(!census.taken.is_empty());
    }

    /// The verdict must be computable and must *fail* when it should. Two runtimes cannot
    /// meet a three-participant bar, so this is the negative control: it proves the bar can
    /// return false rather than always passing.
    #[test]
    fn the_bar_can_fail() {
        let census = take(&[&OrtRuntime, &TractRuntime], OPSET);
        let verdict = GoNoGo::measure(&census);
        assert_eq!(
            verdict.operators_3plus, 0,
            "two runtimes cannot support anything by three participants"
        );
        assert!(!verdict.passes(), "the bar must be able to return false");
    }

    #[test]
    fn support_symbols_are_distinct() {
        let symbols: Vec<char> = [
            Support::Supported,
            Support::Unsupported,
            Support::Rejected,
            Support::Crashed,
        ]
        .into_iter()
        .map(Support::symbol)
        .collect();
        let mut unique = symbols.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(symbols.len(), unique.len());
    }

    /// A census must serialize whole, because it is stored in the repo as data and consumed
    /// by the capability model at N5.
    #[test]
    fn a_census_round_trips_through_json() {
        let census = take(&[&OrtRuntime], OPSET);
        let restored: Census =
            serde_json::from_str(&serde_json::to_string(&census).unwrap()).unwrap();
        assert_eq!(restored.cells.len(), census.cells.len());
        assert_eq!(restored.opset, OPSET);
    }
}
