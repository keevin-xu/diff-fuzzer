//! Does the oracle still catch a broken engine on the configuration the campaign will run?
//!
//! # Why a campaign cannot be trusted without this
//!
//! A long quiet run has two explanations — the engines agree, or **the detector is broken** —
//! and from the outside they are the same silence. Only one is good news. Injecting a known
//! fault and confirming the oracle reports it is what separates them, and it has to be redone
//! whenever the *configuration* changes, because a detector can be perfectly correct and still
//! be unreachable on a corpus that never produces the shape it inspects.
//!
//! # The distinction this runner exists to make
//!
//! **"The fault changed nothing" is not "the oracle missed it", and conflating them corrupts
//! the number in both directions.**
//!
//! `DropLastRow` on an empty result removes nothing, so both engines return the same thing and
//! the oracle *correctly* reports agreement. On the campaign corpus that is not a rare edge
//! case: **44.4% of results are empty** (S10.9). Scoring those as misses would invent a
//! detection failure; scoring them as passes would count cases where the oracle was never
//! actually asked anything.
//!
//! So each case is classified three ways, and only the first is a real test:
//!
//! - **exercised** — the fault changed the output, so the oracle *must* diverge.
//! - **inert** — the fault changed nothing, so agreement is correct. Excluded from the rate.
//! - **unrunnable** — the case did not execute; counted, but nothing to say about it.
//!
//! A miss is a **hard failure** and exits non-zero: it means the campaign about to run cannot
//! be believed.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example detection_check -- [cases] [axis]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Normalizer, Oracle, Verdict,
};
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::DuckDbImpl;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::normalize::SqlNormalizer;
use sql_adapter::oracle::SqlDifferentialOracle;
use sql_adapter::outcome::SqlOutcome;
use sql_adapter::testing::{Fault, FaultyEngine};

/// What happened to one case under one fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The fault changed the output and the oracle caught it. The only outcome that passes.
    Caught,
    /// The fault changed the output and the oracle did **not** catch it. A detection failure.
    Missed,
    /// The fault left the output identical — nothing was asked of the oracle.
    ///
    /// There is deliberately no `Unrunnable` variant: a case that does not execute is filtered
    /// out before `judge` is reached, so representing it here would be a state this function
    /// can never return.
    Inert,
}

fn judge(case: &SqlCase, honest: &SqlOutcome, faulty: &SqlOutcome) -> Outcome {
    // **The gate that makes the number mean something**, applied to the *normalized* results
    // rather than the raw ones.
    //
    // Comparing raw output would be the obvious choice and would be wrong: the oracle judges
    // canonical results, so a fault that normalization legitimately erases — a reordering on a
    // query that promises no order — is invisible to the oracle *correctly*. Calling that a
    // miss would manufacture a detection failure out of the normalizer doing its job.
    //
    // Without any gate at all, an empty-result corpus reports a flawless detection rate while
    // asking the oracle almost nothing: 44.4% of results here are empty (S10.9), and
    // `DropLastRow` removes nothing from an empty grid.
    let honest = SqlNormalizer.normalize(honest.clone());
    let faulty = SqlNormalizer.normalize(faulty.clone());
    if honest == faulty {
        return Outcome::Inert;
    }

    let named = vec![
        // **Named `duckdb`, not `sqlite`.** Both sides here *are* DuckDB — one honest, one
        // faulted — and labelling the honest side `sqlite` would describe a comparison that
        // never happened. `testing.rs` makes the point about the faulty side and it applies
        // equally to this one: a result that names the wrong engine is a fabricated result,
        // even in a check nobody files as a finding.
        NamedOutput {
            implementation: "duckdb".to_string(),
            output: honest.clone(),
        },
        NamedOutput {
            implementation: "duckdb-faulty".to_string(),
            output: faulty.clone(),
        },
    ];

    match SqlDifferentialOracle.check(case, &named) {
        Verdict::Diverged(_) => Outcome::Caught,
        // A skip is a miss for this purpose, and deliberately so: the campaign learns nothing
        // from a case the oracle declines to judge, and a normalizer that skipped everything
        // would otherwise score as perfect.
        Verdict::Agree | Verdict::Skipped(_) => Outcome::Missed,
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let total: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000);
    let axis = arguments.next().unwrap_or_else(|| "all-large".to_string());

    let bounds = match axis.as_str() {
        "all-large" => Bounds::V1_ALL_LARGE,
        "all" => Bounds::V1_ALL,
        "large" => Bounds::V1_LARGE,
        "window" => Bounds::V1_WINDOW,
        other => {
            eprintln!("unknown axis {other:?}. valid: all-large, all, large, window");
            std::process::exit(2);
        }
    };

    let generator = SqlGenerator::new(bounds);
    println!("generator: {}", generator.description());
    println!("cases:     {total} per fault\n");
    println!(
        "  {:<18} {:>9} {:>8} {:>8} {:>12} {:>10}",
        "fault", "exercised", "caught", "missed", "inert", "detection"
    );

    let mut any_missed = false;
    for fault in [
        Fault::DropLastRow,
        Fault::ChangeFirstCell,
        Fault::AlwaysRefuse,
    ] {
        // The fault is applied to **DuckDB**, and one side is enough here only because
        // `testing.rs` already proves the oracle is symmetric; this runner is asking whether
        // the *corpus* reaches the detector, not whether the detector is biased.
        let broken = FaultyEngine::new(DuckDbImpl, fault, "duckdb-faulty");

        let (mut caught, mut missed, mut inert, mut unrunnable) = (0, 0, 0, 0);
        for seed in 0..total as u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));

            let (Ok(honest), Ok(faulty)) = (DuckDbImpl.run(&case), broken.run(&case)) else {
                unrunnable += 1;
                continue;
            };

            match judge(&case, &honest, &faulty) {
                Outcome::Caught => caught += 1,
                Outcome::Missed => {
                    missed += 1;
                    if missed <= 3 {
                        println!("  MISS at seed {seed} under {fault:?}");
                        for statement in case.statements(sql_adapter::render::Dialect::Sqlite) {
                            println!("    {statement}");
                        }
                        println!("    honest: {honest:?}");
                        println!("    faulty: {faulty:?}");
                    }
                }
                Outcome::Inert => inert += 1,
            }
        }

        let exercised = caught + missed;
        let rate = if exercised == 0 {
            // Printed rather than silently shown as 100%: a fault that is never exercised
            // tells you nothing about the oracle, and it is the corpus that needs fixing.
            "NEVER EXERCISED".to_string()
        } else {
            format!("{:.1}%", 100.0 * caught as f64 / exercised as f64)
        };

        println!(
            "  {:<18} {exercised:>9} {caught:>8} {missed:>8} {inert:>12} {rate:>10}",
            format!("{fault:?}"),
        );
        let _ = unrunnable;
        any_missed |= missed > 0;
    }

    if any_missed {
        println!(
            "\nDETECTION FAILURE — the oracle missed a fault it was shown. A quiet campaign on \
             this configuration would be meaningless."
        );
        std::process::exit(1);
    }

    println!(
        "\nEvery exercised fault was caught. This is what licenses reading a quiet campaign as \
         evidence about the engines rather than about the detector."
    );
}
