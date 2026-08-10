//! S7.2 + S7.4: collect negatives, search for a trigger rule, emit the candidate report.
//!
//! # Why this could not run before
//!
//! The predicate machinery was copied from `tensor-adapter` at S7.3 and has never been given a
//! real finding, because until S10 this project had none it could express. It is the last
//! dormant component, and exercising the others has twice found defects unit tests could not.
//!
//! # The constraint that shapes this run
//!
//! `Pool::matched` refuses to score findings against negatives drawn from a **different
//! distribution**, and it is right to: a rule separating two pools would score perfectly while
//! describing which generator ran rather than what triggers a bug. The tensor domain lost real
//! time to exactly that.
//!
//! So the hand-built comma-join case cannot be the finding here — it is `Constructed`, belonging
//! to no distribution. **Findings and negatives must both come from one generator run**, which
//! is what this does: generate under `V1_COMMA_JOINS`, run both engines, and let the divergence
//! decide which pile each case lands in.
use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::negatives::{Negative, Pool, Provenance, SamplingContext, Source, is_interesting};
use sql_adapter::search::search;

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);

    let generator = SqlGenerator::new(Bounds::V1_COMMA_JOINS);
    let description = generator.description();
    let context = SamplingContext::new(description.clone(), &["duckdb", "sqlite"]);

    println!("generator: {description}");
    println!("cases:     {total}\n");

    let mut findings: Vec<SqlCase> = Vec::new();
    let mut negatives: Vec<Negative> = Vec::new();

    for seed in 0..total as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let (Ok(left), Ok(right)) = (SqliteImpl.run(&case), DuckDbImpl.run(&case)) else {
            continue;
        };

        if left != right {
            findings.push(case);
        } else {
            // **Classified, not lumped.** `Source` orders negatives by how much they
            // discriminate: a case carrying a `NULL` or an empty table is a far harder
            // counter-example than an ordinary one, and summing them would hide that.
            let source = if is_interesting(&case) {
                Source::Interesting
            } else {
                Source::Ordinary
            };
            negatives.push(Negative {
                case,
                source,
                provenance: Provenance::SeededDefault,
                generator: description.clone(),
                backends: context.backends.clone(),
            });
        }
    }

    println!("findings:  {}", findings.len());
    println!("negatives: {} before sampling", negatives.len());

    // Keep the ordinary negatives to a modest number: they are cheap and weak, and a pool
    // dominated by them would let a rule look good by clearing a low bar many times over.
    let mut ordinary_kept = 0usize;
    negatives.retain(|n| {
        if n.source == Source::Ordinary {
            ordinary_kept += 1;
            ordinary_kept <= 400
        } else {
            true
        }
    });

    let pool = match Pool::matched(negatives, &context) {
        Ok(pool) => pool,
        Err(error) => {
            println!("\npool rejected: {error:?}");
            println!("  This is the guard working: findings and negatives must come from one");
            println!("  distribution, or a rule describes the generator rather than the bug.");
            return;
        }
    };

    println!("negatives: {} in the matched pool", pool.negatives().len());
    // `matched_by_source` takes the predicate to score; passing one that matches everything
    // gives the pool's composition, which is what a reader needs before any rule is quoted.
    for (source, _, total) in pool.matched_by_source(|_| true) {
        println!("    {:<12} {total}", source.label());
    }

    if findings.is_empty() {
        println!("\nNo findings in this run — nothing to explain.");
        return;
    }

    println!("\n== search ==");
    let result = search(&findings, &pool);
    println!("  predicates considered: {}", result.considered);
    println!("  rules committed to:    {}", result.classes.len());
    println!("  unexplained findings:  {}", result.unexplained.len());

    for (rank, candidate) in result.classes.iter().enumerate() {
        println!("\n  #{} {}", rank + 1, candidate.predicate.describe());
        println!(
            "     covers {} of {} findings",
            candidate.covered.len(),
            findings.len()
        );
        // **Never summed.** A rule firing on three near-misses and one firing on three
        // ordinary cases are different claims, and one number cannot say which.
        for (source, matched, total) in &candidate.negatives_by_source {
            println!("     negatives {:<12} {matched}/{total}", source.label());
        }
    }

    // **Why no rule, when one was committed?** The covering loop reports only rules it accepts;
    // when it accepts none, the useful question is which predicate came *closest* and what it
    // failed on. Without this the output says "vocabulary gap" and leaves the reader to guess
    // which part of the vocabulary.
    if result.classes.is_empty() {
        use sql_adapter::features::extract;
        use sql_adapter::search::enumerate;

        let positives: Vec<_> = findings.iter().map(extract).collect();
        let negative_vecs: Vec<_> = pool.negatives().iter().map(|n| extract(&n.case)).collect();

        let mut best: Option<(usize, usize, String)> = None;
        for predicate in enumerate() {
            let covers = positives.iter().filter(|v| predicate.matches(**v)).count();
            if covers == 0 {
                continue;
            }
            let leaks = negative_vecs
                .iter()
                .filter(|v| predicate.matches(**v))
                .count();
            // Rank by coverage first, then by how few negatives it lets through.
            let score = (covers, usize::MAX - leaks);
            if best
                .as_ref()
                .is_none_or(|(c, l, _)| (covers, usize::MAX - leaks) > (*c, *l))
            {
                best = Some((
                    score.0,
                    score.1,
                    format!(
                        "{} — covers {covers}/{} findings, matches {leaks}/{} negatives",
                        predicate.describe(),
                        findings.len(),
                        negative_vecs.len()
                    ),
                ));
            }
        }
        if let Some((_, _, description)) = best {
            println!("\n  closest predicate: {description}");
        }
    }

    if !result.unexplained.is_empty() {
        println!(
            "\n  **{} findings no rule explains** — a vocabulary gap, and the most useful\n  \
             output here: it names a property the feature list cannot express.",
            result.unexplained.len()
        );
    }
}
