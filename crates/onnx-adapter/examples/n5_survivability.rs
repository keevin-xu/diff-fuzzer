//! **N5.3, N5.4** — what the crash guard and the timeout cost, and whether hostile values abort.
//!
//! Three questions, each of which the project's own rules say must be measured rather than
//! assumed:
//!
//! 1. **What does arming the crash sentinel cost?** It runs on every execution, in the inner
//!    loop. A guard whose cost nobody measured is a guard that gets removed later by someone
//!    who assumes it is expensive.
//! 2. **What does the timeout wrapper cost?** It spawns a thread per execution. Against ONNX
//!    Runtime's 0.18 ms mean, a 30 µs spawn is not obviously negligible.
//! 3. **Do hostile values abort ONNX Runtime?** `PENDING` 1.4 closed at N2 with *zero aborts*
//!    and an explicit caveat: every case carried ordinary values, so it supported "not yet
//!    needed" rather than "never needed", and asked for a re-examination here.
//!
//! If ONNX Runtime aborts during this run, **the process dies and this example does not print a
//! summary** — that is what an abort means. The sentinel file is then the evidence, and finding
//! it is the point.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n5_survivability
use std::time::Instant;

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::sentinel::CrashSentinel;
use onnx_adapter::timeout::{DEFAULT_TIMEOUT, WithTimeout};

const CASES: u64 = 2000;

fn percentiles(mut samples: Vec<f64>) -> (f64, f64, f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let at = |q: f64| samples[((samples.len() as f64 * q) as usize).min(samples.len() - 1)];
    (mean, at(0.5), at(0.99), *samples.last().unwrap())
}

fn main() {
    let generator = OnnxGenerator::new(Bounds::default().with_special_values());
    let cases: Vec<_> = (0..CASES)
        .map(|seed| (seed, generator.generate(&mut SeededRng::from_seed(seed))))
        .filter(|(_, case)| onnx_adapter::validation::is_valid(case))
        .collect();

    let path = format!("{}/in-flight.json", onnx_adapter::FINDINGS_ROOT);
    let (mut sentinel, recovered) = CrashSentinel::open(&path).expect("opening the sentinel");

    // Before anything else: did the *previous* run leave a case in flight?
    match &recovered {
        Some(in_flight) => {
            println!(
                "\n*** RECOVERED an in-flight case from a previous run ***\n\
                 runtime {}, seed {}, operator {}\n\
                 That run did not return from this case. It is a finding.\n",
                in_flight.runtime,
                in_flight.seed,
                in_flight.case.op.onnx_name()
            );
        }
        None => println!("\nno case left in flight by a previous run"),
    }

    println!("\nwarming up");
    for (_, case) in cases.iter().take(50) {
        let _ = OrtRuntime.run(case);
        let _ = TractRuntime.run(case);
    }

    // ── 1. ONNX Runtime, unguarded ──────────────────────────────────────────────────
    let mut bare = Vec::new();
    for (_, case) in &cases {
        let started = Instant::now();
        let _ = OrtRuntime.run(case);
        bare.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    // ── 2. ONNX Runtime, with the sentinel armed around every execution ─────────────
    let mut guarded = Vec::new();
    let mut aborts_survived = 0usize;
    for (seed, case) in &cases {
        let started = Instant::now();
        sentinel
            .arm(OrtRuntime.name(), *seed, case)
            .expect("arming must not fail");
        let outcome = OrtRuntime.run(case);
        sentinel.disarm().expect("disarming must not fail");
        guarded.push(started.elapsed().as_secs_f64() * 1000.0);
        if matches!(outcome, Ok(OnnxOutcome::Crashed { .. })) {
            aborts_survived += 1;
        }
    }

    // ── 3. ONNX Runtime, wrapped in the timeout ─────────────────────────────────────
    let bounded = WithTimeout::with_bound(OrtRuntime, DEFAULT_TIMEOUT);
    let mut timed = Vec::new();
    let mut timeouts = 0usize;
    for (_, case) in &cases {
        let started = Instant::now();
        let outcome = bounded.run(case);
        timed.push(started.elapsed().as_secs_f64() * 1000.0);
        if matches!(outcome, Ok(OnnxOutcome::TimedOut { .. })) {
            timeouts += 1;
        }
    }

    // ── 4. The slowest execution actually observed, for the bound ───────────────────
    let mut tract_times = Vec::new();
    for (_, case) in &cases {
        let started = Instant::now();
        let _ = TractRuntime.run(case);
        tract_times.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "\n══ cost, {} valid cases, hostile values ══\n",
        cases.len()
    );
    println!(
        "{:<34} {:>9} {:>9} {:>9} {:>9}",
        "configuration", "mean ms", "median", "p99", "max"
    );
    for (label, samples) in [
        ("onnxruntime, unguarded", &bare),
        ("onnxruntime + crash sentinel", &guarded),
        ("onnxruntime + 5s timeout", &timed),
        ("tract, unguarded", &tract_times),
    ] {
        let (mean, median, p99, max) = percentiles(samples.clone());
        println!("{label:<34} {mean:>9.4} {median:>9.4} {p99:>9.4} {max:>9.4}");
    }

    let (bare_mean, ..) = percentiles(bare.clone());
    let (guarded_mean, ..) = percentiles(guarded.clone());
    let (timed_mean, ..) = percentiles(timed.clone());
    println!(
        "\n  crash sentinel overhead   {:+.1}% of an unguarded execution",
        100.0 * (guarded_mean - bare_mean) / bare_mean
    );
    println!(
        "  timeout overhead          {:+.1}% of an unguarded execution",
        100.0 * (timed_mean - bare_mean) / bare_mean
    );

    let slowest = [&bare, &guarded, &timed, &tract_times]
        .iter()
        .flat_map(|s| s.iter().copied())
        .fold(0.0f64, f64::max);
    println!("\n══ the bound ══");
    println!("  slowest execution observed here   {slowest:.2} ms");
    println!(
        "  bound in force                    {} ms  ({:.0}x the slowest observed)",
        DEFAULT_TIMEOUT.as_millis(),
        DEFAULT_TIMEOUT.as_secs_f64() * 1000.0 / slowest.max(0.001)
    );
    println!("  cases that hit the bound          {timeouts}");

    println!("\n══ the re-examination PENDING 1.4 asked for ══");
    println!(
        "  ONNX Runtime executions with hostile values: {}",
        cases.len() * 3
    );
    println!("  process-fatal aborts:                        0 (this line printed, so it lived)");
    println!("  catchable crashes recorded:                  {aborts_survived}");
    println!("\n  the sentinel is at {}", sentinel.path().display());
}
