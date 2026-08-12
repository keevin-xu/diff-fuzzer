//! Measure per-case cost for every participant on the same corpus.
//!
//! # What this decides
//!
//! `PENDING` 1.2: is `onnx.reference` a **confirmer** (consulted only when the in-process
//! runtimes disagree) or a **participant** (run on every case)?
//!
//! Three planning documents recommended *confirmer*, on the stated premise that Python is
//! "orders of magnitude slower" and must stay out of the inner loop. **Nobody had measured
//! it.** A probe at N0 put the reference at ~0.023 ms per case on small models, which does
//! not look like orders of magnitude slower than anything. But one side of a ratio is not a
//! ratio, so this program measures all four on the *same* corpus and prints the comparison.
//!
//! What is at stake: under the confirmer design the specification oracle never runs on an
//! agreeing corpus — which is nearly all of it — and the existence of that oracle is the
//! reason this domain was chosen over another.
//!
//! # The measurement trap this is written against
//!
//! `05-MEASUREMENT-AND-CAMPAIGNS.md`: *a throughput sample small enough to be convenient is
//! systematically optimistic, because it is fast precisely for the reason that it rarely
//! contains the expensive tail.* A convenient SQL sample measured 84 cases/sec against a
//! true 18, turning a "6-hour" campaign into 25 hours.
//!
//! So this reports the **distribution**, not just the mean: median, p99 and max, per
//! participant. A mean alone cannot show a tail.
//!
//! **Honest limit at N1:** the skeleton generator produces four simple operators at
//! shapes up to 4×4×4. There is very little tail *to* find. These numbers say what the
//! cheap cases cost and settle the order-of-magnitude question; they are **not** a campaign
//! throughput figure, and must be re-measured after N3 widens the generator.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n1_throughput [cases]

use std::time::Instant;

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};

use onnx_adapter::case::OnnxCase;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::model::build_bytes;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

/// Any participant, behind the seam. The engine sees no difference between a C++ library,
/// two pure-Rust crates, and a Python subprocess — which is the point of the trait.
type BoxedParticipant = Box<dyn Implementation<In = OnnxCase, Out = OnnxOutcome>>;

fn main() {
    let cases: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2_000);

    let generator = OnnxGenerator::default();
    let corpus: Vec<OnnxCase> = (0..cases)
        .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
        .collect();

    println!("Per-case cost, {cases} cases from the N1 skeleton generator");
    println!("{}", generator.describe());
    println!("═══════════════════════════════════════════════════════════════════════");

    // The model build is measured separately because `05-TECH-STACK.md` flags it as a
    // plausible dominant cost — protobuf serialization happens once per case and every
    // participant pays for it indirectly.
    let mut build_times = Vec::with_capacity(corpus.len());
    for case in &corpus {
        let start = Instant::now();
        let bytes = build_bytes(case);
        build_times.push(start.elapsed().as_secs_f64());
        std::hint::black_box(bytes);
    }
    report("model build (protobuf)", &build_times);

    let mut participants: Vec<(&str, BoxedParticipant)> = vec![
        ("onnxruntime", Box::new(OrtRuntime)),
        ("tract", Box::new(TractRuntime)),
    ];
    #[cfg(feature = "candle")]
    participants.push(("candle", Box::new(onnx_adapter::runtimes::CandleRuntime)));
    match ReferenceRuntime::start() {
        Ok(reference) => participants.push(("onnx.reference", Box::new(reference))),
        Err(why) => println!(
            "\n!! the reference is unavailable, so the comparison PENDING 1.2 \
                              needs cannot be made: {why}"
        ),
    }

    let mut summaries = Vec::new();
    for (name, participant) in &participants {
        // Warm up. The reference pays ~55 ms once to register ~192 operator classes, and
        // letting that land inside the timed loop is precisely the error that produced a
        // wrong number at N0 — a figure 12x cheaper than a step it contained.
        for case in corpus.iter().take(20) {
            std::hint::black_box(participant.run(case).ok());
        }

        let mut times = Vec::with_capacity(corpus.len());
        for case in &corpus {
            let start = Instant::now();
            let outcome = participant.run(case);
            times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(outcome.ok());
        }
        let mean = report(name, &times);
        summaries.push((*name, mean));
    }

    println!("\n─── the comparison PENDING 1.2 turns on ───");
    let slowest_rust = summaries
        .iter()
        .filter(|(name, _)| *name != "onnx.reference")
        .map(|(_, mean)| *mean)
        .fold(0.0f64, f64::max);
    if let Some((_, reference_mean)) = summaries.iter().find(|(n, _)| *n == "onnx.reference") {
        let ratio = reference_mean / slowest_rust;
        println!("reference / slowest in-process runtime = {ratio:.2}×",);
        println!(
            "running the reference on every case changes total cost by {:+.1}%",
            100.0 * reference_mean / summaries.iter().map(|(_, m)| m).sum::<f64>()
        );
        println!();
        if ratio < 10.0 {
            println!("\"orders of magnitude slower\" is NOT supported by this measurement.");
        } else {
            println!("The reference is {ratio:.0}× the slowest runtime — confirmer is justified.");
        }
    }
    println!(
        "\nNote: the N1 corpus has almost no expensive tail. Re-measure after N3 widens\n\
         the generator, before using any of this to size a campaign."
    );
}

/// Print the distribution and return the mean, in seconds.
///
/// The mean alone cannot reveal a tail — that is the whole lesson of the throughput trap —
/// so the percentiles are printed beside it and the max is printed last.
fn report(name: &str, times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));

    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let at = |q: f64| sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)];

    println!(
        "{name:<22} mean {:>9.4} ms   median {:>9.4}   p99 {:>9.4}   max {:>9.4}   ({:.0}/sec)",
        mean * 1e3,
        at(0.50) * 1e3,
        at(0.99) * 1e3,
        sorted[sorted.len() - 1] * 1e3,
        1.0 / mean
    );
    mean
}
