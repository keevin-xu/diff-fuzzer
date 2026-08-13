//! **N4.6–N4.7** — turn on adversarial values and see what changes.
//!
//! Two questions, and the second is the one that makes the special-value table safe:
//!
//! 1. What does the special-value axis buy? Measured **against the ordinary-value baseline**,
//!    because *a rate without a baseline is not a measurement* — a rate of 1.1% is equally
//!    consistent with "special values found something" and "this is the background level".
//! 2. **Does the one loosening in the table ever fire?** `NaN` vs `NaN` agreeing accepts two
//!    differing bit patterns as equal. If runtimes never actually produce differing `NaN`
//!    payloads, the loosening costs nothing and rests on a citation nobody needs to lean on.
//!    If it fires often, it is doing real work and the citation matters.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n4_specials --features candle

use diff_fuzzer_core::axes::GenerationAxes;
use onnx_adapter::findings::{Run, StoredFinding};
use std::collections::BTreeMap;

use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation, NamedOutput, Oracle, Verdict};

use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::OnnxNormalizer;
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

fn wide_seeds(count: u64) -> impl Iterator<Item = u64> {
    (0..count).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn main() {
    let cases: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(6_000);

    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run n2_census first");
    let ort = WithCapabilities::new(OrtRuntime, &caps);
    let tract = WithCapabilities::new(TractRuntime, &caps);

    for (label, bounds) in [
        (
            "baseline (ordinary values)",
            Bounds::default().without_special_values(),
        ),
        ("special values on", Bounds::default().with_special_values()),
    ] {
        let generator = OnnxGenerator::new(bounds.clone());
        // Findings from a measurement run go where every other run's findings go — one run
        // directory per configuration, under the oracle that produced them.
        let mut log = Run::open(
            onnx_adapter::OracleKind::Differential,
            if bounds.special_values {
                "n4-special-values"
            } else {
                "n4-baseline"
            },
        )
        .expect("opening the run directory");
        let (mut judged, mut agreed, mut diverged, mut skipped, mut degenerate) = (0, 0, 0, 0, 0);
        let mut signatures: BTreeMap<String, usize> = BTreeMap::new();
        // How often the one loosening actually decides a case.
        let (mut nan_pairs, mut nan_differing_bits) = (0usize, 0usize);

        for seed in wide_seeds(cases) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let outputs: Vec<NamedOutput<_>> = [
                ("onnxruntime", ort.run(&case).expect("never Err")),
                ("tract", tract.run(&case).expect("never Err")),
            ]
            .into_iter()
            .map(|(n, o)| NamedOutput {
                implementation: n.to_string(),
                output: OnnxNormalizer.normalize(o),
            })
            .collect();

            // Does the NaN rule decide anything here?
            if let [a, b] = &outputs[..]
                && a.output.is_ok()
                && b.output.is_ok()
            {
                for (ta, tb) in a.output.tensors.iter().zip(b.output.tensors.iter()) {
                    if ta.elem_type.is_floating() && ta.bits.len() == tb.bits.len() {
                        for (x, y) in ta.bits.iter().zip(tb.bits.iter()) {
                            let (fx, fy) = (f64::from_bits(*x), f64::from_bits(*y));
                            let nan_x = if ta.elem_type == onnx_adapter::case::ElemType::F32 {
                                f32::from_bits(*x as u32).is_nan()
                            } else {
                                fx.is_nan()
                            };
                            let nan_y = if tb.elem_type == onnx_adapter::case::ElemType::F32 {
                                f32::from_bits(*y as u32).is_nan()
                            } else {
                                fy.is_nan()
                            };
                            if nan_x && nan_y {
                                nan_pairs += 1;
                                if x != y {
                                    nan_differing_bits += 1;
                                }
                            }
                        }
                    }
                }
            }

            if outputs.iter().any(|o| o.output.is_degenerate()) {
                degenerate += 1;
            }
            match OnnxOracle.check(&case, &outputs) {
                Verdict::Agree => {
                    agreed += 1;
                    judged += 1;
                }
                Verdict::Diverged(d) => {
                    diverged += 1;
                    judged += 1;
                    let signature = format!("{} | {}", case.op.onnx_name(), d.summary);
                    *signatures.entry(signature.clone()).or_default() += 1;

                    // **The gate asks for divergences logged, not for a log that works.** One
                    // record per distinct signature, each carrying the whole serialized case —
                    // so a reader six months from now can replay it without this generator, and
                    // without trusting that seed 901 still means what it meant today.
                    let finding = StoredFinding::new(
                        signature,
                        d.summary.clone(),
                        seed,
                        bounds.description(),
                        case.clone(),
                        d.outputs.clone(),
                    );
                    log.record(&finding)
                        .expect("writing a finding must not fail");
                }
                Verdict::Skipped(_) => skipped += 1,
            }
        }

        println!("\n══ {label} ══");
        println!(
            "  {} cases: {judged} judged ({agreed} agree, {diverged} diverge), {skipped} skipped",
            cases
        );
        println!(
            "  disagreement rate      {:.2}% of judged",
            100.0 * diverged as f64 / judged.max(1) as f64
        );
        println!(
            "  degenerate results     {:.1}% (a result two runtimes cannot disagree about)",
            100.0 * degenerate as f64 / cases as f64
        );
        println!(
            "  effective bound        1 in {:.0} judged, non-degenerate",
            (judged - degenerate.min(judged)).max(1) as f64
        );
        println!("  NaN-vs-NaN element pairs compared: {nan_pairs}");
        println!(
            "    of which the bit patterns DIFFER: {nan_differing_bits}  <- the loosening firing"
        );
        println!("  distinct divergence signatures: {}", signatures.len());
        println!("  recorded in the findings log:   {}", log.distinct());
        for (sig, n) in signatures.iter().take(12) {
            println!("    {n:>4}x  {sig}");
        }
    }
}
