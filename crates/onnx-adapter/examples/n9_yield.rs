//! **N9.7** — what the quantized surface was actually able to judge.
//!
//! "Zero findings" and "nothing was judged" look identical in a divergence count and mean
//! opposite things. The census says `tract` rejects `QuantizeLinear` and `DequantizeLinear` and
//! candle has no `int8` at all, so those two operators have **one** participant — and a
//! differential oracle over one participant judges nothing. This separates the two.
use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Oracle, SkipReason, Verdict,
};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::ops::{self, Tier};
use onnx_adapter::oracle::OnnxOracle;
use std::collections::BTreeMap;

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT)).unwrap();
    let t = WithCapabilities::new(TractRuntime, &caps);
    let o = WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let c = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);
    let g = OnnxGenerator::new(Bounds::default().with_special_values().with_quantized());

    // per operator: generated, judged, agreed, diverged, skipped-too-few
    let mut stats: BTreeMap<&'static str, [usize; 5]> = BTreeMap::new();

    for seed in 0..60000u64 {
        let case = g.generate(&mut SeededRng::from_seed(seed));
        if ops::spec(case.op).tier != Tier::Q || !onnx_adapter::validation::is_valid(&case) {
            continue;
        }
        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut outs = Vec::from([
            ("tract", t.run(&case).unwrap()),
            ("onnxruntime", o.run(&case).unwrap()),
        ]);
        #[cfg(feature = "candle")]
        outs.push(("candle", c.run(&case).unwrap()));
        let named: Vec<NamedOutput<Canonical>> = outs
            .iter()
            .map(|(n, x)| NamedOutput {
                implementation: (*n).into(),
                output: OnnxNormalizer.normalize(x.clone()),
            })
            .collect();
        let e = stats.entry(case.op.onnx_name()).or_insert([0; 5]);
        e[0] += 1;
        match OnnxOracle.check(&case, &named) {
            Verdict::Agree => {
                e[1] += 1;
                e[2] += 1;
            }
            Verdict::Diverged(_) => {
                e[1] += 1;
                e[3] += 1;
            }
            Verdict::Skipped(SkipReason::TooFewResults { .. }) => e[4] += 1,
            Verdict::Skipped(_) => {}
        }
    }
    std::panic::set_hook(previous);

    println!("\nquantized surface — what was judged, over 60,000 seeds\n");
    println!(
        "{:<24} {:>10} {:>8} {:>8} {:>9} {:>18}",
        "operator", "generated", "judged", "agreed", "diverged", "skipped: 1 runtime"
    );
    for (op, s) in &stats {
        println!(
            "{op:<24} {:>10} {:>8} {:>8} {:>9} {:>18}",
            s[0], s[1], s[2], s[3], s[4]
        );
    }
    let total: usize = stats.values().map(|s| s[0]).sum();
    let judged: usize = stats.values().map(|s| s[1]).sum();
    let diverged: usize = stats.values().map(|s| s[3]).sum();
    println!(
        "\n  {total} quantized cases generated, {judged} judged ({:.0}%), {diverged} diverged",
        100.0 * judged as f64 / total.max(1) as f64
    );
    if judged == 0 {
        println!("  *** nothing was judged — a zero here means the oracle could not look ***");
    } else if diverged == 0 {
        println!(
            "  zero divergences out of {judged} judged: an honest zero, not an absence of looking"
        );
    }
}
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
