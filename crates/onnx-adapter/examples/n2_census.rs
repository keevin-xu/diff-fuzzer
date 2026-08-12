//! Take the capability census and report the go/no-go.
//!
//! **This is PHASE-N2, and its gate can stop the domain.** `08-RISKS.md` §1 names a small
//! operator intersection as the risk most likely to sink this work, so the honest outcome of
//! this program may be "stop" or "reduce scope" — and that would be a successful phase, not
//! a failed one.
//!
//! The bar was agreed at G-N0, *before* any of these numbers existed: **≥20 operators
//! supported by ≥3 participants, of which ≥10 are Tier A and ≥8 are value-dependent.** The
//! last clause is the one added during review, and it is the load-bearing one — without it
//! the bar is clearable by `Shape`, `Size`, `Identity` and `Reshape`, none of which reads
//! its input values, and this domain's entire thesis is adversarial values.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n2_census --features candle
//!
//! Writes `findings/onnx/census.json`, which the capability model consumes at N5.

use std::collections::BTreeMap;

use diff_fuzzer_core::traits::Implementation;

use onnx_adapter::case::{ElemType, OnnxCase, OpKind};
use onnx_adapter::census::{self, GoNoGo, Support};
use onnx_adapter::model::DEFAULT_OPSET;
use onnx_adapter::ops::{self, Tier};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

type Participant = Box<dyn Implementation<In = OnnxCase, Out = OnnxOutcome>>;

fn main() {
    let opset = DEFAULT_OPSET;

    let mut owned: Vec<Participant> = vec![Box::new(OrtRuntime), Box::new(TractRuntime)];
    #[cfg(feature = "candle")]
    owned.push(Box::new(onnx_adapter::runtimes::CandleRuntime));
    match ReferenceRuntime::start() {
        Ok(reference) => owned.push(Box::new(reference)),
        Err(why) => {
            // Not a warning to be read past. The reference is the validity gate: without it
            // a probe failure cannot be told from our own malformed model, and the census
            // would be measuring the wrong thing.
            eprintln!("FATAL: the reference implementation is unavailable: {why}");
            eprintln!("The census cannot be trusted without it — it is the validity gate.");
            std::process::exit(1);
        }
    }
    let participants: Vec<&dyn Implementation<In = OnnxCase, Out = OnnxOutcome>> =
        owned.iter().map(std::convert::AsRef::as_ref).collect();

    let candidates = ops::candidates(opset);
    println!("Capability census — measured, not read from documentation");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("opset          {opset}");
    println!("operators      {}", OpKind::ALL.len());
    println!(
        "candidate pairs {} (operator × element type the spec permits)",
        candidates.len()
    );
    println!(
        "participants   {} — {}",
        participants.len(),
        participants
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!();

    let census = census::take(&participants, opset);

    // ── the matrix ────────────────────────────────────────────────────────────────
    println!("legend: + supported   - unsupported   ? rejected   ! CRASHED");
    println!();
    let names: Vec<&str> = census.runtimes.iter().map(String::as_str).collect();
    print!("{:<12}{:<7}", "operator", "type");
    for name in &names {
        print!("{:>16}", name);
    }
    println!();
    println!(
        "{:-<12}{:-<7}{:-<width$}",
        "",
        "",
        "",
        width = 16 * names.len()
    );

    for (op, elem) in &candidates {
        print!("{:<12}{:<7}", op.onnx_name(), format!("{elem:?}"));
        for name in &names {
            let cell = census
                .cells
                .iter()
                .find(|c| c.op == op.onnx_name() && c.elem_type == *elem && c.runtime == *name);
            let symbol = cell.map_or(' ', |c| c.support.symbol());
            print!("{symbol:>16}");
        }
        println!();
    }

    // ── per-runtime tally ─────────────────────────────────────────────────────────
    println!();
    println!("per-runtime outcomes over {} probes each", candidates.len());
    println!(
        "{:<18}{:>10}{:>14}{:>11}{:>10}",
        "runtime", "supported", "unsupported", "rejected", "CRASHED"
    );
    for (runtime, counts) in census.tally() {
        let get = |s: Support| counts.get(&s).copied().unwrap_or(0);
        println!(
            "{:<18}{:>10}{:>14}{:>11}{:>10}",
            runtime,
            get(Support::Supported),
            get(Support::Unsupported),
            get(Support::Rejected),
            get(Support::Crashed)
        );
    }

    // ── coverage by participant count ─────────────────────────────────────────────
    println!();
    println!("operators supported by N participants (at any element type)");
    for n in 1..=participants.len() {
        let ops_at_n = census.operators_supported_by(n);
        let tier_a = ops_at_n
            .iter()
            .filter(|o| ops::spec(**o).tier == Tier::A)
            .count();
        let value_dep = ops_at_n
            .iter()
            .filter(|o| ops::spec(**o).value_dependent)
            .count();
        println!(
            "  ≥{n}: {:>2} operators   ({tier_a} Tier A, {value_dep} value-dependent)",
            ops_at_n.len()
        );
    }

    // ── what nobody supports, and what only one does ──────────────────────────────
    let unsupported_anywhere: Vec<&OpKind> = OpKind::ALL
        .iter()
        .filter(|op| {
            ElemType::ALL
                .into_iter()
                .all(|e| census.supporting(**op, e).is_empty())
        })
        .collect();
    if !unsupported_anywhere.is_empty() {
        println!();
        println!("supported by NOBODY — worth checking these are not our bug:");
        for op in unsupported_anywhere {
            println!("  {}", op.onnx_name());
        }
    }

    // ── crashes are findings, not statistics ──────────────────────────────────────
    let crashes = census.crashes();
    println!();
    if crashes.is_empty() {
        println!("crashes: none on minimal valid models with ordinary values");
    } else {
        println!("CRASHES — {} of them, each a finding:", crashes.len());
        for cell in &crashes {
            println!(
                "  {} {} on {:?}: {}",
                cell.runtime,
                cell.op,
                cell.elem_type,
                cell.detail.as_deref().unwrap_or("")
            );
        }
    }

    // ── the verdict ───────────────────────────────────────────────────────────────
    let verdict = GoNoGo::measure(&census);
    println!();
    println!("═══ GO / NO-GO, against the bar agreed before any of this was measured ═══");
    let row = |label: &str, got: usize, need: usize| {
        println!(
            "  {label:<34} {got:>3} / {need:<3}  {}",
            if got >= need { "PASS" } else { "FAIL" }
        );
    };
    row(
        "operators on ≥3 participants",
        verdict.operators_3plus,
        census::MIN_OPERATORS,
    );
    row("of which Tier A", verdict.tier_a, census::MIN_TIER_A);
    row(
        "of which value-dependent",
        verdict.value_dependent,
        census::MIN_VALUE_DEPENDENT,
    );
    println!();
    println!(
        "  verdict: {}   tightest margin {:+.0}%",
        if verdict.passes() { "GO" } else { "NO-GO" },
        verdict.tightest_margin() * 100.0
    );
    if verdict.passes() && verdict.tightest_margin() < 0.20 {
        println!("  NOTE: cleared by under 20% on some clause — not a mandate. Discuss before");
        println!("        building on it.");
    }

    // ── store it as data ──────────────────────────────────────────────────────────
    let path = format!("{}/census.json", onnx_adapter::FINDINGS_ROOT);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).expect("creating the findings directory");
    }
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&census).expect("a census must serialize"),
    )
    .expect("writing the census");
    println!();
    println!("written to {path} ({} cells)", census.cells.len());

    // A brief per-runtime note on *why* things were declined — the reasons are where the
    // interesting capability facts live, and a bare matrix hides them.
    let mut reasons: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for cell in &census.cells {
        if cell.support != Support::Supported
            && let Some(detail) = &cell.detail
        {
            *reasons.entry((&cell.runtime, detail)).or_default() += 1;
        }
    }
    if !reasons.is_empty() {
        println!();
        println!("most common reasons for declining (top 12):");
        let mut sorted: Vec<_> = reasons.into_iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for ((runtime, detail), count) in sorted.into_iter().take(12) {
            println!("  {count:>3}× {runtime:<14} {detail}");
        }
    }
}
