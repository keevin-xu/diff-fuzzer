//! The control for the coverage measurement.
//!
//! Identical to `sql_diff` except that it **never calls either engine** — it generates a case
//! and renders it, then stops. If its counter count is close to `sql_diff`'s, then executing
//! SQLite and DuckDB contributes almost nothing to coverage, which means libFuzzer is steering
//! on our own Rust and not on the software under test.
//!
//! Without this control, `sql_diff`'s number is just a number: 1,867 counters could mean
//! "the engines are barely instrumented" or "the engines are heavily instrumented and small".
//! A rate without a baseline is not a measurement.

#![no_main]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::Generator;
use libfuzzer_sys::fuzz_target;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::render::Dialect;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let seed = u64::from_le_bytes(data[..8].try_into().expect("eight bytes"));

    let generator = SqlGenerator::new(Bounds::V1_ALL);
    let case = generator.generate(&mut SeededRng::from_seed(seed));

    // Render both dialects so the renderer is exercised too — everything `sql_diff` does
    // except handing the SQL to an engine.
    std::hint::black_box(case.statements(Dialect::Sqlite));
    std::hint::black_box(case.statements(Dialect::DuckDb));
});
