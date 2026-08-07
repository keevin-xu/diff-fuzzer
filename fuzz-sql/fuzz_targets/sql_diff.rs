//! A deliberately minimal target, built to answer **one question**: do libFuzzer's coverage
//! counters reach inside SQLite and DuckDB, or do they only see our own Rust?
//!
//! # Why this is not a real fuzzer, and must not be mistaken for one
//!
//! It decodes eight bytes into a seed and hands that to the existing generator. Mutating
//! those bytes therefore just picks a different random case — there is no gradient for
//! libFuzzer to follow, so as a *fuzzer* this is strictly worse than running `hunt`.
//!
//! But it exercises both engines fully on every execution, which is all the counters need.
//! If the answer is "coverage is blind", the real byte→`SqlCase` decoder (S6.2) never has to
//! be written, and the phase ends having cost an hour instead of a week. If the answer is
//! "coverage reaches the engines", *then* the decoder is worth building, because only then
//! does a gradient exist to follow.

#![no_main]

use diff_fuzzer_core::traits::{Implementation, Normalizer, Oracle, Verdict};
use libfuzzer_sys::fuzz_target;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::normalize::SqlNormalizer;
use sql_adapter::oracle::SqlDifferentialOracle;
use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, NamedOutput};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let seed = u64::from_le_bytes(data[..8].try_into().expect("eight bytes"));

    let generator = SqlGenerator::new(Bounds::V1_ALL);
    let case = generator.generate(&mut SeededRng::from_seed(seed));

    // Both engines, every execution — the point of the exercise.
    let (Ok(from_sqlite), Ok(from_duckdb)) = (SqliteImpl.run(&case), DuckDbImpl.run(&case)) else {
        return;
    };

    let outputs = [
        NamedOutput {
            implementation: "sqlite".to_string(),
            output: SqlNormalizer.normalize(from_sqlite),
        },
        NamedOutput {
            implementation: "duckdb".to_string(),
            output: SqlNormalizer.normalize(from_duckdb),
        },
    ];

    // A panic is the bug signal libFuzzer understands. Minimization and reporting live in
    // `hunt`; this target only has to notice.
    if let Verdict::Diverged(divergence) = SqlDifferentialOracle.check(&case, &outputs) {
        panic!("divergence at seed {seed}: {}", divergence.summary);
    }
});
