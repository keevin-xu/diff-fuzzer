//! The precedence question, asked directly — the smallest form of the domain's first
//! genuine semantic divergence.
//!
//! `A UNION B INTERSECT C`, unparenthesized, has two readings:
//!
//! - **left to right**: `(A UNION B) INTERSECT C` — what SQLite documents itself as doing,
//!   and documents as *disagreeing with SQL92* (`SPECS.md` §1.4).
//! - **`INTERSECT` first**: `A UNION (B INTERSECT C)` — what SQL92 requires.
//!
//! DuckDB documents neither (`SPECS.md` §5.9 — two failed retrievals), so this program is
//! the only way to find out which it does.
//!
//! Run with: cargo run --release -p sql-adapter --example precedence

fn column(conn_rows: Vec<i64>) -> Vec<i64> {
    let mut rows = conn_rows;
    rows.sort_unstable();
    rows
}

fn main() {
    let sqlite = rusqlite::Connection::open_in_memory().expect("open sqlite");
    let duckdb = duckdb::Connection::open_in_memory().expect("open duckdb");

    let ambiguous = "SELECT 1 UNION SELECT 2 INTERSECT SELECT 2";

    let from_sqlite = column({
        let mut statement = sqlite.prepare(ambiguous).expect("sqlite prepares");
        let rows: Vec<i64> = statement
            .query_map([], |row| row.get(0))
            .expect("sqlite runs")
            .map(|value| value.expect("a row"))
            .collect();
        rows
    });
    let from_duckdb = column({
        let mut statement = duckdb.prepare(ambiguous).expect("duckdb prepares");
        let rows: Vec<i64> = statement
            .query_map([], |row| row.get(0))
            .expect("duckdb runs")
            .map(|value| value.expect("a row"))
            .collect();
        rows
    });

    println!("{ambiguous}\n");
    println!("  sqlite  {from_sqlite:?}");
    println!("  duckdb  {from_duckdb:?}\n");
    println!(
        "  (A UNION B) INTERSECT C  would be [2]        — left to right, SQLite's documented rule"
    );
    println!("  A UNION (B INTERSECT C)  would be [1, 2]     — INTERSECT first, the SQL92 rule\n");

    match (from_sqlite.as_slice(), from_duckdb.as_slice()) {
        ([2], [1, 2]) => println!(
            "SQLite groups left to right; DuckDB binds INTERSECT tighter. \
             Each is self-consistent, and only SQLite documents which it does."
        ),
        _ => println!(
            "Behaviour has changed since 2026-08-06 — re-derive before trusting anything downstream."
        ),
    }

    // Worth recording: SQLite cannot even express the grouping explicitly.
    let parenthesized = "(SELECT 1 UNION SELECT 2) INTERSECT SELECT 2";
    match sqlite.prepare(parenthesized) {
        Ok(_) => println!("\nSQLite accepts a parenthesized compound select."),
        Err(error) => println!(
            "\nSQLite rejects a parenthesized compound select — the grouping cannot be \
             written out to disambiguate it:\n  {error}"
        ),
    }
    match duckdb.prepare(parenthesized) {
        Ok(_) => println!("DuckDB accepts one, so the ambiguity is resolvable there."),
        Err(error) => println!("DuckDB rejects one too: {error}"),
    }
}
