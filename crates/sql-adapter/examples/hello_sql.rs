//! The smallest possible proof that both engines work: run the *same* SQL on SQLite and
//! DuckDB, in memory, and print what each returns.
//!
//! There is no framework here on purpose — no traits, no oracle, no comparison in code.
//! The point of this step is narrower than it looks: show that two independent database
//! engines embed inside one Rust process, accept identical SQL text, and hand back
//! identical answers. Everything built later assumes all three of those, and each is
//! cheaper to disprove now than after a generator sits on top of it.
//!
//! Run with:  cargo run -p sql-adapter --example hello_sql

use std::error::Error;

/// The case, as three statements. Identical text for both engines.
///
/// Deliberately dull SQL: an integer column, a text column, one `NULL`, and a `WHERE`
/// that excludes exactly one row. `ORDER BY a` makes the row order part of the answer
/// rather than something each engine may choose — without it, comparing printed output
/// would be comparing something neither engine promises.
const SCHEMA: &str = "CREATE TABLE t (a INTEGER, b TEXT)";
const DATA: &str = "INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, NULL), (-1, 'neg')";
const QUERY: &str = "SELECT a, b FROM t WHERE a > 0 ORDER BY a";

/// One row of the answer, in a form both engines can be rendered into.
///
/// `Option<String>` is how Rust says "this may be `NULL`" without a null of its own:
/// `Some(text)` or `None`, and the compiler will not let the `None` case go unhandled.
/// That is the whole reason a `NULL` sits in the seed data — it is the first thing that
/// would differ between the two drivers, so it should appear in the very first test.
type Row = (i64, Option<String>);

fn main() -> Result<(), Box<dyn Error>> {
    // `Box<dyn Error>` lets one function return errors of *different* types. `rusqlite`
    // and `duckdb` each define their own error type, and `?` converts either into this
    // box automatically. In library code we will use a named error type instead; for an
    // example, this is the honest minimum.
    let (sqlite_version, sqlite_rows) = run_sqlite()?;
    let (duckdb_version, duckdb_rows) = run_duckdb()?;

    print_result("SQLite", &sqlite_version, &sqlite_rows);
    print_result("DuckDB", &duckdb_version, &duckdb_rows);

    // Printed, not asserted. Judging agreement is the oracle's job and it does not exist
    // yet; claiming a verdict here would be a comparison nobody designed.
    println!("\nsame text ran on both engines. compare the two grids above by eye.");

    Ok(())
}

/// Open a fresh in-memory SQLite database, apply the case, and read the answer back.
fn run_sqlite() -> Result<(String, Vec<Row>), Box<dyn Error>> {
    // `:memory:` — the database exists only inside this process and vanishes when the
    // connection is dropped. That is what makes a case self-contained: no file, no
    // cleanup, and nothing that can leak into the next case.
    let conn = rusqlite::Connection::open_in_memory()?;

    let version: String = conn.query_row("SELECT sqlite_version()", [], |row| row.get(0))?;

    conn.execute(SCHEMA, [])?;
    conn.execute(DATA, [])?;

    // `prepare` compiles the statement, `query_map` runs it and applies a closure to
    // each row. The closure returns `Result`, so a type mismatch surfaces as an error
    // rather than a panic — worth noticing, because reading a cell as the wrong type is
    // exactly the kind of bug this project exists to find in *other* people's code.
    let mut statement = conn.prepare(QUERY)?;
    let rows = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        // Each item is a `Result<Row, _>`; collecting into `Result<Vec<_>, _>` turns a
        // sequence of results into one result holding the whole vector — the first
        // failure wins. This is a standard Rust idiom worth recognising on sight.
        .collect::<Result<Vec<Row>, _>>()?;

    Ok((version, rows))
}

/// The same, against DuckDB. The driver is modelled on `rusqlite`, so the code is nearly
/// identical — which is convenient here and is *not* something the rest of the project
/// will rely on.
fn run_duckdb() -> Result<(String, Vec<Row>), Box<dyn Error>> {
    let conn = duckdb::Connection::open_in_memory()?;

    let version: String = conn.query_row("SELECT version()", [], |row| row.get(0))?;

    conn.execute(SCHEMA, [])?;
    conn.execute(DATA, [])?;

    let mut statement = conn.prepare(QUERY)?;
    let rows = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<Vec<Row>, _>>()?;

    Ok((version, rows))
}

/// Print one engine's answer as a small grid.
///
/// `NULL` is printed as the word `NULL` rather than as an empty cell, so it cannot be
/// confused with an empty string. Distinguishing those two is a real requirement later —
/// it is one of the rules the result normalizer will adopt from `sqllogictest`.
fn print_result(engine: &str, version: &str, rows: &[Row]) {
    println!("\n{engine} ({version})");
    println!("  {QUERY}");
    for (a, b) in rows {
        let b = b.as_deref().unwrap_or("NULL");
        println!("    {a} | {b}");
    }
    println!("  {} rows", rows.len());
}
