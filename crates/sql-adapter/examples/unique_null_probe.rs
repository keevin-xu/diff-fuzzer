//! Does a `UNIQUE` constraint permit several `NULL`s? Measured, because it is not documented.
//!
//! `SPECS.md` §2.8 cites SQLite: *"For the purposes of unique indices, all NULL values are
//! considered different from all other NULL values and are thus unique."* — so SQLite permits
//! any number of `NULL` rows.
//!
//! DuckDB's side is **not documented**: neither its constraints page nor its indexes page states
//! how multiple `NULL`s are treated (`SPECS.md` §5.11, two failed retrievals). Rather than assume
//! it matches SQLite, this measures it — and the result must be recorded as **measured**, never
//! written as though a specification said it.
//!
//! Both spellings are tried, because they are different code paths: a column-level `UNIQUE`
//! constraint, and a standalone `CREATE UNIQUE INDEX`.
use duckdb::Connection as DuckConnection;
use rusqlite::Connection as SqliteConnection;

fn main() {
    let cases: [(&str, &[&str]); 2] = [
        (
            "column-level UNIQUE constraint",
            &[
                "CREATE TABLE t (a INTEGER UNIQUE)",
                "INSERT INTO t VALUES (NULL)",
                "INSERT INTO t VALUES (NULL)",
            ],
        ),
        (
            "standalone CREATE UNIQUE INDEX",
            &[
                "CREATE TABLE t (a INTEGER)",
                "CREATE UNIQUE INDEX i ON t (a)",
                "INSERT INTO t VALUES (NULL)",
                "INSERT INTO t VALUES (NULL)",
            ],
        ),
    ];

    for (label, statements) in cases {
        println!("== {label} ==");

        let sqlite = SqliteConnection::open_in_memory().expect("sqlite opens");
        let mut sqlite_result = Ok(());
        for statement in statements {
            if let Err(error) = sqlite.execute(statement, []) {
                sqlite_result = Err(format!("{error}"));
                break;
            }
        }
        match &sqlite_result {
            Ok(()) => {
                let n: i64 = sqlite
                    .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
                    .unwrap_or(-1);
                println!("  sqlite: accepted both NULLs, COUNT(*) = {n}");
            }
            Err(e) => println!("  sqlite: REFUSED — {e}"),
        }

        let duck = DuckConnection::open_in_memory().expect("duckdb opens");
        let mut duck_result = Ok(());
        for statement in statements {
            if let Err(error) = duck.execute(statement, []) {
                duck_result = Err(format!("{error}"));
                break;
            }
        }
        match &duck_result {
            Ok(()) => {
                let n: i64 = duck
                    .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
                    .unwrap_or(-1);
                println!("  duckdb: accepted both NULLs, COUNT(*) = {n}");
            }
            Err(e) => println!("  duckdb: REFUSED — {e}"),
        }

        println!(
            "  => {}",
            if sqlite_result.is_ok() == duck_result.is_ok() {
                "AGREE — no divergence here; a UNIQUE axis would be a catalogue entry at best"
            } else {
                "DIVERGE — one engine permits what the other refuses"
            }
        );
    }
}
