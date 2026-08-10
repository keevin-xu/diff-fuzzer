//! Does SQLite really parse comma-joins with the wrong precedence, and does DuckDB differ?
//!
//! `SPECS.md` §2.11 quotes SQLite's own documentation: *"SQLite gives all join operators equal
//! precedence and processes them from left to right. But this is not quite correct… comma-joins
//! have lower precedence than all others join operators."* — an engine documenting a defect in
//! its own parser, on a construct this project has never generated.
//!
//! # The case, and why it distinguishes the two parses
//!
//! `FROM a, b RIGHT JOIN c ON b.x = c.x` where `b.x` does **not** match `c.x`:
//!
//! - **Standard** (comma binds loosest): `a CROSS JOIN (b RIGHT JOIN c)`. The right join finds no
//!   match, so it yields `(NULL, 3)`; crossing with `a` gives **`(1, NULL, 3)`**.
//! - **SQLite's documented rule** (all joins equal, left to right): `(a CROSS JOIN b) RIGHT JOIN
//!   c`. The cross join gives `(1, 2)`; the right join finds no match, so `a`'s column is also
//!   nulled, giving **`(NULL, NULL, 3)`**.
//!
//! Same row count, different values — so a difference here cannot be explained away as row
//! ordering or as a legal difference. And unlike every other lead in this project, **SQLite's
//! documentation already says which side is wrong.**
use duckdb::Connection as DuckConnection;
use rusqlite::Connection as SqliteConnection;

const SETUP: [&str; 6] = [
    "CREATE TABLE a (x INTEGER)",
    "CREATE TABLE b (y INTEGER)",
    "CREATE TABLE c (z INTEGER)",
    "INSERT INTO a VALUES (1)",
    "INSERT INTO b VALUES (2)",
    "INSERT INTO c VALUES (3)",
];

const QUERY: &str = "SELECT a.x, b.y, c.z FROM a, b RIGHT JOIN c ON b.y = c.z";

fn main() {
    println!("query: {QUERY}\n");

    let sqlite = SqliteConnection::open_in_memory().expect("sqlite opens");
    for statement in SETUP {
        sqlite.execute(statement, []).expect("sqlite setup");
    }
    let sqlite_rows: Result<Vec<String>, _> = (|| {
        let mut prepared = sqlite.prepare(QUERY)?;
        let rows = prepared.query_map([], |row| {
            Ok(format!(
                "({:?}, {:?}, {:?})",
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<i64>>(2)?
            ))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
    })();

    let duck = DuckConnection::open_in_memory().expect("duckdb opens");
    for statement in SETUP {
        duck.execute(statement, []).expect("duckdb setup");
    }
    let duck_rows: Result<Vec<String>, _> = (|| {
        let mut prepared = duck.prepare(QUERY)?;
        let rows = prepared.query_map([], |row| {
            Ok(format!(
                "({:?}, {:?}, {:?})",
                row.get::<_, Option<i64>>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, Option<i64>>(2)?
            ))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
    })();

    match &sqlite_rows {
        Ok(rows) => println!("  sqlite: {rows:?}"),
        Err(e) => println!("  sqlite: REFUSED — {e}"),
    }
    match &duck_rows {
        Ok(rows) => println!("  duckdb: {rows:?}"),
        Err(e) => println!("  duckdb: REFUSED — {e}"),
    }

    println!();
    println!(
        "  standard parse  a CROSS JOIN (b RIGHT JOIN c)  would give  [\"(Some(1), None, Some(3))\"]"
    );
    println!(
        "  sqlite's rule   (a CROSS JOIN b) RIGHT JOIN c  would give  [\"(None, None, Some(3))\"]"
    );
    println!();
    match (&sqlite_rows, &duck_rows) {
        (Ok(l), Ok(r)) if l == r => {
            println!("  => AGREE. The documented parser defect is not observable this way.")
        }
        (Ok(_), Ok(_)) => println!(
            "  => DIVERGE, and SQLite's own documentation says which side is wrong. \
             This is worth building an axis for."
        ),
        _ => println!("  => one engine refused the query; the construct may be unreachable"),
    }
}
