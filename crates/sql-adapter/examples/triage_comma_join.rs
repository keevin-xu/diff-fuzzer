//! Put the one real finding through the pipeline that has never seen one.
//!
//! # Why this exists
//!
//! The shrinker (S5), the signature (S5), `known.rs` (S4) and the reporting (S5) were all
//! written, unit-tested against **fabricated** cases, and never run on a real divergence —
//! because until S10 this project had none it could express. Tests written by the same person
//! who wrote the code, against inputs that person invented, do not establish that the machinery
//! works on something it did not anticipate.
//!
//! The finding: `FROM a, b RIGHT JOIN c ON b.y = c.z` parses differently on the two engines.
//! SQLite gives all join operators equal precedence left to right and **documents this as
//! incorrect** (`SPECS.md` §2.11); DuckDB binds the comma loosest, per the standard.
//!
//! This starts from a deliberately **bloated** version of that case — extra tables, extra
//! columns, extra rows, an irrelevant predicate — and asks the minimizer to find the small one.
//! Starting from the already-minimal case would prove nothing.
use diff_fuzzer_core::minimize::minimize;
use diff_fuzzer_core::traits::Implementation;
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::known::known_comma_join_defect;
use sql_adapter::render::Dialect;
use sql_adapter::schema::{
    BinaryOp, Column, ColumnRef, Expr, Index, InsertRows, Join, JoinKind, Literal, SelectStmt,
    SqlType, Table,
};
use sql_adapter::shrink::complexity;
use sql_adapter::signature::{DisagreementKind, signature};

fn column(table: &str, name: &str) -> ColumnRef {
    ColumnRef {
        table: table.to_string(),
        column: name.to_string(),
    }
}

/// The finding, buried in noise a real campaign would also produce.
fn bloated() -> SqlCase {
    let table = |name: &str, columns: &[&str]| Table {
        name: name.to_string(),
        columns: columns
            .iter()
            .map(|c| Column {
                name: (*c).to_string(),
                sql_type: SqlType::Integer,
            })
            .collect(),
    };
    let rows = |name: &str, values: &[&[i64]]| InsertRows {
        table: name.to_string(),
        rows: values
            .iter()
            .map(|row| row.iter().map(|v| Literal::Integer(*v)).collect())
            .collect(),
    };

    SqlCase {
        schema: vec![
            table("a", &["x", "x2", "x3"]),
            table("b", &["y", "y2"]),
            table("c", &["z"]),
            // A table the query never reads — the shrinker should drop it entirely.
            table("d", &["w"]),
        ],
        indexes: vec![Index {
            name: "i0".to_string(),
            table: "a".to_string(),
            columns: vec!["x".to_string()],
            unique: false,
        }],
        data: vec![
            rows("a", &[&[1, 10, 100], &[2, 20, 200], &[3, 30, 300]]),
            rows("b", &[&[2, 22], &[4, 44]]),
            rows("c", &[&[3]]),
            rows("d", &[&[9]]),
        ],
        query: SelectStmt {
            having: None,
            distinct: false,
            projection: vec![
                Expr::Column(column("a", "x")),
                Expr::Column(column("a", "x2")),
                Expr::Column(column("b", "y")),
                Expr::Column(column("c", "z")),
            ],
            from: vec!["a".to_string(), "b".to_string()],
            join: Some(Join {
                kind: JoinKind::Right,
                table: "c".to_string(),
                on: Expr::Binary {
                    op: BinaryOp::Equal,
                    left: Box::new(Expr::Column(column("b", "y"))),
                    right: Box::new(Expr::Column(column("c", "z"))),
                },
            }),
            set_op: None,
            group_by: Vec::new(),
            // An irrelevant predicate: true of every row, so removing it cannot change the
            // divergence. The shrinker should discover that rather than be told.
            filter: Some(Expr::Binary {
                op: BinaryOp::GreaterOrEqual,
                left: Box::new(Expr::Column(column("a", "x3"))),
                right: Box::new(Expr::Literal(Literal::Integer(-1000))),
            }),
            order_by: Vec::new(),
            limit: None,
        },
    }
}

/// Do the engines disagree on this case? The predicate the minimizer preserves.
fn diverges(case: &SqlCase) -> bool {
    match (SqliteImpl.run(case), DuckDbImpl.run(case)) {
        (Ok(left), Ok(right)) => left != right,
        // A case one engine cannot run is not a divergence — the same rule the oracle uses.
        _ => false,
    }
}

fn main() {
    let case = bloated();
    println!("== the finding, as a campaign might record it ==");
    assert!(case.validate().is_ok(), "the bloated case must be valid");
    assert!(diverges(&case), "the bloated case must actually diverge");
    for statement in case.statements(Dialect::Sqlite) {
        println!("  {statement};");
    }
    let before = complexity(&case);
    println!("\n  complexity (query nodes, data cells): {before:?}");
    println!(
        "  signature: {}",
        signature(&case, DisagreementKind::RowContent)
    );
    println!(
        "  known-defect catalog: {:?}",
        known_comma_join_defect(&case).map(|entry| entry.name)
    );

    println!("\n== minimizing ==");
    let minimized = minimize(case, diverges);
    let after = complexity(&minimized.input);

    for statement in minimized.input.statements(Dialect::Sqlite) {
        println!("  {statement};");
    }
    println!(
        "\n  {} steps, {} candidates tried, stopped: {:?}",
        minimized.steps, minimized.candidates_tried, minimized.stopped
    );
    println!("  complexity {before:?} -> {after:?}");
    println!(
        "  signature: {}",
        signature(&minimized.input, DisagreementKind::RowContent)
    );

    println!("\n== checks ==");
    let still_diverges = diverges(&minimized.input);
    println!("  still diverges:            {still_diverges}");
    println!(
        "  still valid:               {}",
        minimized.input.validate().is_ok()
    );
    println!(
        "  still caught by the catalog: {:?}",
        known_comma_join_defect(&minimized.input).map(|entry| entry.name)
    );
    println!(
        "  signature preserved:       {}",
        signature(&minimized.input, DisagreementKind::RowContent)
            == signature(&bloated(), DisagreementKind::RowContent)
    );
    println!(
        "  strictly simpler:          {}",
        after.0 <= before.0 && after.1 <= before.1 && after != before
    );
}
