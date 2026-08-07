//! What is actually in the corpus a setting produces?
//!
//! This exists because the same mistake has now happened three times: enabling one axis
//! silently *removed* other cases, and the verdict counts looked fine each time. A run
//! reporting "5,000 cases, 100% agreed" is worthless if 97% of those cases were shapes the
//! previous run already covered — and nothing in the verdict tells you.
//!
//! So: before trusting any campaign, look at what it generates. Especially the
//! **interactions**, which are the whole point of running the axes together — an aggregate
//! over an outer-joined table, a set operation whose branches are grouped.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example corpus_shape -- [cases] [setting]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::Generator;
use sql_adapter::ast::SqlCase;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::oracle::SortMode;

/// One property worth counting, and how to detect it.
struct Feature {
    name: &'static str,
    holds: fn(&SqlCase) -> bool,
}

const FEATURES: &[Feature] = &[
    Feature {
        name: "where",
        holds: |case| case.query.filter.is_some(),
    },
    Feature {
        name: "order by",
        holds: |case| !case.query.order_by.is_empty(),
    },
    Feature {
        name: "totally ordered",
        holds: |case| SortMode::for_case(case) == SortMode::Ordered,
    },
    Feature {
        name: "limit",
        holds: |case| case.query.limit.is_some(),
    },
    Feature {
        name: "group by",
        holds: |case| !case.query.group_by.is_empty(),
    },
    Feature {
        name: "aggregate",
        holds: SqlCase::aggregates,
    },
    Feature {
        name: "join",
        holds: |case| case.query.join.is_some(),
    },
    Feature {
        name: "outer join",
        holds: |case| {
            case.query
                .join
                .as_ref()
                .is_some_and(|join| join.kind.pads_with_nulls())
        },
    },
    Feature {
        name: "set op",
        holds: |case| case.query.set_op.is_some(),
    },
    Feature {
        name: "chained set op",
        holds: |case| {
            case.query
                .set_op
                .as_ref()
                .is_some_and(|branch| branch.right.set_op.is_some())
        },
    },
    Feature {
        name: "empty table",
        holds: |case| case.queried_rows().is_empty(),
    },
    Feature {
        name: "null in data",
        holds: |case| {
            case.data.iter().any(|insert| {
                insert
                    .rows
                    .iter()
                    .flatten()
                    .any(|value| matches!(value, sql_adapter::schema::Literal::Null))
            })
        },
    },
];

/// Combinations that no single-axis run can produce, and that are the reason to run the axes
/// together at all.
const INTERACTIONS: &[Feature] = &[
    Feature {
        name: "aggregate over a join",
        holds: |case| case.aggregates() && case.query.join.is_some(),
    },
    Feature {
        name: "aggregate over an OUTER join",
        holds: |case| {
            case.aggregates()
                && case
                    .query
                    .join
                    .as_ref()
                    .is_some_and(|join| join.kind.pads_with_nulls())
        },
    },
    Feature {
        name: "grouped query over a join",
        holds: |case| !case.query.group_by.is_empty() && case.query.join.is_some(),
    },
    Feature {
        name: "set op over a joined query",
        holds: |case| case.query.set_op.is_some() && case.query.join.is_some(),
    },
    Feature {
        name: "outer join + NULL in the data",
        holds: |case| {
            case.query
                .join
                .as_ref()
                .is_some_and(|join| join.kind.pads_with_nulls())
                && case.data.iter().any(|insert| {
                    insert
                        .rows
                        .iter()
                        .flatten()
                        .any(|value| matches!(value, sql_adapter::schema::Literal::Null))
                })
        },
    },
];

fn main() {
    let mut arguments = std::env::args().skip(1);
    let total: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let bounds = match arguments.next().as_deref() {
        Some("all") => Bounds::V1_ALL,
        Some("joins") => Bounds::V1_JOINS,
        Some("setops") => Bounds::V1_SET_OPS,
        Some("chained") => Bounds::V1_CHAINED_SET_OPS,
        Some("aggregates") => Bounds::V1_AGGREGATES,
        Some("wide") => Bounds::V1_WIDE_ARITHMETIC,
        _ => Bounds::V1,
    };

    let generator = SqlGenerator::new(bounds);
    println!(
        "generator: {}\ncases:     {total}\n",
        generator.description()
    );

    let cases: Vec<SqlCase> = (0..total as u64)
        .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
        .collect();

    let report = |label: &str, features: &[Feature]| {
        println!("{label}");
        for feature in features {
            let count = cases.iter().filter(|case| (feature.holds)(case)).count();
            let share = 100.0 * count as f64 / total as f64;
            let bar = "#".repeat((share / 2.5).round() as usize);
            println!("  {:<28} {count:>6} ({share:>5.1}%) {bar}", feature.name);
        }
        println!();
    };

    report("constructs", FEATURES);
    report(
        "interactions (only the combined setting can produce these)",
        INTERACTIONS,
    );

    // The check that would have caught all three confounds: a construct at 0% is a construct
    // the run does not test, whatever its verdict counts say.
    let absent: Vec<&str> = FEATURES
        .iter()
        .chain(INTERACTIONS)
        .filter(|feature| !cases.iter().any(|case| (feature.holds)(case)))
        .map(|feature| feature.name)
        .collect();
    if absent.is_empty() {
        println!("every construct above occurs at least once.");
    } else {
        println!("NOT PRESENT AT ALL: {}", absent.join(", "));
        println!("A construct at 0% is untested, whatever the verdicts say.");
    }
}
