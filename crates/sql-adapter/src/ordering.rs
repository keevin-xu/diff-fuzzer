//! Whether a query's `ORDER BY` actually orders the rows of *this* case.
//!
//! # Why this is one function used in two places
//!
//! Two parts of this crate need the same answer, and they must never disagree:
//!
//! - The **generator** may only attach a `LIMIT` to a query whose order is total. Without
//!   a total order, `LIMIT n` lets two engines legally return *different rows* — not the
//!   same rows in a different order — which sorting cannot repair.
//! - The **normalizer** (S3) decides from the same property whether to compare row order
//!   as given or to sort both sides first.
//!
//! If those two used different notions of "total", the tool would generate a `LIMIT` query
//! believing its order was total and then compare it believing the opposite. The result
//! would be divergences invented by this crate, indistinguishable from findings. One
//! definition removes the possibility.
//!
//! # Totality is a property of the case, not of the query
//!
//! `ORDER BY c0` orders the rows totally when no two rows share a value in `c0`, and fails
//! to when they do — so the answer depends on the **data**, which is why this takes rows
//! rather than just the statement. That is the whole reason the plan's original "read the
//! sort mode from the AST" was wrong.
//!
//! Two rows that tie on every ordering column may be returned in either order by either
//! engine, and neither is wrong.

use crate::schema::{ColumnRef, Literal, OrderKey, Table};

/// Does `order_by` put the rows of `table` into exactly one legal order?
///
/// True when the ordering columns, taken together, take a distinct combination of values in
/// every row. Vacuously true for fewer than two rows: with nothing to swap, there is only
/// one possible order.
///
/// Returns `false` if `order_by` is empty (no ordering at all), or if any key names a column
/// the table does not have — an unresolvable key cannot be shown to order anything, and the
/// safe answer is the one that leads to sorting rather than to trusting the engine's order.
pub fn orders_rows_totally(order_by: &[OrderKey], table: &Table, rows: &[Vec<Literal>]) -> bool {
    if order_by.is_empty() {
        return false;
    }
    if rows.len() < 2 {
        return true;
    }

    let Some(positions) = key_positions(order_by, table) else {
        return false;
    };

    // Compare every pair. Quadratic, on at most a handful of rows, and it avoids requiring
    // `Hash` or an ordering on `Literal` — `NULL` has no natural place in either.
    for (index, row) in rows.iter().enumerate() {
        for other in &rows[index + 1..] {
            if positions
                .iter()
                .all(|&position| row.get(position) == other.get(position))
            {
                // Two rows agree on every ordering column, so their relative order is
                // unspecified. Note this counts two `NULL`s as tied, which is right for
                // *ordering*: `NULLS FIRST` puts them in the same group, and nothing says
                // which comes first within it.
                return false;
            }
        }
    }

    true
}

/// Where each ordering key's column sits in a row, or `None` if any key does not resolve.
fn key_positions(order_by: &[OrderKey], table: &Table) -> Option<Vec<usize>> {
    order_by
        .iter()
        .map(|key| position_of(&key.column, table))
        .collect()
}

fn position_of(reference: &ColumnRef, table: &Table) -> Option<usize> {
    if reference.table != table.name {
        return None;
    }
    table.column(&reference.column).map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Column, Direction, SqlType};

    fn table() -> Table {
        Table {
            name: "t0".to_string(),
            columns: vec![
                Column {
                    name: "c0".to_string(),
                    sql_type: SqlType::Integer,
                },
                Column {
                    name: "c1".to_string(),
                    sql_type: SqlType::Integer,
                },
            ],
        }
    }

    fn key(column: &str) -> OrderKey {
        OrderKey {
            column: ColumnRef {
                table: "t0".to_string(),
                column: column.to_string(),
            },
            direction: Direction::Ascending,
            nulls_first: true,
        }
    }

    fn rows(values: &[[i64; 2]]) -> Vec<Vec<Literal>> {
        values
            .iter()
            .map(|pair| pair.iter().map(|n| Literal::Integer(*n)).collect())
            .collect()
    }

    #[test]
    fn distinct_values_order_totally() {
        assert!(orders_rows_totally(
            &[key("c0")],
            &table(),
            &rows(&[[1, 9], [2, 9], [3, 9]])
        ));
    }

    #[test]
    fn a_tie_in_the_ordering_column_means_the_order_is_not_total() {
        // The case the naive "has an ORDER BY, therefore ordered" rule gets wrong. Rows 0
        // and 1 both have c0 = 1, so either may come first, on either engine.
        assert!(!orders_rows_totally(
            &[key("c0")],
            &table(),
            &rows(&[[1, 7], [1, 8], [3, 9]])
        ));
    }

    #[test]
    fn a_second_key_can_break_the_tie() {
        assert!(orders_rows_totally(
            &[key("c0"), key("c1")],
            &table(),
            &rows(&[[1, 7], [1, 8], [3, 9]])
        ));
    }

    #[test]
    fn two_nulls_are_a_tie() {
        // `NULLS FIRST` puts both in the same group and says nothing about their order
        // within it. Treating `NULL` as distinct from itself here would mark an order
        // total when it is not — and the consequence would be comparing row order that no
        // engine promised.
        let rows = vec![
            vec![Literal::Null, Literal::Integer(1)],
            vec![Literal::Null, Literal::Integer(2)],
        ];
        assert!(!orders_rows_totally(&[key("c0")], &table(), &rows));
    }

    #[test]
    fn no_ordering_at_all_is_not_a_total_order() {
        assert!(!orders_rows_totally(
            &[],
            &table(),
            &rows(&[[1, 1], [2, 2]])
        ));
    }

    #[test]
    fn fewer_than_two_rows_are_always_totally_ordered() {
        assert!(orders_rows_totally(&[key("c0")], &table(), &[]));
        assert!(orders_rows_totally(
            &[key("c0")],
            &table(),
            &rows(&[[1, 1]])
        ));
    }

    #[test]
    fn an_unresolvable_key_is_treated_as_not_ordering() {
        // The safe direction: if we cannot show the order is total, sort before comparing.
        // Being wrong this way costs a hidden ordering bug in a case that should not exist;
        // being wrong the other way invents divergences in cases that do.
        assert!(!orders_rows_totally(
            &[key("nope")],
            &table(),
            &rows(&[[1, 1], [2, 2]])
        ));
    }
}
