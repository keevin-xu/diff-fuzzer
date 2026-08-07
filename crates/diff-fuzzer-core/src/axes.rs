//! What a generator was configured to produce, as named on/off axes.
//!
//! # The problem this exists to solve
//!
//! A generator has settings, and **the settings decide what a campaign can possibly find**.
//! Two failures follow from that, and both have happened here:
//!
//! **A configuration whose answers are known crowds out the rest.** A six-hour tensor
//! campaign produced 1,834 findings, every one of them the same `max`/`min` ordering
//! disagreement, while `softmax`, `log` and `mean` produced nothing at all. The easily-reached
//! class saturated the corpus. The SQL adapter reached the same conclusion independently and
//! wrote it down: *a campaign's configuration excludes the axes whose answers are known.*
//!
//! **A configuration that changes without its description changing** makes two incomparable
//! runs look comparable. Negatives are scoped to how they were produced — a case that agreed
//! under one setting says nothing about another — so a stale description silently reinstates
//! the exact confound `Pool::matched` exists to prevent.
//!
//! # What is general here, and what is not
//!
//! **The fields are not general.** `max_rank` and `max_tables` have nothing in common, and no
//! trait should pretend otherwise. What generalises is narrower:
//!
//! 1. a configuration is a set of **named axes** that are on or off;
//! 2. it has a **description naming every axis**, so two settings are distinguishable;
//! 3. corpora and negatives are **scoped to that description**.
//!
//! This trait is those three things and nothing else. Each domain lists its own axes; the
//! description is derived here so it cannot drift from them.
//!
//! **A fourth thing, added when the second domain adopted it.** The derived description
//! catches drift in *declared configuration* and is blind to drift in *generation logic* — a
//! change to how a construct is chosen, touching no axis and no bound. See
//! [`GenerationAxes::logic_version`], and note that the example which motivated this whole
//! module is one the first version failed to catch.
//!
//! # The rule that is easiest to get wrong
//!
//! **Enabling an axis must add cases, never remove them.** The SQL adapter learned this by
//! enabling joins and finding that a run reporting clean agreement had quietly stopped
//! testing ordering — joined queries are unordered, so every query became one. An axis that
//! displaces rather than adds turns a widened campaign into a narrower one while looking like
//! progress.
//!
//! # Why this is in the engine at all
//!
//! It was implemented twice, differently, before being lifted here: SQL had boolean axes,
//! named presets and a test that its description names every axis; tensors had none of it and
//! a hand-written description string. Lifting it is what stops the two drifting, and gives
//! the missing test to both.
//!
//! It is a real claim about domains — that generation is configurable by named axes — and it
//! is **untested against a third domain**. Smaller than the claim the five seams already
//! make, but not free.

/// A generator configuration, described by the axes it enables.
///
/// Implementors list their axes; everything else is derived, so a configuration cannot change
/// while its identity stays the same.
pub trait GenerationAxes {
    /// Every axis this configuration can toggle, and whether it is on.
    ///
    /// **List every axis, including the disabled ones.** A description that mentions only
    /// what is enabled cannot distinguish "this axis is off" from "this axis did not exist
    /// yet", and the second is what makes an old corpus silently incomparable.
    ///
    /// Order must be stable across calls — return them in a fixed order rather than from a
    /// hash map — because the derived description is compared verbatim.
    fn axes(&self) -> Vec<(&'static str, bool)>;

    /// An identity for the generation **logic**, when the domain has one.
    ///
    /// # The gap this closes, found by the second domain adopting the trait
    ///
    /// [`Self::description`] catches drift in *declared configuration* — an axis flipped, a
    /// bound changed. It cannot catch drift in *generation logic*, and the trait's own
    /// motivating example is one it misses.
    ///
    /// The SQL adapter's joins-versus-ordering fix made joins probabilistic at 60% rather
    /// than unconditional. That materially changed the distribution and touched **no axis and
    /// no scalar**, so the derived description before and after is byte-identical. Two runs
    /// either side of it would have been treated as comparable, which is exactly the confound
    /// this module claims to prevent.
    ///
    /// So there are two kinds of drift and they need two instruments:
    ///
    /// | drift in | caught by |
    /// |---|---|
    /// | declared configuration | the derived description |
    /// | generation logic | this hook |
    ///
    /// **`None` by default, and that default is a claim worth making deliberately:** a domain
    /// returning `None` is saying its generation logic never changes in ways that matter, or
    /// that it accepts old corpora being silently reinterpreted. Neither is usually true for
    /// long. A hash of the generator's own source is the cheap implementation.
    fn logic_version(&self) -> Option<String> {
        None
    }

    /// Bounds that are not on/off, rendered for the description.
    ///
    /// Magnitudes, dimensions, row counts: the numbers that change what is generated without
    /// being a construct that is present or absent. Empty by default, because many
    /// configurations have none.
    fn scalars(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// A stable identity for this configuration.
    ///
    /// **Derived rather than written**, which is the point: a hand-maintained description is
    /// one edit away from describing a configuration that no longer exists, and nothing fails
    /// when it does. Every axis appears, enabled or not.
    fn description(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (name, on) in self.axes() {
            parts.push(format!("{name}={}", if on { "on" } else { "off" }));
        }
        for (name, value) in self.scalars() {
            parts.push(format!("{name}={value}"));
        }
        // Last, so a domain without one produces exactly what it did before this hook existed.
        if let Some(version) = self.logic_version() {
            parts.push(format!("logic={version}"));
        }
        parts.join(" ")
    }

    /// Whether cases produced under these two configurations may be compared.
    ///
    /// Verbatim equality of the description, deliberately. A looser rule — "same axes
    /// enabled, ignore the numbers" — would let a run at one magnitude be scored against a
    /// run at another, which is the distributional confound in a different coat.
    fn comparable_with(&self, other: &dyn GenerationAxes) -> bool {
        self.description() == other.description()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Config {
        joins: bool,
        subqueries: bool,
        max_rows: usize,
    }

    impl GenerationAxes for Config {
        fn axes(&self) -> Vec<(&'static str, bool)> {
            vec![("joins", self.joins), ("subqueries", self.subqueries)]
        }
        fn scalars(&self) -> Vec<(&'static str, String)> {
            vec![("max_rows", self.max_rows.to_string())]
        }
    }

    fn config() -> Config {
        Config {
            joins: true,
            subqueries: false,
            max_rows: 8,
        }
    }

    #[test]
    fn the_description_names_every_axis_including_the_disabled_ones() {
        let described = config().description();

        assert!(described.contains("joins=on"), "{described}");
        assert!(
            described.contains("subqueries=off"),
            "a disabled axis must still be named, or it cannot be told apart from an axis \
             that did not exist: {described}"
        );
        assert!(described.contains("max_rows=8"), "{described}");
    }

    /// **The failure this trait exists to prevent.** A configuration that changes while its
    /// description does not makes two incomparable runs look comparable — and negatives are
    /// scoped by exactly that description.
    #[test]
    fn changing_any_axis_changes_the_description() {
        let base = config();

        let flipped = Config {
            subqueries: true,
            ..config()
        };
        assert_ne!(base.description(), flipped.description());

        let rescaled = Config {
            max_rows: 4096,
            ..config()
        };
        assert_ne!(
            base.description(),
            rescaled.description(),
            "a scalar bound changes what is generated and must change the identity too"
        );
    }

    #[test]
    fn identical_configurations_are_comparable_and_different_ones_are_not() {
        assert!(config().comparable_with(&config()));

        let other = Config {
            joins: false,
            ..config()
        };
        assert!(!config().comparable_with(&other));
    }

    /// The description is compared verbatim, so its order must not wobble between calls.
    struct Versioned {
        logic: &'static str,
    }

    impl GenerationAxes for Versioned {
        fn axes(&self) -> Vec<(&'static str, bool)> {
            vec![("joins", true)]
        }
        fn logic_version(&self) -> Option<String> {
            Some(self.logic.to_string())
        }
    }

    /// **The gap the second domain found.** Identical axes and scalars, different generation
    /// logic: without this hook the two are indistinguishable, and a corpus from one would be
    /// silently reused against the other.
    #[test]
    fn a_logic_change_alone_changes_the_description() {
        let before = Versioned { logic: "a1b2c3d4" };
        let after = Versioned { logic: "9f8e7d6c" };

        assert_eq!(
            before.axes(),
            after.axes(),
            "premise: the declared config is identical"
        );
        assert_ne!(before.description(), after.description());
        assert!(!before.comparable_with(&after));
    }

    /// A domain without a logic version produces exactly what it did before the hook existed,
    /// so adopting it is not a breaking change for the domain that does not need it.
    #[test]
    fn omitting_a_logic_version_leaves_the_description_unchanged() {
        assert!(!config().description().contains("logic="));
    }

    #[test]
    fn the_description_is_stable_across_calls() {
        let c = config();
        assert_eq!(c.description(), c.description());
    }
}
