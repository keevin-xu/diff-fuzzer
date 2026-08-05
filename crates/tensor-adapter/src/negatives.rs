//! Cases that were judged and **agreed** — the counter-examples a trigger claim must survive.
//!
//! # Why a divergence-finding tool stores non-divergences
//!
//! A claim about *what triggers* a bug is only worth something if it separates cases that
//! diverge from cases that do not. Fitted to divergences alone, any rule can be found:
//! with 41 findings that all contain an overflowing product, "contains an overflowing
//! product" looks like an explanation — until you notice it also holds for cases that
//! agree perfectly. That is exactly what happened here, and only the non-diverging cases
//! exposed it.
//!
//! So negatives are not incidental. **They are the half of the evidence that makes the
//! other half falsifiable.**
//!
//! # Why they must be captured *during* a run
//!
//! They cannot be reconstructed afterwards. A finding records a seed, but a fuzzing
//! finding's seed is meaningless — libFuzzer's stream depends on a corpus that evolves as
//! it runs, and under `-fork=1` on child processes that no longer exist. Sampling as the
//! campaign runs is the only way to get negatives drawn from **the same distribution as
//! the findings**, which matters more than it sounds: scored against negatives from a
//! different generator, a search would happily learn *"which generator produced this
//! case"* instead of *"what triggers the bug"* — and would score well doing it.
//!
//! # What does *not* count
//!
//! Only a case the oracle judged **`Agree`**. A `Skipped` case — a backend refused it, or
//! both returned `NaN` so no arithmetic was compared — is not evidence that the case fails
//! to diverge. It is evidence that nothing was learned, and recording it as a negative
//! would quietly poison the set with cases that were never actually tested.
//!
//! # Not all negatives are worth the same, which is why each records its source
//!
//! **A negative only tests a rule that could plausibly have matched it.** A case with
//! ordinary magnitudes and no special values fails `overflow_product AND mixed_sign`
//! trivially — it was never going to challenge that rule, so keeping it proves nothing.
//! Sample the stream uniformly and almost every negative is of that kind.
//!
//! The consequence is concrete: the search ranks candidates by *fewest negatives matched*
//! first. If no candidate matches any negative, they all tie, the ranking falls through to
//! "covers the most findings", and that is overfitting by another name. `overflow_product`
//! alone matches all 41 findings and every boring negative rejects it for free — **yet it
//! is wrong**, and only the hand-built near-misses demote it.
//!
//! So [`Source`] is recorded per case. "Survived 12 near-misses" and "survived 500
//! ordinary cases" are wildly different claims, and without provenance they look identical
//! in a report.

use crate::input::TensorOp;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Where a negative came from, which is a proxy for how hard it is to satisfy.
///
/// Ordered by discriminating power, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// Rejected by the shrinker while minimising a real finding — **one edit away from a
    /// case that diverges**, and therefore the closest counter-example obtainable. Free:
    /// the shrinker already generates and discards these.
    NearMiss,
    /// Built by hand to probe a specific hypothesis, like the batched-matmul cases that
    /// falsified `overflow AND mixed_sign`.
    Constructed,
    /// Sampled from a campaign, and carrying something a rule might plausibly key on —
    /// an overflowing product, a special value, an extreme magnitude ratio.
    Interesting,
    /// Sampled from a campaign with nothing notable in it. Cheap to collect and weak
    /// evidence; kept in small numbers so a rule can be checked against the ordinary case
    /// as well as the hard one.
    Ordinary,
}

impl Source {
    /// A short, stable name — used as a directory, so it must not contain separators.
    pub fn label(self) -> &'static str {
        match self {
            Source::NearMiss => "near-miss",
            Source::Constructed => "constructed",
            Source::Interesting => "interesting",
            Source::Ordinary => "ordinary",
        }
    }
}

/// A non-diverging case together with where it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Negative {
    pub case: TensorOp,
    pub source: Source,
    /// **Which generator produced it** — not the same question as [`Source`].
    ///
    /// `Source` says how hard the negative is to satisfy. This says which *distribution* it
    /// was drawn from, and the two are independent: a near-miss from a fuzzing run and a
    /// near-miss from a seeded campaign are equally hard and come from entirely different
    /// input distributions.
    ///
    /// **The search needs this to avoid learning the wrong thing.** Score fuzz-derived
    /// findings against seeded negatives and the two pools differ on four axes — magnitude
    /// 10 versus 1000, dimension 8 versus 64, special-value rate 0 versus 0.125, domains
    /// unrestricted versus restricted. A rule separating those pools would score *perfectly*
    /// while describing which generator ran, not what triggers a bug.
    ///
    /// Defaults to [`Provenance::Unknown`] when reading files written before this existed —
    /// which is honest: their provenance genuinely is unrecorded, and a search should treat
    /// them with suspicion rather than assume.
    #[serde(default)]
    pub provenance: Provenance,
}

/// Which generator a case came from.
///
/// Deliberately coarse. The question a search must answer is "were these drawn from the
/// same distribution?", and a finer taxonomy would invite false confidence about pools that
/// merely *look* similar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Provenance {
    /// Bytes mutated by libFuzzer and decoded — a corpus-biased, evolving distribution.
    Fuzzer,
    /// The seeded generator at its default bounds.
    SeededDefault,
    /// The seeded generator at wide bounds — larger shapes and magnitudes.
    SeededWide,
    /// Built by hand for an experiment. Belongs to no distribution, which is exactly why
    /// such cases are strong evidence individually and useless for distribution matching.
    Constructed,
    /// Recorded before provenance existed, or from a source that did not say.
    #[default]
    Unknown,
}

impl Provenance {
    /// A short name, used in reports.
    pub fn label(self) -> &'static str {
        match self {
            Provenance::Fuzzer => "fuzzer",
            Provenance::SeededDefault => "seeded-default",
            Provenance::SeededWide => "seeded-wide",
            Provenance::Constructed => "constructed",
            Provenance::Unknown => "unknown",
        }
    }

    /// Read from the `generator` string a `DivergenceReport` records.
    ///
    /// Parsing prose is unpleasant, and it is what the recorded field contains. Anything
    /// unrecognised becomes `Unknown` rather than a guess.
    pub fn from_generator(description: &str) -> Self {
        if description.contains("fuzzer bytes") {
            Provenance::Fuzzer
        } else if description.contains("magnitude: 1000") || description.contains("max_dim: 64") {
            Provenance::SeededWide
        } else if description.starts_with("Bounds") {
            Provenance::SeededDefault
        } else {
            Provenance::Unknown
        }
    }

    /// Whether two pools may be scored against each other.
    ///
    /// **`Constructed` matches anything**: a hand-built near-miss is not drawn from a
    /// distribution at all, so it cannot introduce a distributional confound. It is the one
    /// kind of negative that is always safe to include.
    ///
    /// **`Unknown` matches nothing.** A pool whose origin was never recorded cannot be
    /// shown to match, and assuming it does is precisely the leak this guard exists for.
    pub fn comparable_with(self, other: Self) -> bool {
        match (self, other) {
            (Provenance::Unknown, _) | (_, Provenance::Unknown) => false,
            (Provenance::Constructed, _) | (_, Provenance::Constructed) => true,
            (a, b) => a == b,
        }
    }
}

/// Whether a case carries anything a trigger rule might plausibly key on.
///
/// **This is the stratification test**, and it is deliberately coarse: it asks only "could
/// a rule about extreme floating-point behaviour possibly match this?", not "does any
/// particular rule match". A finer test would amount to choosing negatives with the very
/// vocabulary the search uses, which risks selecting cases that confirm what is already
/// believed.
///
/// Erring toward *interesting* is the safe direction — a false positive here costs one
/// extra file, while a false negative discards exactly the evidence that discriminates.
pub fn is_interesting(case: &TensorOp) -> bool {
    /// Every value a case carries, across however many operands it has.
    fn operand_values(case: &TensorOp) -> Vec<&f32> {
        let operands: Vec<&crate::input::TensorValue> = match case {
            TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => vec![arg],
            TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => vec![lhs, rhs],
        };
        operands.into_iter().flat_map(|o| o.data().iter()).collect()
    }

    const OVERFLOW_RISK: f32 = 1e18; // squares to beyond f32's range
    const EXTREME_RATIO: f32 = 1e12;

    let mut smallest_nonzero = f32::INFINITY;
    let mut largest = 0.0f32;

    for &value in operand_values(case) {
        if !value.is_finite() {
            return true; // an infinity or NaN already in the input
        }
        let magnitude = value.abs();
        if magnitude >= OVERFLOW_RISK {
            return true;
        }
        if magnitude > 0.0 {
            smallest_nonzero = smallest_nonzero.min(magnitude);
            largest = largest.max(magnitude);
        }
        if magnitude > 0.0 && magnitude < f32::MIN_POSITIVE {
            return true; // subnormal
        }
    }

    largest > 0.0 && smallest_nonzero.is_finite() && largest / smallest_nonzero > EXTREME_RATIO
}

/// Write one non-diverging case, filed under its source.
///
/// Named by a hash of the case, so a case seen twice overwrites rather than accumulating —
/// the same content-derived naming the findings use, and for the same reason: a directory
/// that grows without bound stops being readable.
pub fn save_case(
    directory: impl AsRef<Path>,
    case: &TensorOp,
    source: Source,
    provenance: Provenance,
) -> io::Result<()> {
    let directory = directory.as_ref().join(source.label());
    std::fs::create_dir_all(&directory)?;

    let record = Negative {
        case: case.clone(),
        source,
        provenance,
    };
    let path = directory.join(format!("neg-{}-{:x}.json", case.name(), digest(case)));
    std::fs::write(path, serde_json::to_string(&record)?)
}

/// Write a batch of cases sharing one source, as a single file.
///
/// For cases produced by a deliberate experiment, which are only meaningful read as a set.
pub fn save_batch(
    path: impl AsRef<Path>,
    cases: &[TensorOp],
    source: Source,
    provenance: Provenance,
) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let records: Vec<Negative> = cases
        .iter()
        .map(|case| Negative {
            case: case.clone(),
            source,
            provenance,
        })
        .collect();
    std::fs::write(path, serde_json::to_string_pretty(&records)?)
}

/// Load every negative at or below `directory`.
///
/// Four on-disk shapes are accepted, because they arrive from different places and a
/// caller should not have to care which: one record, an array of records, and — for files
/// written before provenance existed — a bare case or an array of bare cases, which are
/// read as [`Source::Constructed`].
pub fn load(directory: impl AsRef<Path>) -> Vec<Negative> {
    let mut cases = Vec::new();
    let mut pending = vec![directory.as_ref().to_path_buf()];

    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };

        for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                // Named rather than silently skipped: a lost negative weakens every
                // predicate scored against this set, and does so invisibly.
                eprintln!("could not read {}", path.display());
                continue;
            };

            if let Ok(batch) = serde_json::from_str::<Vec<Negative>>(&text) {
                cases.extend(batch);
            } else if let Ok(one) = serde_json::from_str::<Negative>(&text) {
                cases.push(one);
            } else if let Ok(batch) = serde_json::from_str::<Vec<TensorOp>>(&text) {
                cases.extend(batch.into_iter().map(|case| Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                }));
            } else if let Ok(case) = serde_json::from_str::<TensorOp>(&text) {
                cases.push(Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                });
            } else {
                eprintln!("could not parse {} as a negative", path.display());
            }
        }
    }

    cases
}

/// How many negatives of each source a set contains.
///
/// **The number a report should lead with.** "Survived 12 near-misses" and "survived 500
/// ordinary cases" are different claims, and a bare total conflates them.
pub fn by_source(negatives: &[Negative]) -> Vec<(Source, usize)> {
    [
        Source::NearMiss,
        Source::Constructed,
        Source::Interesting,
        Source::Ordinary,
    ]
    .into_iter()
    .map(|source| {
        (
            source,
            negatives.iter().filter(|n| n.source == source).count(),
        )
    })
    .filter(|(_, count)| *count > 0)
    .collect()
}

/// A stable identifier for a case, used only for naming files.
fn digest(case: &TensorOp) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{case:?}").hash(&mut hasher);
    hasher.finish()
}

/// A set of negatives a candidate rule can be scored against.
///
/// Exists so the two rules that make scoring honest cannot be forgotten: **refuse to score
/// across mismatched distributions**, and **report survival by source rather than as a
/// total**.
#[derive(Debug, Clone, Default)]
pub struct Pool {
    negatives: Vec<Negative>,
}

/// Why a pool could not be used, when it could not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// The positives and the negatives came from different generators.
    ///
    /// **Not a technicality.** The two distributions differ on magnitude, dimension,
    /// special-value rate and domain restriction, so a rule separating them scores
    /// perfectly while describing which generator ran. Declining is the correct output.
    DistributionMismatch {
        positives: Provenance,
        negatives: Vec<Provenance>,
    },
    /// Nothing to score against. A rule that survives an empty set has survived nothing.
    Empty,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::DistributionMismatch {
                positives,
                negatives,
            } => write!(
                f,
                "findings came from {} but the negatives are {} — scoring across these \
                 would learn which generator produced a case, not what triggers a bug",
                positives.label(),
                negatives
                    .iter()
                    .map(|p| p.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            PoolError::Empty => write!(
                f,
                "no negatives; a rule surviving none has survived nothing"
            ),
        }
    }
}

impl Pool {
    /// Build a pool usable against findings of a given provenance.
    ///
    /// Keeps only negatives drawn from a comparable distribution, and **fails rather than
    /// silently narrowing to nothing** — a search that quietly ends up with an empty pool
    /// would report every candidate as surviving.
    pub fn matched(all: Vec<Negative>, positives: Provenance) -> Result<Self, PoolError> {
        let kept: Vec<Negative> = all
            .into_iter()
            .filter(|n| n.provenance.comparable_with(positives))
            .collect();

        if kept.is_empty() {
            return Err(PoolError::Empty);
        }
        Ok(Self { negatives: kept })
    }

    /// Every negative, regardless of provenance. **For inspection, never for scoring.**
    pub fn unchecked(all: Vec<Negative>) -> Self {
        Self { negatives: all }
    }

    pub fn len(&self) -> usize {
        self.negatives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.negatives.is_empty()
    }

    pub fn cases(&self) -> impl Iterator<Item = &TensorOp> {
        self.negatives.iter().map(|n| &n.case)
    }

    /// How many negatives of each source a rule failed to exclude.
    ///
    /// **Returned by source, never as a total, and that is the point.** "Survived 12
    /// near-misses" and "survived 500 ordinary cases" are different claims about a rule's
    /// strength, and a single number conflates them — which is the easiest way to satisfy a
    /// gate's negative-result requirement by accident.
    pub fn matched_by_source(
        &self,
        mut matches: impl FnMut(&TensorOp) -> bool,
    ) -> Vec<(Source, usize, usize)> {
        [
            Source::NearMiss,
            Source::Constructed,
            Source::Interesting,
            Source::Ordinary,
        ]
        .into_iter()
        .filter_map(|source| {
            let of_source: Vec<&Negative> = self
                .negatives
                .iter()
                .filter(|n| n.source == source)
                .collect();
            if of_source.is_empty() {
                return None;
            }
            let hit = of_source.iter().filter(|n| matches(&n.case)).count();
            Some((source, hit, of_source.len()))
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{TensorValue, UnaryOp};

    fn case(value: f32) -> TensorOp {
        TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![2], vec![value, value]))
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("diff-fuzzer-neg-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_saved_case_loads_back_with_its_source() {
        let dir = temp_dir("roundtrip");
        save_case(&dir, &case(1.5), Source::NearMiss, Provenance::Fuzzer).expect("writable");

        let loaded = load(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].source, Source::NearMiss);
        assert_eq!(format!("{:?}", loaded[0].case), format!("{:?}", case(1.5)));
    }

    /// Content-derived naming, so a campaign meeting the same case twice does not grow the
    /// directory. Without it a long run accumulates duplicates of whatever it sees most —
    /// which is exactly the least interesting case.
    #[test]
    fn saving_the_same_case_twice_leaves_one_file() {
        let dir = temp_dir("dedup");
        save_case(&dir, &case(2.0), Source::Ordinary, Provenance::Fuzzer).expect("writable");
        save_case(&dir, &case(2.0), Source::Ordinary, Provenance::Fuzzer).expect("writable");

        assert_eq!(load(&dir).len(), 1);
    }

    #[test]
    fn sources_are_kept_apart_on_disk_and_counted_separately() {
        let dir = temp_dir("sources");
        save_case(&dir, &case(1.0), Source::NearMiss, Provenance::Fuzzer).expect("writable");
        save_case(&dir, &case(2.0), Source::Ordinary, Provenance::Fuzzer).expect("writable");
        save_case(&dir, &case(3.0), Source::Ordinary, Provenance::Fuzzer).expect("writable");

        let counts = by_source(&load(&dir));
        assert_eq!(counts, vec![(Source::NearMiss, 1), (Source::Ordinary, 2)]);
    }

    /// Files written before provenance existed must still load, rather than being dropped
    /// silently — a lost negative weakens every predicate scored against the set.
    #[test]
    fn a_file_without_provenance_still_loads() {
        let dir = temp_dir("legacy");
        std::fs::create_dir_all(&dir).expect("writable");
        let bare = serde_json::to_string(&[case(1.0), case(2.0)]).expect("serialisable");
        std::fs::write(dir.join("old.json"), bare).expect("writable");

        let loaded = load(&dir);
        assert_eq!(loaded.len(), 2);
        assert!(loaded.iter().all(|n| n.source == Source::Constructed));
    }

    #[test]
    fn loading_a_missing_directory_yields_nothing_rather_than_failing() {
        assert!(load(temp_dir("absent")).is_empty());
    }

    // --- the stratification test ---------------------------------------------------

    // --- the pool: distribution matching and survival reporting ---------------------

    fn negative(value: f32, source: Source, provenance: Provenance) -> Negative {
        Negative {
            case: case(value),
            source,
            provenance,
        }
    }

    /// **The leak this guard exists to prevent.** Fuzz findings and seeded negatives differ
    /// on magnitude, dimension, special-value rate and domain restriction — a rule
    /// separating those pools would score *perfectly* while describing which generator ran.
    #[test]
    fn a_pool_refuses_to_mix_distributions() {
        let seeded = vec![negative(1.0, Source::Ordinary, Provenance::SeededWide)];

        let result = Pool::matched(seeded, Provenance::Fuzzer);
        assert_eq!(result.unwrap_err(), PoolError::Empty);
    }

    /// Mismatched negatives are dropped, not silently tolerated — and if nothing survives
    /// the filter, the pool **fails** rather than becoming an empty set that every
    /// candidate would trivially survive.
    #[test]
    fn mismatched_negatives_are_excluded_and_matching_ones_kept() {
        let mixed = vec![
            negative(1.0, Source::Ordinary, Provenance::Fuzzer),
            negative(2.0, Source::Ordinary, Provenance::SeededWide),
        ];

        let pool = Pool::matched(mixed, Provenance::Fuzzer).expect("one matches");
        assert_eq!(pool.len(), 1);
    }

    /// **Hand-built cases belong to no distribution**, so they cannot introduce a
    /// distributional confound and are always safe to score against. They are also the
    /// strongest negatives available, which makes excluding them costly.
    #[test]
    fn constructed_negatives_are_comparable_with_anything() {
        let built = vec![negative(1.0, Source::Constructed, Provenance::Constructed)];

        assert!(Pool::matched(built.clone(), Provenance::Fuzzer).is_ok());
        assert!(Pool::matched(built, Provenance::SeededWide).is_ok());
    }

    /// **An unrecorded origin cannot be shown to match**, and assuming it does is exactly
    /// the leak. Negatives written before provenance existed land here.
    #[test]
    fn negatives_of_unknown_origin_are_never_comparable() {
        let unknown = vec![negative(1.0, Source::Ordinary, Provenance::Unknown)];

        assert_eq!(
            Pool::matched(unknown, Provenance::Fuzzer).unwrap_err(),
            PoolError::Empty
        );
        assert!(!Provenance::Unknown.comparable_with(Provenance::Unknown));
    }

    /// **Survival is reported per source, never as a total.** "Survived 12 near-misses" and
    /// "survived 500 ordinary cases" are different claims about a rule's strength, and one
    /// number conflates them.
    #[test]
    fn survival_is_reported_by_source_rather_than_as_a_total() {
        let pool = Pool::unchecked(vec![
            negative(1.0, Source::NearMiss, Provenance::Fuzzer),
            negative(2.0, Source::Ordinary, Provenance::Fuzzer),
            negative(3.0, Source::Ordinary, Provenance::Fuzzer),
        ]);

        // A rule that fires on everything: the breakdown must show both sources separately.
        let breakdown = pool.matched_by_source(|_| true);

        assert_eq!(
            breakdown,
            vec![(Source::NearMiss, 1, 1), (Source::Ordinary, 2, 2)]
        );
    }

    /// A source with no members is omitted rather than reported as `0 of 0`, which reads as
    /// a rule having survived something it was never tested against.
    #[test]
    fn a_source_with_no_negatives_is_omitted_from_the_breakdown() {
        let pool = Pool::unchecked(vec![negative(1.0, Source::NearMiss, Provenance::Fuzzer)]);

        let breakdown = pool.matched_by_source(|_| false);
        assert_eq!(breakdown, vec![(Source::NearMiss, 0, 1)]);
    }

    /// Provenance is read from the string a report actually records.
    #[test]
    fn provenance_is_recognised_from_a_recorded_generator_description() {
        assert_eq!(
            Provenance::from_generator("decoded from fuzzer bytes"),
            Provenance::Fuzzer
        );
        assert_eq!(
            Provenance::from_generator("Bounds { max_rank: 3, max_dim: 64, magnitude: 1000.0 }"),
            Provenance::SeededWide
        );
        assert_eq!(
            Provenance::from_generator("Bounds { max_rank: 4, max_dim: 8, magnitude: 10.0 }"),
            Provenance::SeededDefault
        );
        // Anything unrecognised becomes Unknown rather than a guess.
        assert_eq!(Provenance::from_generator("who knows"), Provenance::Unknown);
    }

    #[test]
    fn ordinary_values_are_not_interesting() {
        assert!(!is_interesting(&case(1.0)));
        assert!(!is_interesting(&case(-42.5)));
        assert!(!is_interesting(&case(0.0)));
    }

    /// The cases that actually discriminate: anything a rule about extreme floating-point
    /// behaviour could plausibly key on.
    #[test]
    fn extreme_values_are_interesting() {
        assert!(is_interesting(&case(1e30)), "overflows when squared");
        assert!(is_interesting(&case(-1e30)));
        assert!(is_interesting(&case(f32::INFINITY)));
        assert!(is_interesting(&case(f32::NAN)));
        assert!(is_interesting(&case(1e-45)), "subnormal");
    }

    /// A wide spread between the largest and smallest magnitude is where cancellation and
    /// accumulation-order effects live, even when no single value is extreme.
    #[test]
    fn an_extreme_magnitude_ratio_is_interesting() {
        let spread = TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![2], vec![1e15, 1e-3]));
        assert!(is_interesting(&spread));
    }

    /// Erring toward *interesting* is the safe direction — a false positive costs one file,
    /// a false negative discards the evidence that discriminates. This pins the intent so a
    /// future tightening of the thresholds has to be deliberate.
    #[test]
    fn the_test_errs_toward_keeping() {
        let borderline = TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![2], vec![1e18, 1.0]));
        assert!(is_interesting(&borderline));
    }
}
