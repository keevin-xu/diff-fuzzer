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

use crate::ast::SqlCase;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Where a negative came from, which is a proxy for how hard it is to satisfy.
///
/// Ordered by discriminating power, strongest first.
// `Ord` follows the declaration order below, which is deliberately *descending order of
// how much a negative discriminates*. Sorting a breakdown therefore puts the hardest
// negatives first, where a reader looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
    pub case: SqlCase,
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

    /// **The generator description, verbatim.** [`Provenance`] is a readable *class*;
    /// this is the exact identity, and matching is done on this rather than on the class.
    ///
    /// The distinction is not pedantry. Widening the fuzzer's decode bounds produces a
    /// genuinely different distribution that still reports "decoded from fuzzer bytes" —
    /// so both old and new negatives classify as [`Provenance::Fuzzer`] while being drawn
    /// from different regions of the input space. Comparing the raw strings catches that;
    /// comparing the class does not, and would silently reintroduce the very confound the
    /// pool exists to prevent.
    #[serde(default)]
    pub generator: String,

    /// **Which implementations the non-divergence was observed on**, sorted.
    ///
    /// A negative is the claim "these implementations agreed on this case". Change the
    /// implementations and the claim is simply about something else — after the flex swap,
    /// 810 of 814 recorded *findings* stopped reproducing, and negatives are no more
    /// durable than findings. Empty means unrecorded, which is never comparable.
    #[serde(default)]
    pub backends: Vec<String>,
}

/// How the fuzz target describes its generator, **including the decode bounds**.
///
/// The bounds are named because the pool matches on this string verbatim. Widening
/// `DECODE_BOUNDS` without changing this would let negatives from the old, narrower
/// distribution be scored against findings from the new one — indistinguishable, since both
/// would read simply as "decoded from fuzzer bytes".
///
/// `decode::bounds_are_named_in_the_generator_description` fails if the two drift apart.
pub const FUZZER_GENERATOR: &str =
    "decoded from fuzzer bytes at max_dim 64, magnitude 10, budget 1048576";

/// The conditions a set of cases was observed under.
///
/// Carried alongside findings and negatives alike, because scoring one against the other is
/// only meaningful when both were produced the same way.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SamplingContext {
    /// The generator description, verbatim — see [`Negative::generator`].
    pub generator: String,
    /// The implementations that were run, sorted.
    pub backends: Vec<String>,
}

impl SamplingContext {
    /// Build from a generator description and the implementation names, sorting the latter
    /// so that ordering cannot make two identical contexts compare unequal.
    pub fn new(generator: impl Into<String>, backends: &[&str]) -> Self {
        let mut backends: Vec<String> = backends.iter().map(|b| (*b).to_string()).collect();
        backends.sort();
        Self {
            generator: generator.into(),
            backends,
        }
    }

    /// The readable class this context belongs to.
    pub fn provenance(&self) -> Provenance {
        Provenance::from_generator(&self.generator)
    }
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
        } else if description.contains("by hand") {
            // Without this the parser cannot return `Constructed` at all, and every
            // hand-built negative silently classifies as `Unknown` — which the pool then
            // discards, throwing away the strongest negatives available. Found by a test.
            Provenance::Constructed
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
/// **This is the stratification test**, and it is deliberately coarse: it asks only "could a
/// rule about SQL semantics possibly match this?", not "does any particular rule match". A
/// finer test would amount to choosing negatives with the very vocabulary the search uses,
/// which risks selecting cases that confirm what is already believed.
///
/// Erring toward *interesting* is the safe direction — a false positive here costs one extra
/// file, while a false negative discards exactly the evidence that discriminates.
///
/// # Rewritten for SQL, and the only function in this file that needed it
///
/// The tensor version asked about magnitudes: overflow risk, subnormals, extreme ratios.
/// None of that means anything here. What makes a SQL case discriminating is where its
/// **three-valued logic and empty cases** are — a `NULL` that a predicate or a join key
/// actually touches, a table with no rows, an aggregate with nothing to aggregate. A case of
/// ordinary integers with no `NULL`s rejects every plausible rule for free and can therefore
/// never demote a wrong one.
pub fn is_interesting(case: &SqlCase) -> bool {
    use crate::schema::Literal;

    // An empty table, or an aggregate over one: the classic edge, and cheap to spot.
    if case.queried_rows().is_empty() {
        return true;
    }

    // Any `NULL` in the data the query reads. Three-valued logic is where these engines
    // could most plausibly differ, so a case containing none is weak evidence about almost
    // any rule.
    if case
        .queried_rows()
        .iter()
        .flatten()
        .any(|value| matches!(value, Literal::Null))
    {
        return true;
    }

    // Duplicate rows, which is what every deduplicating construct is defined by.
    let rows = case.queried_rows();
    for (index, row) in rows.iter().enumerate() {
        if rows[index + 1..].contains(row) {
            return true;
        }
    }

    // A query whose shape is where the engines share least implementation.
    case.query.set_op.is_some()
        || case.query.join.is_some()
        || case.query.contains_subquery()
        || !case.query.group_by.is_empty()
}

/// Write one non-diverging case, filed under its source.
///
/// Named by a hash of the case, so a case seen twice overwrites rather than accumulating —
/// the same content-derived naming the findings use, and for the same reason: a directory
/// that grows without bound stops being readable.
pub fn save_case(
    directory: impl AsRef<Path>,
    case: &SqlCase,
    source: Source,
    context: &SamplingContext,
) -> io::Result<()> {
    let directory = directory.as_ref().join(source.label());
    std::fs::create_dir_all(&directory)?;

    let record = Negative {
        case: case.clone(),
        source,
        provenance: context.provenance(),
        generator: context.generator.clone(),
        backends: context.backends.clone(),
    };
    // The tensor version used `case.name()` — the operation, which gave a negative a
    // readable prefix. A SQL case has no single operation, so the query's clause shape plays
    // that role: `neg-where+join-<hash>.json` is as scannable as `neg-matmul-<hash>.json`.
    let path = directory.join(format!(
        "neg-{}-{:x}.json",
        crate::signature::clause_shape(case).join("+"),
        digest(case)
    ));
    std::fs::write(path, serde_json::to_string(&record)?)
}

/// Write a batch of cases sharing one source, as a single file.
///
/// For cases produced by a deliberate experiment, which are only meaningful read as a set.
pub fn save_batch(
    path: impl AsRef<Path>,
    cases: &[SqlCase],
    source: Source,
    context: &SamplingContext,
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
            provenance: context.provenance(),
            generator: context.generator.clone(),
            backends: context.backends.clone(),
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
            } else if let Ok(batch) = serde_json::from_str::<Vec<SqlCase>>(&text) {
                // A bare case records no backends, so it will fail the pool's backend
                // check — correctly: nothing says which implementations agreed on it.
                cases.extend(batch.into_iter().map(|case| Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                    generator: String::new(),
                    backends: Vec::new(),
                }));
            } else if let Ok(case) = serde_json::from_str::<SqlCase>(&text) {
                cases.push(Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                    generator: String::new(),
                    backends: Vec::new(),
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
fn digest(case: &SqlCase) -> u64 {
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
    /// The negatives were observed on different implementations than the findings.
    ///
    /// **A different axis from `DistributionMismatch`, and not covered by it.** A negative
    /// is the claim "these implementations agreed on this case"; run a different set and
    /// the claim is about something else entirely. `Constructed` is exempt from the
    /// *distribution* check but never from this one, because being hand-built says nothing
    /// about which backends were compared.
    BackendMismatch {
        positives: Vec<String>,
        negatives: Vec<String>,
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
            PoolError::BackendMismatch {
                positives,
                negatives,
            } => write!(
                f,
                "findings were observed on [{}] but every negative was observed on \
                 something else (e.g. [{}]) — a case that agreed on one set of backends \
                 says nothing about another",
                positives.join(", "),
                negatives.join(", ")
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
    /// Two independent checks, in order, because they fail for different reasons and a
    /// caller deserves to be told which:
    ///
    /// 1. **Same implementations.** No exemptions — not even for `Constructed`.
    /// 2. **Same distribution**, matched on the verbatim generator string rather than on
    ///    the provenance class, so two runs of the same *kind* of generator at different
    ///    bounds do not pass as equivalent. `Constructed` is exempt here and only here.
    pub fn matched(all: Vec<Negative>, positives: &SamplingContext) -> Result<Self, PoolError> {
        let (same_backends, wrong_backends): (Vec<Negative>, Vec<Negative>) = all
            .into_iter()
            .partition(|n| !n.backends.is_empty() && n.backends == positives.backends);

        if same_backends.is_empty() {
            return Err(PoolError::BackendMismatch {
                positives: positives.backends.clone(),
                negatives: wrong_backends
                    .first()
                    .map(|n| n.backends.clone())
                    .unwrap_or_default(),
            });
        }

        let kept: Vec<Negative> = same_backends
            .into_iter()
            .filter(|n| match n.provenance {
                // Hand-built cases belong to no distribution, so they cannot introduce a
                // distributional confound — but they already passed the backend check above.
                Provenance::Constructed => true,
                Provenance::Unknown => false,
                _ => n.generator == positives.generator,
            })
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

    /// The negatives themselves, with their source and provenance.
    ///
    /// Used by the search, which pre-extracts features once rather than re-deriving them
    /// on each of its thousands of predicate evaluations.
    pub fn negatives(&self) -> &[Negative] {
        &self.negatives
    }

    pub fn cases(&self) -> impl Iterator<Item = &SqlCase> {
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
        mut matches: impl FnMut(&SqlCase) -> bool,
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
    

    /// A case carrying `value` in its first cell — enough to make two cases distinguishable,
    /// which is all these tests need.
    fn case(value: i64) -> SqlCase {
        let mut case = SqlCase::fixed_example();
        case.data[0].rows[0][0] = crate::schema::Literal::Integer(value);
        case
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("diff-fuzzer-neg-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_saved_case_loads_back_with_its_source() {
        let dir = temp_dir("roundtrip");
        save_case(&dir, &case(1), Source::NearMiss, &fuzzer()).expect("writable");

        let loaded = load(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].source, Source::NearMiss);
        assert_eq!(format!("{:?}", loaded[0].case), format!("{:?}", case(1)));
    }

    /// Content-derived naming, so a campaign meeting the same case twice does not grow the
    /// directory. Without it a long run accumulates duplicates of whatever it sees most —
    /// which is exactly the least interesting case.
    #[test]
    fn saving_the_same_case_twice_leaves_one_file() {
        let dir = temp_dir("dedup");
        save_case(&dir, &case(2), Source::Ordinary, &fuzzer()).expect("writable");
        save_case(&dir, &case(2), Source::Ordinary, &fuzzer()).expect("writable");

        assert_eq!(load(&dir).len(), 1);
    }

    #[test]
    fn sources_are_kept_apart_on_disk_and_counted_separately() {
        let dir = temp_dir("sources");
        save_case(&dir, &case(1), Source::NearMiss, &fuzzer()).expect("writable");
        save_case(&dir, &case(2), Source::Ordinary, &fuzzer()).expect("writable");
        save_case(&dir, &case(3), Source::Ordinary, &fuzzer()).expect("writable");

        let counts = by_source(&load(&dir));
        assert_eq!(counts, vec![(Source::NearMiss, 1), (Source::Ordinary, 2)]);
    }

    /// Files written before provenance existed must still load, rather than being dropped
    /// silently — a lost negative weakens every predicate scored against the set.
    #[test]
    fn a_file_without_provenance_still_loads() {
        let dir = temp_dir("legacy");
        std::fs::create_dir_all(&dir).expect("writable");
        let bare = serde_json::to_string(&[case(1), case(2)]).expect("serialisable");
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

    /// The two backends every test in this module pretends to have run.
    fn pair() -> [&'static str; 2] {
        ["flex", "libtorch"]
    }

    fn context(generator: &str) -> SamplingContext {
        SamplingContext::new(generator, &pair())
    }

    fn fuzzer() -> SamplingContext {
        context("decoded from fuzzer bytes")
    }

    fn constructed() -> SamplingContext {
        context("built by hand")
    }

    /// A context whose generator string is unrecognised — `Provenance::Unknown`.
    fn unknown() -> SamplingContext {
        context("who knows")
    }

    fn wide() -> SamplingContext {
        context("sql-v1(tables<=2, columns<=8, rows<=32, depth<=3)")
    }

    fn negative(value: i64, source: Source, ctx: &SamplingContext) -> Negative {
        Negative {
            case: case(value),
            source,
            provenance: ctx.provenance(),
            generator: ctx.generator.clone(),
            backends: ctx.backends.clone(),
        }
    }

    /// **The leak this guard exists to prevent.** Fuzz findings and seeded negatives differ
    /// on magnitude, dimension, special-value rate and domain restriction — a rule
    /// separating those pools would score *perfectly* while describing which generator ran.
    #[test]
    fn a_pool_refuses_to_mix_distributions() {
        let seeded = vec![negative(1, Source::Ordinary, &wide())];

        let result = Pool::matched(seeded, &fuzzer());
        assert_eq!(result.unwrap_err(), PoolError::Empty);
    }

    /// Mismatched negatives are dropped, not silently tolerated — and if nothing survives
    /// the filter, the pool **fails** rather than becoming an empty set that every
    /// candidate would trivially survive.
    #[test]
    fn mismatched_negatives_are_excluded_and_matching_ones_kept() {
        let mixed = vec![
            negative(1, Source::Ordinary, &fuzzer()),
            negative(2, Source::Ordinary, &wide()),
        ];

        let pool = Pool::matched(mixed, &fuzzer()).expect("one matches");
        assert_eq!(pool.len(), 1);
    }

    /// **Hand-built cases belong to no distribution**, so they cannot introduce a
    /// distributional confound and are always safe to score against. They are also the
    /// strongest negatives available, which makes excluding them costly.
    #[test]
    fn constructed_negatives_are_comparable_with_anything() {
        let built = vec![negative(1, Source::Constructed, &constructed())];

        assert!(Pool::matched(built.clone(), &fuzzer()).is_ok());
        assert!(Pool::matched(built, &wide()).is_ok());
    }

    /// **A negative is a claim about a backend pair, not about a case.** Run different
    /// implementations and the claim is about something else — after the flex swap 810 of
    /// 814 recorded findings stopped reproducing, and negatives are no more durable.
    #[test]
    fn negatives_observed_on_other_backends_are_refused() {
        let elsewhere = vec![Negative {
            case: case(1),
            source: Source::NearMiss,
            provenance: Provenance::Fuzzer,
            generator: FUZZER_GENERATOR.to_string(),
            backends: vec!["cuda".to_string(), "libtorch".to_string()],
        }];

        let error = Pool::matched(elsewhere, &fuzzer()).unwrap_err();
        assert!(matches!(error, PoolError::BackendMismatch { .. }));
    }

    /// **Not even `Constructed` is exempt from the backend check.** Being hand-built means a
    /// case belongs to no *distribution*; it says nothing about which backends agreed on it.
    #[test]
    fn hand_built_negatives_are_still_bound_to_the_backends_they_ran_on() {
        let built = vec![Negative {
            case: case(1),
            source: Source::Constructed,
            provenance: Provenance::Constructed,
            generator: "built by hand".to_string(),
            backends: vec!["wgpu".to_string()],
        }];

        assert!(matches!(
            Pool::matched(built, &fuzzer()).unwrap_err(),
            PoolError::BackendMismatch { .. }
        ));
    }

    /// Backend order must not decide the outcome.
    #[test]
    fn backend_order_does_not_affect_matching() {
        let reversed = SamplingContext::new(FUZZER_GENERATOR, &["libtorch", "flex"]);
        assert_eq!(reversed.backends, fuzzer().backends);
    }

    /// **The collision the verbatim string exists to catch.** Both of these are
    /// `Provenance::Fuzzer`, and they are drawn from different regions of the input space.
    #[test]
    fn two_fuzzer_runs_at_different_bounds_do_not_match() {
        let old = context("decoded from fuzzer bytes at max_dim 8, magnitude 10");
        assert_eq!(old.provenance(), Provenance::Fuzzer);
        assert_eq!(fuzzer().provenance(), Provenance::Fuzzer);

        let recorded = vec![negative(1, Source::Ordinary, &old)];
        assert_eq!(
            Pool::matched(recorded, &fuzzer()).unwrap_err(),
            PoolError::Empty,
            "matching on the provenance class alone would have let these through"
        );
    }

    /// **An unrecorded origin cannot be shown to match**, and assuming it does is exactly
    /// the leak. Negatives written before provenance existed land here.
    #[test]
    fn negatives_of_unknown_origin_are_never_comparable() {
        let unknown = vec![negative(1, Source::Ordinary, &unknown())];

        assert_eq!(
            Pool::matched(unknown, &fuzzer()).unwrap_err(),
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
            negative(1, Source::NearMiss, &fuzzer()),
            negative(2, Source::Ordinary, &fuzzer()),
            negative(3, Source::Ordinary, &fuzzer()),
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
        let pool = Pool::unchecked(vec![negative(1, Source::NearMiss, &fuzzer())]);

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
        assert_eq!(
            Provenance::from_generator("built by hand: batched_probe"),
            Provenance::Constructed,
            "the parser must be able to return every class it claims to"
        );
        // Anything unrecognised becomes Unknown rather than a guess.
        assert_eq!(Provenance::from_generator("who knows"), Provenance::Unknown);
    }

    /// A case with ordinary data and a plain query rejects every plausible rule for free,
    /// so it can never demote a wrong one. Note `fixed_example` carries a `NULL`, so these
    /// have to remove it to be ordinary at all.
    #[test]
    fn ordinary_cases_are_not_interesting() {
        use crate::schema::Literal;

        let mut plain = SqlCase::fixed_example();
        for row in &mut plain.data[0].rows {
            for value in row.iter_mut() {
                if matches!(value, Literal::Null) {
                    *value = Literal::Integer(7);
                }
            }
        }
        // Distinct rows, no NULLs, no join/set-op/subquery/grouping.
        for (index, row) in plain.data[0].rows.iter_mut().enumerate() {
            row[0] = Literal::Integer(index as i64);
        }
        assert!(!is_interesting(&plain));
    }

    /// The cases that actually discriminate: anything a rule about **SQL semantics** could
    /// plausibly key on. Rewritten wholesale from the tensor version, which asked about
    /// infinities and subnormals — the concepts do not survive the move, though the
    /// *question* does.
    #[test]
    fn cases_with_sql_edges_are_interesting() {
        use crate::schema::Literal;

        // A NULL in the data the query reads: three-valued logic is the richest source of
        // engine disagreement in SQL.
        let mut with_null = SqlCase::fixed_example();
        with_null.data[0].rows[0][1] = Literal::Null;
        assert!(is_interesting(&with_null), "a NULL in the data");

        // An empty table — the aggregate-over-nothing edge.
        let mut empty = SqlCase::fixed_example();
        empty.data[0].rows.clear();
        assert!(is_interesting(&empty), "an empty table");

        // Duplicate rows, which is what every deduplicating construct is defined by.
        let mut duplicated = SqlCase::fixed_example();
        let first = duplicated.data[0].rows[0].clone();
        duplicated.data[0].rows[1] = first;
        assert!(is_interesting(&duplicated), "duplicate rows");
    }

    /// A `NULL` makes a case interesting even when nothing else about it is unusual.
    #[test]
    fn a_null_alone_is_enough() {
        let spread = {
            let mut case = SqlCase::fixed_example();
            case.data[0].rows[0][1] = crate::schema::Literal::Null;
            case
        };
        assert!(is_interesting(&spread));
    }

    /// Erring toward *interesting* is the safe direction — a false positive costs one file,
    /// a false negative discards the evidence that discriminates. This pins the intent so a
    /// future tightening of the thresholds has to be deliberate.
    #[test]
    fn the_test_errs_toward_keeping() {
        // Borderline for SQL: nothing extreme, but two rows share a value — which every
        // deduplicating construct is defined by, so it is worth keeping.
        let borderline = {
            let mut case = SqlCase::fixed_example();
            let first = case.data[0].rows[0].clone();
            case.data[0].rows[1] = first;
            case
        };
        assert!(is_interesting(&borderline));
    }
}
