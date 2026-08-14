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

use crate::case::OnnxCase;
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
    /// Built by hand to probe a specific hypothesis, like the signed-zero cases that
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
    pub case: OnnxCase,
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
    pub runtimes: Vec<String>,
}

// **The fuzz-target generator description is not carried over.** The tensor domain derives it
// from a `decode` module that exists only behind its `fuzzing` feature; this adapter has no
// cargo-fuzz target and no decoder, so the function would name a thing that does not exist.
// Callers here build a `SamplingContext` from `Bounds::description()`, which is the same string
// the campaign logs record — and reading it off a finding is the only way to be sure of the
// configuration that actually ran rather than the one currently compiled.

/// The conditions a set of cases was observed under.
///
/// Carried alongside findings and negatives alike, because scoring one against the other is
/// only meaningful when both were produced the same way.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SamplingContext {
    /// The generator description, verbatim — see [`Negative::generator`].
    pub generator: String,
    /// The implementations that were run, sorted.
    pub runtimes: Vec<String>,
}

impl SamplingContext {
    /// Build from a generator description and the implementation names, sorting the latter
    /// so that ordering cannot make two identical contexts compare unequal.
    pub fn new(generator: impl Into<String>, runtimes: &[&str]) -> Self {
        let mut runtimes: Vec<String> = runtimes.iter().map(|b| (*b).to_string()).collect();
        runtimes.sort();
        Self {
            generator: generator.into(),
            runtimes,
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
    ///
    /// **Unreachable in this domain**, which has no cargo-fuzz target and no decoder. Kept so the
    /// enum matches the other adapters' and a shared record stays readable across domains, rather
    /// than deleted and re-added when a fuzz target arrives.
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
        if description.contains("by hand") {
            // Without this the parser cannot return `Constructed` at all, and every hand-built
            // negative silently classifies as `Unknown` — which the pool then discards, throwing
            // away the strongest negatives available. Found by a test in the tensor domain, and
            // the same test earns its keep here.
            Provenance::Constructed
        } else if description.contains("float-elementwise=") {
            // A real `Bounds::description()`. The quantized axis is the one that changes the
            // distribution enough to matter: it admits `int8`/`uint8`, four operators nothing
            // else reaches, and a rounding surface with no analogue in the rest of the corpus.
            if description.contains("quantized=on") {
                Provenance::SeededWide
            } else {
                Provenance::SeededDefault
            }
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
/// **This is the stratification test**, and it is deliberately coarse: it asks only "could a rule
/// about awkward inputs possibly match this?", not "does any particular rule match". A finer test
/// would amount to choosing negatives with the very vocabulary the search uses, which risks
/// selecting cases that confirm what is already believed.
///
/// Erring toward *interesting* is the safe direction — a false positive here costs one extra
/// file, while a false negative discards exactly the evidence that discriminates.
///
/// **The overlap with `features.rs` is real and worth naming.** Both look at special values,
/// because those are what this domain's problems are made of. The guard is that this test is
/// value-only and unconditional: it never consults shapes, operators or attributes, so the
/// negatives it keeps are not filtered by the structural half of the vocabulary the search will
/// use against them.
pub fn is_interesting(case: &OnnxCase) -> bool {
    use crate::case::TensorData;

    for input in &case.inputs {
        match &input.data {
            TensorData::F32(values) => {
                for v in values {
                    if !v.is_finite() || v.is_subnormal() {
                        return true;
                    }
                    // Signed zero, on the bit pattern: `-0.0 == 0.0` is true, so a comparison
                    // would answer "no" for the one value this most needs to catch.
                    if v.to_bits() == (-0.0f32).to_bits() {
                        return true;
                    }
                    if *v == f32::MAX || *v == f32::MIN {
                        return true;
                    }
                }
            }
            TensorData::F64(values) => {
                for v in values {
                    if !v.is_finite() || v.is_subnormal() {
                        return true;
                    }
                    if v.to_bits() == (-0.0f64).to_bits() {
                        return true;
                    }
                    if *v == f64::MAX || *v == f64::MIN {
                        return true;
                    }
                }
            }
            // Integers have no special values, but they do have boundaries, and those are where
            // wrapping and saturation part company — `int32::MIN / -1` lives exactly there.
            TensorData::I32(values) => {
                if values.iter().any(|v| *v == i32::MIN || *v == i32::MAX) {
                    return true;
                }
            }
            TensorData::I64(values) => {
                if values.iter().any(|v| *v == i64::MIN || *v == i64::MAX) {
                    return true;
                }
            }
            TensorData::I8(values) => {
                if values.iter().any(|v| *v == i8::MIN || *v == i8::MAX) {
                    return true;
                }
            }
            TensorData::U8(values) => {
                if values.iter().any(|v| *v == u8::MIN || *v == u8::MAX) {
                    return true;
                }
            }
            TensorData::Bool(_) => {}
        }
    }
    false
}

/// Write one non-diverging case, filed under its source.
///
/// Named by a hash of the case, so a case seen twice overwrites rather than accumulating —
/// the same content-derived naming the findings use, and for the same reason: a directory
/// that grows without bound stops being readable.
pub fn save_case(
    directory: impl AsRef<Path>,
    case: &OnnxCase,
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
        runtimes: context.runtimes.clone(),
    };
    // The tensor domain's `case.name()` was an operation name; here that is `op.onnx_name()`,
    // lowercased so a directory listing sorts predictably.
    let path = directory.join(format!(
        "neg-{}-{:x}.json",
        case.op.onnx_name().to_lowercase(),
        digest(case)
    ));
    std::fs::write(path, serde_json::to_string(&record)?)
}

/// Write a batch of cases sharing one source, as a single file.
///
/// For cases produced by a deliberate experiment, which are only meaningful read as a set.
pub fn save_batch(
    path: impl AsRef<Path>,
    cases: &[OnnxCase],
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
            runtimes: context.runtimes.clone(),
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
            } else if let Ok(batch) = serde_json::from_str::<Vec<OnnxCase>>(&text) {
                // A bare case records no runtimes, so it will fail the pool's backend
                // check — correctly: nothing says which implementations agreed on it.
                cases.extend(batch.into_iter().map(|case| Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                    generator: String::new(),
                    runtimes: Vec::new(),
                }));
            } else if let Ok(case) = serde_json::from_str::<OnnxCase>(&text) {
                cases.push(Negative {
                    case,
                    source: Source::Constructed,
                    provenance: Provenance::Constructed,
                    generator: String::new(),
                    runtimes: Vec::new(),
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
fn digest(case: &OnnxCase) -> u64 {
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
    /// about which runtimes were compared.
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
                 something else (e.g. [{}]) — a case that agreed on one set of runtimes \
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
        let (same_runtimes, wrong_runtimes): (Vec<Negative>, Vec<Negative>) = all
            .into_iter()
            .partition(|n| !n.runtimes.is_empty() && n.runtimes == positives.runtimes);

        if same_runtimes.is_empty() {
            return Err(PoolError::BackendMismatch {
                positives: positives.runtimes.clone(),
                negatives: wrong_runtimes
                    .first()
                    .map(|n| n.runtimes.clone())
                    .unwrap_or_default(),
            });
        }

        let kept: Vec<Negative> = same_runtimes
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

    pub fn cases(&self) -> impl Iterator<Item = &OnnxCase> {
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
        mut matches: impl FnMut(&OnnxCase) -> bool,
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
    use crate::case::{OpKind, TensorData, TensorValue};

    const OPSET: i64 = 22;

    fn case(data: TensorData) -> OnnxCase {
        let n = data.len() as i64;
        OnnxCase::new(
            OpKind::Abs,
            OPSET,
            vec![TensorValue::new("a", vec![n], data)],
        )
    }

    fn context() -> SamplingContext {
        // A real `Bounds::description()` prefix, not an invented string: the parser keys on it,
        // and a fixture that does not look like the real thing tests the fixture.
        SamplingContext::new(
            "float-elementwise=on comparisons=on quantized=off special-values=on logic=3b64999b",
            &["tract", "onnxruntime"],
        )
    }

    fn negative(source: Source, context: &SamplingContext) -> Negative {
        Negative {
            case: case(TensorData::F32(vec![1.0, 2.0])),
            source,
            provenance: context.provenance(),
            generator: context.generator.clone(),
            runtimes: context.runtimes.clone(),
        }
    }

    /// **An empty negative set is an error, not an empty set** (N11.4).
    ///
    /// A rule scored against nothing matches no negatives by definition, so it wins the search's
    /// dominant criterion perfectly while having survived no test at all. Returning `Ok(empty)`
    /// would hand that rule to a reader as a validated trigger.
    #[test]
    fn an_empty_pool_is_refused_rather_than_returned() {
        assert!(matches!(
            Pool::matched(Vec::new(), &context()),
            Err(PoolError::BackendMismatch { .. })
        ));
    }

    /// Negatives drawn under a different generator are refused.
    ///
    /// **The failure this prevents is subtle and total.** If the positives came from a
    /// special-value generator and the negatives from an ordinary one, then `has_nan_input`
    /// separates them perfectly — and describes which generator ran, not what triggers a bug.
    /// The search would report a flawless rule that means nothing.
    #[test]
    fn negatives_from_a_different_generator_are_refused() {
        let theirs = SamplingContext::new(
            "float-elementwise=off comparisons=on quantized=off special-values=on logic=deadbeef",
            &["tract", "onnxruntime"],
        );
        let pool = Pool::matched(vec![negative(Source::Interesting, &theirs)], &context());
        assert!(matches!(pool, Err(PoolError::Empty)));
    }

    /// Negatives observed on different runtimes are refused, even when hand-built.
    ///
    /// A negative asserts "these implementations agreed on this case". Run a different set and
    /// the assertion is about something else. `Constructed` is exempt from the *distribution*
    /// check and never from this one.
    #[test]
    fn negatives_from_different_runtimes_are_refused_even_when_constructed() {
        let elsewhere = SamplingContext::new(
            "float-elementwise=on comparisons=on quantized=off special-values=on logic=3b64999b",
            &["candle"],
        );
        let mut hand_built = negative(Source::Constructed, &elsewhere);
        hand_built.provenance = Provenance::Constructed;

        assert!(matches!(
            Pool::matched(vec![hand_built], &context()),
            Err(PoolError::BackendMismatch { .. })
        ));
    }

    /// A matching pool is accepted and keeps its members.
    ///
    /// The positive control for the three refusals above: without it, a `matched` that refused
    /// everything would pass all of them.
    #[test]
    fn a_pool_from_the_same_conditions_is_accepted() {
        let ctx = context();
        let pool = Pool::matched(
            vec![
                negative(Source::Interesting, &ctx),
                negative(Source::NearMiss, &ctx),
            ],
            &ctx,
        )
        .expect("same generator and same runtimes");
        assert_eq!(pool.len(), 2);
    }

    /// **The provenance parser must recognise a description this domain actually emits.**
    ///
    /// It was copied from the tensor adapter, where it keys on `"fuzzer bytes"`, `"magnitude:
    /// 1000"` and a `"Bounds"` prefix — none of which appear in an ONNX generator description.
    /// Every real negative therefore parsed as `Unknown`, and `Pool::matched` discards `Unknown`,
    /// so the pool would have been empty for every corpus this adapter can produce.
    ///
    /// Caught by the *positive* control above, not by any of the three tests asserting that bad
    /// pools are refused — all three passed while nothing could ever be accepted. **A guard suite
    /// with no positive case cannot tell "correctly strict" from "broken shut".**
    ///
    /// The string below is copied verbatim from a campaign log.
    #[test]
    fn provenance_is_parsed_from_a_description_this_domain_really_emits() {
        let real = "float-elementwise=on comparisons=on logical=on structural=on \
                    shape-input-ops=on quantized=on float64=on integer-types=on bool-type=on \
                    special-values=on degenerate-shapes=on max-rank=4 max-dim=8 \
                    element-budget=256 special-rate=0.25 opset=22 logic=3b64999b";
        assert_eq!(
            Provenance::from_generator(real),
            Provenance::SeededWide,
            "the quantized axis widens the distribution"
        );

        let unquantized = real.replace("quantized=on", "quantized=off");
        assert_eq!(
            Provenance::from_generator(&unquantized),
            Provenance::SeededDefault
        );

        assert_eq!(
            Provenance::from_generator("built by hand for F-005"),
            Provenance::Constructed
        );
        assert_eq!(
            Provenance::from_generator("something nobody recorded"),
            Provenance::Unknown
        );
    }

    /// `is_interesting` finds a signed zero, which is what two of four problems are made of.
    #[test]
    fn a_signed_zero_is_interesting() {
        assert!(is_interesting(&case(TensorData::F32(vec![-0.0]))));
        assert!(
            !is_interesting(&case(TensorData::F32(vec![0.0, 1.0]))),
            "positive zero is an ordinary value"
        );
    }

    /// And an integer at its type boundary, where wrapping and saturation part company.
    #[test]
    fn an_integer_boundary_is_interesting() {
        assert!(is_interesting(&case(TensorData::I32(vec![i32::MIN]))));
        assert!(!is_interesting(&case(TensorData::I32(vec![0, 5, -5]))));
    }

    /// **The stratification test never looks at shapes or operators.**
    ///
    /// Deliberate, and the reason is recorded on `is_interesting`: filtering negatives with the
    /// structural half of the search's own vocabulary would select cases that confirm what is
    /// already believed. An empty tensor is a *feature*, so it must not also be a selection
    /// criterion here.
    #[test]
    fn stratification_ignores_shape_so_it_cannot_pre_select_on_the_vocabulary() {
        let empty = OnnxCase::new(
            OpKind::Abs,
            OPSET,
            vec![TensorValue::new("a", vec![0], TensorData::F32(vec![]))],
        );
        assert!(
            !is_interesting(&empty),
            "shape must not drive the negative selection"
        );
    }
}
