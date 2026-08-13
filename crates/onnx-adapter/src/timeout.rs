//! A bound on how long a runtime may take, and what happens when it exceeds it.
//!
//! # Measuring afterwards is not a timeout
//!
//! The easy implementation calls the runtime, checks the elapsed time, and reports `TimedOut` if
//! it was too long. That detects **slow**. It does nothing about **hung**: an infinite loop
//! inside the runtime never returns, so the elapsed check is never reached and the campaign stops
//! forever with no record of why.
//!
//! A hang is one of the outcomes this domain exists to report. So the work runs on its own
//! thread and the caller waits on a channel with a deadline. If the deadline passes, the caller
//! gives up and reports `TimedOut` while the thread keeps running.
//!
//! **The thread is then leaked, deliberately.** Rust has no safe way to kill a running thread,
//! and there is no safe way in general: the victim may hold a lock, or be halfway through
//! ONNX Runtime's C++ allocator. Leaking is the honest cost of detecting a hang in-process.
//! It is bounded in practice because a campaign that leaks threads is a campaign finding hangs,
//! which is a result worth stopping for.
//!
//! # Choosing the bound
//!
//! The bound must be **generous**, because reporting a slow-but-correct runtime as a finding is
//! a false positive that costs triage and credibility. `POLICY.md` records the measured
//! distribution it comes from. The short version, measured over the N4 corpus: ONNX Runtime's
//! mean is 0.18 ms and its **maximum** is 23 ms — and that maximum is 130× the mean while the
//! p99 is only 1.8×, so the tail is real and a bound set from a percentile would be wrong.
//!
//! The bound is set far above the observed maximum rather than just above it. A bound near the
//! tail would turn ordinary variance — a busy machine, a cold cache, a garbage collection in the
//! Python reference — into findings.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use diff_fuzzer_core::traits::{Implementation, RunError};

use crate::case::OnnxCase;
use crate::outcome::OnnxOutcome;

/// The default bound. Recorded in `POLICY.md` with the distribution behind it.
///
/// **Roughly 200× the slowest execution ever observed** (23 ms, ONNX Runtime, N3 throughput
/// measurement). Deliberately far above the tail rather than just above it: the cost of a bound
/// that is too loose is a hang taking five seconds to report, while the cost of one that is too
/// tight is a false finding against a correct runtime.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Wraps an implementation so that exceeding the bound becomes [`OnnxOutcome::TimedOut`].
///
/// The inner implementation must be `Clone + Send + 'static` because the work outlives the call
/// when it times out. In practice the runtimes are zero-sized types, so the clone is free.
///
/// **Wrap the bare runtime, not a capability-classified one.** The capability layer borrows the
/// census and so is not `'static`; the correct nesting is
/// `WithCapabilities::new(WithTimeout::new(runtime), &caps)`. That order is also the right
/// semantics: a timeout is a property of the runtime, and the capability layer must see it
/// already classified — it never rewrites `TimedOut`, just as it never rewrites `Crashed`.
#[derive(Debug, Clone)]
pub struct WithTimeout<I> {
    inner: I,
    bound: Duration,
    name: String,
}

impl<I> WithTimeout<I>
where
    I: Implementation<In = OnnxCase, Out = OnnxOutcome>,
{
    /// Wrap with [`DEFAULT_TIMEOUT`].
    pub fn new(inner: I) -> Self {
        Self::with_bound(inner, DEFAULT_TIMEOUT)
    }

    /// Wrap with an explicit bound.
    pub fn with_bound(inner: I, bound: Duration) -> Self {
        let name = inner.name().to_string();
        Self { inner, bound, name }
    }
}

impl<I> Implementation for WithTimeout<I>
where
    I: Implementation<In = OnnxCase, Out = OnnxOutcome> + Clone + Send + 'static,
{
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        let (sender, receiver) = mpsc::channel();
        let inner = self.inner.clone();
        let case = input.clone();

        let started = Instant::now();
        std::thread::spawn(move || {
            let outcome = inner.run(&case);
            // The receiver is gone if the caller already gave up. That is the normal timeout
            // path, not an error, so the send result is discarded rather than unwrapped.
            let _ = sender.send(outcome);
        });

        match receiver.recv_timeout(self.bound) {
            Ok(outcome) => outcome,
            // The thread is still running and is now unreachable. See the module comment on
            // why it is left that way.
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(OnnxOutcome::TimedOut {
                after_ms: started.elapsed().as_millis() as u64,
            }),
            // The worker died without sending — a panic that escaped the runtime's own
            // `catch_unwind`. That is a crash, and calling it a timeout would misattribute it.
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(OnnxOutcome::Crashed {
                detail: "the execution thread ended without producing a result".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::validation::well_formed;

    /// A runtime that sleeps, standing in for one that hangs.
    #[derive(Debug, Clone)]
    struct Slow(Duration);

    impl Implementation for Slow {
        type In = OnnxCase;
        type Out = OnnxOutcome;
        fn name(&self) -> &str {
            "slow"
        }
        fn run(&self, _input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
            std::thread::sleep(self.0);
            Ok(OnnxOutcome::Ok(vec![TensorValue::f32(
                "out",
                vec![1],
                vec![1.0],
            )]))
        }
    }

    fn case() -> OnnxCase {
        well_formed(OpKind::Add, &[2], 22)
    }

    /// **The property the module exists for**: the caller gets an answer even though the work
    /// never finished. An elapsed-time check could not produce this test.
    #[test]
    fn a_hang_is_reported_rather_than_waited_out() {
        let slow =
            WithTimeout::with_bound(Slow(Duration::from_secs(30)), Duration::from_millis(50));
        let started = Instant::now();
        let outcome = slow.run(&case()).unwrap();

        assert!(
            matches!(outcome, OnnxOutcome::TimedOut { .. }),
            "expected TimedOut, got {outcome:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the caller waited for the work instead of the bound"
        );
    }

    /// And a runtime that finishes inside the bound is untouched — the wrapper must not turn a
    /// correct answer into a finding.
    #[test]
    fn a_prompt_answer_passes_through_unchanged() {
        let quick = WithTimeout::with_bound(Slow(Duration::ZERO), Duration::from_secs(5));
        assert!(matches!(
            quick.run(&case()).unwrap(),
            OnnxOutcome::Ok(tensors) if tensors.len() == 1
        ));
    }

    /// The reported duration must be the bound that was exceeded, not zero and not the eventual
    /// completion time — a report saying "timed out after 0 ms" tells a reader nothing.
    #[test]
    fn the_reported_duration_reflects_the_bound() {
        let slow =
            WithTimeout::with_bound(Slow(Duration::from_secs(30)), Duration::from_millis(80));
        let OnnxOutcome::TimedOut { after_ms } = slow.run(&case()).unwrap() else {
            panic!("expected a timeout");
        };
        assert!(
            (80..2000).contains(&after_ms),
            "implausible duration: {after_ms} ms"
        );
    }

    /// The name must survive the wrapper, or every report attributes findings to "timeout"
    /// rather than to the runtime that produced them.
    #[test]
    fn the_wrapped_runtime_keeps_its_name() {
        let wrapped = WithTimeout::new(Slow(Duration::ZERO));
        assert_eq!(wrapped.name(), "slow");
    }
}
