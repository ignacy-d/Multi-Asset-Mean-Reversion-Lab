# Canonical research workflow

Every study follows this lifecycle:

```text
idea
  -> research question
  -> preregistration
  -> implementation
  -> unit, synthetic, and point-in-time tests
  -> pre-run adversarial review
  -> SPEC LOCK
  -> authenticated empirical run
  -> classification
  -> post-run software review
  -> post-run quantitative-research review
  -> bug-fix-only loop, when necessary
  -> external review
  -> human merge decision
```

## Preregistration states and empirical visibility

A **DRAFT** preregistration may be edited while the research question and
method are being designed. It must not authorize an empirical run. A
preregistration becomes **FROZEN** only after its research contract is complete,
reviewed, and ready to be bound by a spec lock.

**Empirical visibility** begins when anyone or any process can observe an
outcome computed from research data for the frozen study. The preregistration
must be frozen and locked before that boundary. Development before the boundary
uses unit tests, synthetic fixtures, and explicitly permitted development data.
Sealed out-of-sample data remains inaccessible until a separately authorized
confirmation stage opens it.

The spec lock records the exact preregistration bytes used by a run. It binds a
run to a revision; it does not prove that the study design is scientifically
valid. An empirical runner must require the lock, verify the frozen-spec digest,
and record authenticated input identities in its run manifest. Research data
access is explicit and allowlisted. A runner must fail if a declared input or
identity is missing and must never discover a replacement from the filesystem.

## Changes after empirical visibility

A **legitimate implementation bug fix** makes the implementation conform to
the already-frozen contract. Record the defect, add a regression test, preserve
the original run and classification, issue corrected artifacts with provenance,
and repeat both post-run reviews. If a proposed correction changes the research
contract, it is not a bug fix.

A **forbidden methodology rescue** changes the hypothesis, primary outcome,
thresholds, gates, universe, horizon, sample requirement, classification rule,
or another frozen choice in response to visible results. Such a change requires
a new study and preregistration; it cannot replace or improve the original
classification. PASS, KILL, and INCONCLUSIVE are all valid outcomes.

The post-run software review checks execution and provenance. The separate
quantitative-research review checks adherence to the frozen contract, discovery
versus confirmation boundaries, and interpretation. External review follows
only after legitimate corrections and reruns are complete. Software never
manages or merges a pull request for a study; a human makes the merge decision.

Research modules remain independent of broker, live-runtime, and execution
integrations throughout this lifecycle.
