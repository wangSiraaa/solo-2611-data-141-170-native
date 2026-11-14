# Offline adaptive detection-policy analyser

A small, exact, offline tool that decides **when to buy an inspection test and
when to stop and act** for a device whose true state is unknown.  It is a
decision tool only: no model training, no online learning, no connection to a
device, and no generic decision platform.

Everything is computed with Python's `fractions.Fraction` from the original
text literals, so there are **no floating-point comparisons and no tolerances**
anywhere in the decisions.

## What it optimises

At every reachable observation history `h`:

```
V(h) = min( min_action  E[ loss(action, state) | h],
            min_test    fee(t) + E[ V(h + result(t)) ] )
```

subject to

* each test is bought at most once on any path,
* `requires` tests are bought before the test that needs them,
* cumulative fees along **each branch** never exceed the budget,
* the posterior is computed from the frozen **joint** distribution of results
  (correlated tests are never multiplied as if independent),
* a zero-probability observation is never given a fabricated posterior.

The reported cost is the expectation of `(fees actually paid) + (terminal
loss)`.  Information value is reported separately (see below).

## Layout

```
policy_tool/
  model.py       exact parsing, validation, posterior/likelihood from the joint
  solver.py      order-free DP over reachable histories; exact ties; accounting
  replay.py      replay one state/flat vector, re-average over the original joint
  report.py      static strategy tree + Markdown report
  bruteforce.py  independent exhaustive cross-check (tests only)
  cli.py         solve / replay / validate
examples/demo.py regenerates everything under out/
tests/test_tool.py  unittest suite (stdlib only)
fixtures/            provided inputs + correlated.json diagnostic case
```

## Usage

```sh
# validate inputs
python3 -m policy_tool validate --model fixtures/model.json \
    --actions fixtures/actions.csv --plans fixtures/plans.json

# solve one budget, readable tree + cost composition
python3 -m policy_tool solve --model fixtures/model.json \
    --actions fixtures/actions.csv --plans fixtures/plans.json --budget 5

# solve all plan budgets and write JSON + Markdown + tree per budget
python3 -m policy_tool solve --model fixtures/model.json \
    --actions fixtures/actions.csv --plans fixtures/plans.json \
    --all-budgets --out-dir out/model

# replay a concrete state and full result vector
python3 -m policy_tool replay --model fixtures/model.json \
    --actions fixtures/actions.csv --budget 5 --state bearing --outcomes 1011

# tiny hand-check
python3 -m policy_tool solve --model fixtures/tiny.json

# regenerate the report bundle
python3 examples/demo.py

# tests
python3 -m unittest discover -s tests -v
```

`tiny.json` is self-contained (embedded `joint`/`actions`); the main fixtures use
`model.json` + `joint.csv` + `actions.csv` + `plans.json`.

## History representation and why it merges

A history is keyed by

```
( set of tests already bought , observed result for each bought test )
```

not by purchase order.  Two orders that buy the same set and see the same
results have the same posterior, the same remaining budget, and -- because the
joint is frozen ("原始检测结果一次抽定") -- the same outcome distribution.  They
are therefore one DP sub-problem and are solved once (memoised).  This is what
the `memo_cache_hits_merged` counter reports.

Tests are never globally pre-ordered and executed top to bottom; the DP compares
"stop" against "buy any currently purchasable test" at every node.

## Ties and numerical equality

A candidate is a tie **only** when its exact `Fraction` cost is identical to the
node value.  There is no tolerance, so policies that differ by `1e-12` are not
treated as equal (covered by a test).  The public, stable rule for selecting the
single primary policy among exact ties is:

1. stopping beats buying a test on an exact tie (a fee exactly offset by
   information is not worth paying);
2. among tying terminal actions, the lexicographically smallest action id;
3. otherwise among tying tests, the smallest `test_order` position, then id.

All tying candidates are still listed at the node (`ties`).

## Stop explanations

Every stop node records a `reason_code`:

* `additional_information_not_worth_its_fee` -- the best buyable test costs
  more (fee + continuation) than stopping;
* `additional_information_exactly_offsets_fee` -- exact tie, stop chosen;
* `constraints_block_all_remaining_tests` -- remaining tests are blocked by
  budget and/or prerequisites;
* `all_tests_already_bought` / `no_tests_in_model`.

Plus flags `budget_limited_at_this_node`, `prerequisites_limited_at_this_node`,
and the margin over the best buyable test.

## Cost composition

Reported separately, never merged:

* **no-detection** expected terminal loss (best immediate action under the
  prior),
* expected **inspection fees** actually paid under the strategy,
* expected **terminal loss** under the strategy,
* total expected cost = fees + terminal loss,
* **gross information value** = no-detection loss - strategy terminal loss,
* **net information value** = gross - fees.

## Replay

`replay` takes a true state and the **full outcome vector**, walks the recorded
primary policy, re-scores the terminal actions from each node's posterior,
sums the fees actually paid and the real terminal loss, and *re-derives* the
expectation by enumerating every non-zero `(state, vector)` row of the frozen
joint and re-walking the policy -- it never returns the solver's stored total.
It checks that the re-averaged expectation equals the tree value and that path
probabilities reconstruct to 1.

## Input validation

Loads fail with a specific message, e.g. priors not summing to exactly 1,
joint rows for a state not summing to exactly 1, unknown state in a joint row,
wrong number of result symbols, duplicate joint rows, unknown/cyclic
`requires`, negative test cost, `test_order` referencing an unknown test,
missing action loss for a state, or probability outside `[0, 1]`.
A combination absent from the joint table is treated as probability 0; a
history that is unreachable in that way is refused rather than given a
posterior.

## Scope

Maximum supported problem: 4 states, 4 binary tests, 3 terminal actions.
The DP enumerates all reachable histories exactly.
