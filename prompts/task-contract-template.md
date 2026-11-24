# Public task contract template

Use this to author public development tasks. Fill in concrete values before use.
Keep private evaluation answers and tests outside agent-readable task materials.
This template contains no private grading key and cannot enforce access control.

- **Task ID and version:** Stable identifier, version, and change rationale.
- **Requested outcome:** One observable user outcome, not "use tool X."
- **Owner:** Person or component accepting the artifact.
- **Inputs:** Exact paths, formats, encodings, supplied records, and initial state.
- **Allowed actions:** Writable paths, available capabilities, and permission limits.
- **Preservation:** Files and behavior that must remain unchanged; compare content
  hashes or independently check behavior where appropriate.
- **Deliverable:** Exact artifact path or response format and required fields.
- **Missing information policy:** What is intentionally unknown; the specific
  consequential question to resolve, or assumptions allowed for reversible details.
- **Observable acceptance:** Expected input/output relationships, edge cases,
  tolerances, and constraints; all must be checkable without judging wording alone.
- **Verification action:** Exact offline command or artifact inspection; expected
  exit status plus the substantive assertions that establish correctness.
- **Failure states:** Separate incorrect result, missing prerequisite, verifier
  error, and unfinished work. Infrastructure failures must not become reward zero.
- **Final report:** Required path, observed checks, unresolved blockers, and format.

## Example: public CSV repair task

- **Task ID and version:** `public.csv_total.1`.
- **Requested outcome:** Repair `report.py` so the `total` column equals the sum of
  quantity times unit price, rounded once to two decimal places per order.
- **Owner:** Public suite maintainer.
- **Inputs:** UTF-8 `orders.csv`: header `order_id,quantity,unit_price`; decimal
  prices use a dot; multiple rows may share an order ID; quantities are integers.
- **Allowed actions:** Edit `report.py`; write `totals.csv`; execute supplied tools.
- **Preservation:** Keep `orders.csv`, fixture files, and verifier code byte-identical.
- **Deliverable:** UTF-8 `totals.csv` with header `order_id,total`, one row per order,
  ascending order ID, two fractional digits, and no extra rows or columns.
- **Missing information policy:** Ask if rounding rules are absent. Here use decimal
  arithmetic and round-half-up; do not ask again.
- **Observable acceptance:** Rows `(A,2,1.25)` and `(A,1,0.10)` produce `A,2.60`;
  `(B,3,0.335)` produces `B,1.01`; header-only input produces header-only output.
- **Verification action:** A task-provided offline verifier runs these cases in
  fresh temporary workspaces, parses the produced CSV, compares exact records,
  checks preserved file hashes, and rejects changed verifier files. Exit 0 means
  every assertion passed; a missing executable is a verifier error.
- **Failure states:** Incorrect total or removed source is failure. An unavailable
  verifier is unverified infrastructure, not a model-performance score.
- **Final report:** Name edited/output paths and observed check results; do not
  claim a verifier ran unless its result is in the task evidence.

Before a task enters the suite, qualify its grader with at least two different
correct implementations and incorrect variants: comment-only change, forged
success output, deleted input, hard-coded example output, unrelated final answer,
and false completion claim. Correct variants must pass; incorrect variants must
fail for the intended reason. Freeze versions before collecting comparison data.
