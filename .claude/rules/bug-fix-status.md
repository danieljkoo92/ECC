# Bug Fix Verification and Status Light

Applies to every bug fix, regression, failing test, or "it's broken" request in this repo, in local and cloud Claude Code sessions alike.

## Fix the Root Cause

1. Reproduce the failure and find why it happens. Do not patch the symptom.
2. Fix the cause in the code that owns it.
3. Verify with the relevant checks: the test or command that reproduced the failure, plus the repo's own fast checks (lint, tests, build) that touch the change.

## Close Every Response With One Status Line

End every response with exactly one short line: what changed, then a single status light.

| Light | Meaning | Rule |
|---|---|---|
| Green circle | Verified working | Only after running a check that proves it. Name the command or test that ran. Never green without that. |
| Yellow circle | Should work, untested | The change is reasoned through but no check ran, or no check could run here. |
| Red circle | Guessed or stuck | The cause is unknown, the fix is a guess, or a check failed and is not resolved. Say what is blocking. |

Print the light as the actual emoji (green, yellow, or red circle), not as a word. It is spelled out here only to keep this file plain ASCII.

Example closing line: "Changed: moved the null check ahead of first use in session-start. Ran node tests/run-all.js, all passing." followed by the green circle emoji.

## Examples

- A test fails, you change one line, and the test suite passes when re-run: green, naming the suite.
- You fix a bug by reading the code but the project has no runnable tests here: yellow, saying no check ran.
- You cannot reproduce the bug and change what looks likely: red, saying the cause is unconfirmed.
