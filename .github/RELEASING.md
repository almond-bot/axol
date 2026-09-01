# Release safety

Axol releases after `v0.1.35` use tags named `release-vX.Y.Z`. Never create or
push another `vX.Y.Z` tag: servers on `v0.1.3` through `v0.1.35` poll that
legacy namespace with an updater that cannot preserve the hosted tracker,
plugin, Ultimate, or custom CUDA environment.

Versions `v0.1.0` through `v0.1.2` are an earlier special case: their updater
follows the repository's default branch and can run when an old UI reads
`/api/info` or `/api/op/status`. Before merging a large release, confirm no
customer host or already-open old panel remains on those versions, or first
ship the safe hosted-UI probe/installer migration separately and migrate those
hosts. Current hosted UI probes `/api/update/status` first and will not touch
the triggering endpoints when that route returns 404, but it cannot revoke an
old tab that is already open.

Before publishing `release-v0.1.36`, a repository or organization administrator
must create an active GitHub tag ruleset with all of these properties:

- target tags matching `refs/tags/v*`;
- restrict tag creation;
- no bypass actors.

Add a repository Actions secret named `RULESET_AUDIT_TOKEN` containing a
dedicated fine-grained credential scoped only to `almond-bot/axol`, with
read access to repository Administration/rulesets. The credential's owner must
be able to view the ruleset's `bypass_actors`; verify that the repository
ruleset API returns that field when authenticated with the credential. The
default `GITHUB_TOKEN` does not expose it and must not be used for this audit.

The publish workflow uses that credential only for the release-time ruleset
audit. It refuses to upload packages when the secret is missing, the API hides
the bypass list, or no matching active ruleset exists. Confirm the rule and
credential remain active before every release. Existing legacy tags stay as
historical version markers; do not update or delete them.

Release checklist:

1. Complete the `v0.1.0`-`v0.1.2` fleet/cached-panel check above.
2. Create the `SLACK_WEBHOOK_URLS` Actions secret as a JSON array containing
   the values of the existing per-channel Slack webhook secrets, then remove
   those individually named secrets after the aggregate secret is verified.
   This keeps the customer/channel roster out of the public workflow.
3. Merge only after the pull-request validation jobs pass.
4. Bump `pyproject.toml` and `uv.lock` to the same version.
5. Create a GitHub release tagged `release-vX.Y.Z` for that exact version.
6. Wait for both PyPI publishes and web validation to pass. Customer
   notifications run only after both packages publish successfully.
7. For `0.1.36`, call out the one-time installer migration in the release notes:
   hosts on `0.1.35` or earlier must run
   `curl https://axol.almond.bot/install -fsS | bash` instead of using the old
   control-panel Update button.
