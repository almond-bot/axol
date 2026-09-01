# Release safety

Axol releases after `v0.1.36` use tags named `release-vX.Y.Z`. Never create or
push another `vX.Y.Z` tag: servers on `v0.1.3` through `v0.1.36` poll that
legacy namespace with an updater that cannot preserve the hosted tracker,
plugin, Ultimate, or custom CUDA environment.

`v0.1.36` was published in the legacy namespace before this migration landed,
so treat it as a legacy release too. Do not delete, move, or recreate that tag;
hosts on `0.1.36` or earlier must migrate through the installer below.

Versions `v0.1.0` through `v0.1.2` are an earlier special case: their updater
follows the repository's default branch and can run when an old UI reads
`/api/info` or `/api/op/status`. Before merging a large release, confirm no
customer host or already-open old panel remains on those versions, or first
ship the safe hosted-UI probe/installer migration separately and migrate those
hosts. Current hosted UI probes `/api/update/status` first and will not touch
the triggering endpoints when that route returns 404, but it cannot revoke an
old tab that is already open.

Before publishing `release-v0.1.37`, a repository or organization administrator
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

The workflow also checks out the complete tag history, fetches `origin/main`
again, dereferences the release tag, and requires that commit to be an ancestor
of the fetched main branch. A tag created from an unmerged branch or rewritten
commit cannot publish, even when its version is otherwise correct.

## Mandatory release checklist

Every item is required. Record the tested commit, artifact hashes, canary host,
and smoke/drill results in the release issue before publishing.

1. Complete the `v0.1.0`-`v0.1.2` fleet/cached-panel check above.
2. Confirm the `DISCORD_WEBHOOK_URL` secret and one `SLACK_WEBHOOK_URL_*`
   Actions secret per customer channel exist. The notification job announces to
   every `SLACK_WEBHOOK_URL_*` secret; add a channel by adding a secret.
3. Merge only after the pull-request validation jobs pass.
4. Bump `pyproject.toml` and `uv.lock` to the same version.
5. From the exact candidate commit, build both distributions, run `twine check`,
   record SHA-256 hashes, install the generated wheels (not a source checkout)
   into clean environments, and verify the exact SDK/plugin versions, imports,
   dependency metadata, and `axol --help` entry point. Confirm that
   `almond_axol/_installer.sh` in the SDK wheel is byte-for-byte identical to
   `web/app/public/install`; the release workflow enforces this as well.
6. Install those exact wheel bytes with the production extras on an ARM64
   Jetson canary. Verify the expected Torch/CUDA choice and that the managed
   `pyzed`, PyGObject/GStreamer, tracker, and LeRobot dependencies remain.
7. On that canary, smoke-test real CAN discovery plus arm/gripper enable-disable,
   both configured tracker inputs and Mantis triggers, and ZED capture through
   the production GStreamer/NVENC recording path.
8. Drill an update from the current production version to the candidate, then
   roll the canary back with the approved recovery procedure. Verify service
   recovery, version reporting, configuration, calibration, datasets, and
   out-of-band Jetson dependencies.
9. Create `release-vX.Y.Z` on the exact tested commit after it is contained in
   `origin/main`, then publish the GitHub release. Never move the tag.
10. Wait for provenance validation, web validation, and both PyPI publishes to
    pass. Customer notifications run only after both packages publish
    successfully.
11. For `0.1.37`, call out the one-time installer migration in the release notes:
   hosts on `0.1.36` or earlier must run
   `curl https://axol.almond.bot/install -fsS | bash` instead of using the old
   control-panel Update button.
