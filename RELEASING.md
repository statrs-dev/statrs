# Releasing

Releases are handled by [release-plz](https://release-plz.dev/) via
`.github/workflows/release-plz.yml`, running on `statrs-dev/statrs`.

## How it works

1. Every push to `main` runs the `release-plz-pr` job. It looks at commits
   since the last tag, works out the next version per semver (pre-1.0: a
   breaking change bumps minor, everything else patch), updates
   `Cargo.toml` and `CHANGELOG.md`, and opens or updates a single "Release"
   PR with the diff.
2. Review that PR like any other. Adjust the changelog wording, or edit the
   version in the PR directly if the auto-computed bump isn't the one you
   want to ship.
3. Merging the Release PR to `main` triggers the `release-plz-release` job,
   which tags the commit, runs `cargo publish`, and cuts a GitHub release.

`RELEASE_PLZ_TOKEN` and `CARGO_REGISTRY_TOKEN` are already set as repo
secrets on `statrs-dev/statrs`.

## Making a release

1. Merge whatever's going into the release into `main` as usual.
2. Wait for release-plz to open (or update) the Release PR.
3. If the version it proposed isn't the one you want to ship, edit
   `Cargo.toml` (and the changelog heading) in that PR directly —
   release-plz only reads commit history, it doesn't know about anything
   you haven't written as a `feat`/`fix`/breaking-change commit.
4. Merge — the release job publishes whatever version is in `Cargo.toml`
   at merge time, so any manual edit is picked up automatically.

No local `cargo publish` or manual tagging needed once this is merged to
`main` and the workflow is live.
