# Security Policy

## Supported versions

We patch the latest minor release on `main`. No back-port branches.

| version  | status        |
|----------|---------------|
| 0.x.x    | ✅ supported   |
| < 0.3    | ❌ unsupported |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security bugs. Use one
of these private channels:

1. [GitHub private vulnerability report](https://github.com/Ary5272/turbocpp/security/advisories/new)
   (preferred — gives us a private fork to coordinate the fix in)
2. Email the maintainer through the GitHub profile contact link

We respond within **72 hours** and aim to ship a fix within **14 days**
of confirmation. Reporters who follow this disclosure process get
credit in the release notes (opt-out available).

## Scope

In scope:

- The `turbocpp` Python package and its CLI
- The published Docker images at `ghcr.io/ary5272/turbocpp:*`
- The HuggingFace Space at `AIencoder/turboquant-visualizer`
- The release / CI workflows that publish artifacts

Out of scope:

- Bugs in the upstream `ggml-org/llama.cpp` Docker image we pull at
  runtime (report to https://github.com/ggml-org/llama.cpp/security)
- Bugs in `llama-cpp-python` (report to https://github.com/abetlen/llama-cpp-python)
- DoS by feeding huge models on a small machine (architectural)

## Supply-chain hardening in this repo

- All third-party GitHub Actions are pinned to immutable commit SHAs,
  not floating tags. See the `# v…` inline comments in `.github/workflows/`.
- Workflow `permissions:` default to `contents: read` and grant write
  scopes only on the steps that need them.
- Wheels and Docker images shipped from this repo carry [SLSA build
  provenance attestations](https://slsa.dev/attestation-model). Verify with
  `gh attestation verify`.
- `actions/checkout` runs with `persist-credentials: false` so the
  `GITHUB_TOKEN` is never written to `.git/config` on the runner.
- A weekly [`gitleaks`](https://github.com/gitleaks/gitleaks) scan and
  GitHub's CodeQL static analysis run against `main`.
- Dependabot is on for `pip`, `docker`, and `github-actions`.

## Verifying a release

```bash
# wheel (download from a Release first)
gh attestation verify turbocpp-py3-none-any.whl \
       --owner Ary5272

# docker image
gh attestation verify oci://ghcr.io/ary5272/turbocpp:cpu \
       --owner Ary5272
```

A green check means the artifact came out of *this* repo's release
workflow at *that* commit, signed by GitHub's sigstore-backed OIDC
identity. No replay attack on the bytes is possible without invalidating
the attestation.
