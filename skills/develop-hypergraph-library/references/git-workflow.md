# Git and GitHub workflow on this Windows machine

Use this SOP for Git operations in Hypergraph-Computation-Library.

## Local operations

Run status, diff, add, commit, and revision inspection without a PowerShell
login shell. This avoids the unrelated disabled-script warning from
`WindowsPowerShell/profile.ps1`.

Before committing:

1. Run `git status --short --branch`.
2. Inspect the staged diff and run `git diff --cached --check`.
3. Commit one coherent milestone.

## GitHub network operations

The system Git HTTPS helper has previously opened repeated
`git-remote-https.exe - Application Error` dialogs when GitHub operations were
first attempted inside the restricted sandbox. The successful path was a
standard `git push origin main` executed once outside that restriction, with
PowerShell login-profile loading disabled.

For `git fetch`, `git pull`, `git push`, or `git ls-remote`:

1. Confirm the remote and local branch with read-only local Git commands.
2. If the operation is already authorized by the user's request, run the
   normal system Git command directly with `sandbox_permissions` set to
   `require_escalated` and shell `login` set to `false`.
3. For a push, use the narrow reusable approval prefix `["git", "push"]`.
4. Do not first retry the same HTTPS command repeatedly inside the sandbox.
5. Do not switch to the bundled runtime Git or alter credential, TLS, or
   certificate configuration as a workaround; those attempts did not solve
   this machine-specific problem.
6. If approval is denied or authentication genuinely fails, stop and report
   the single failure instead of generating more application-error dialogs.

After a successful push:

1. Run `git status --short --branch` locally.
2. Compare `git rev-parse HEAD` with `git rev-parse origin/main`.
3. Report the pushed commit hash.

## Synchronizing the AutoDL working copy

The remote server is an execution copy, not the credential authority. Do not
save a GitHub password or token there and do not repeatedly retry HTTPS when
the server has no usable GitHub credential.

After the local milestone is pushed:

1. Confirm local `HEAD` equals `origin/main`.
2. On the server, use `git pull --ff-only` only when existing authentication is
   known to work.
3. If GitHub authentication fails, create a full Git bundle from the local
   repository, transfer it to the server with SSH/SCP, then run
   `git fetch <bundle-path> main` followed by
   `git merge --ff-only FETCH_HEAD` in the remote working copy.
4. Confirm the remote `HEAD` equals the pushed local commit before running new
   validation.
5. Remove temporary local and remote bundle files after the revision is
   confirmed; never include checkpoints, datasets, credentials, or ignored run
   assets in a bundle.

Never store GitHub passwords, tokens, credential-helper output, or other
secrets in the repository or this skill.
