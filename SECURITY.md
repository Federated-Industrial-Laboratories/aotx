# Security

Report a security problem through a private report on GitHub (the Security tab), not through
a public issue. Include the version, the profile, the steps and the observed result.

AOTX is a local program. A tool program requested by an agent runs only after an operator
grant. The model fetch verifies the digest of each file before use. The terminal socket
accepts connections from the same user only. `HF_TOKEN` is read from the environment and is
never written to disk.
