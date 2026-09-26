# Security

## Report a problem

Use private vulnerability reporting in the repository's Security tab when that option is enabled.
Otherwise, contact the repository maintainers through an established private channel.
Do not place credentials, private memory, model files or exploit details in a public issue.

Include the source revision, build profile, enabled components, reproduction steps and observed result.
Use synthetic input when it can reproduce the defect.
Keep the original logs available for a private follow-up.

## Local operator authority

The runtime, gateway and disk workers use a trusted operating-system account.
Account access has operator authority over files, local commands and deployment grants.
The local terminal and service sockets restrict connections to the same user.
These sockets do not isolate an operator from the runtime it owns.

Native CUDA tools and creator policies share the runtime's CUDA context.
They are trusted code. The loader does not sandbox their memory access or guarantee kernel termination.
A native creator policy requires the exact external trust digest before activation.

Tool policy restricts the calls available to model-generated requests.
Requested module rights cannot exceed the operator's effective tool mask.
The file tool root constrains the supplied file interface; it is not a sandbox for arbitrary native code.

## Network access

The gateway authenticates bearer credentials and translates them into scoped device requests.
CUDA checks current grants, ownership, membership and resource limits.
A request handle, media digest or conversation ID is not an access credential.

Keep configuration and grant files private to the service account.
Use TLS, a trusted reverse proxy, a VPN or a tunnel for remote access.
The gateway does not trust forwarded identity headers or cookie authentication.
See [gateway deployment](docs/31-http-gateway.md#network-deployment) for TLS, origin and HTTPS-import rules.

Changing a grant revision can revoke access to admitted requests.
Restoring a complete runtime requires fresh deployment grants.
Credentials are not packaged as part of the runtime identity.

## Files and model assets

The model reader verifies declared file digests and supported tensor layouts.
A matching digest identifies bytes; it does not establish safe native code or correct model behavior.
Control qualification files also require operator review of their referenced evidence.

CCIR validates committed generations and required component identities.
The file is not encrypted. It can contain private text, source media, model assets and historical decisions.
Apply operating-system access controls and suitable storage protection to runtime files, journals and backups.
Deleting a current object or retiring media does not erase its historical journal records.

The model fetcher reads `HF_TOKEN` from the environment and does not write it to its model files.
Keep credentials outside model prompts, shared messages and diagnostic captures.

## Scope of checks

Boundary checks cover malformed input, stale identities, scope, capacity and recorded recovery.
They do not prove that a model will follow every instruction or select every relevant memory.
See [support and qualification](docs/support.md) for current functional and numerical limits.
