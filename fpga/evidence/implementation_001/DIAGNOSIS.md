# Vivado implementation failure: 2026-11-24

Resolved on 2026-11-24 after the user changed Smart App Control through Windows settings. A fresh run completed successfully; reports are retained in `../implementation_retry_002/`. This file preserves the earlier failure evidence.

Synthesis completed successfully. `opt_design` stopped during timing-cache initialization while loading `C:/AMDDesignTools/2025.2/Vivado/lib/win64.o/veam/zynquplus/zynquplus.dll`.

Windows Code Integrity events 3077 and 3033 at 2026-11-24 08:53:03 JST identify vivado.exe and this DLL as failing the signing requirements of policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. The DLL exists and its Authenticode status is NotSigned. This is direct evidence of an application-control block, not merely an inference from a DLL-loading message. Exported events are in `code_integrity_events.json`.

No security setting was disabled or changed. Completing placement/routing requires an administrator-approved policy allowance or an approved Vivado environment on another computer/OS. Repeating the same run under the same policy is not expected to resolve the block.

The existing Icarus real-feature simulation is independent and can continue. Do not interpret the failed run as a post-route timing/resource result. The Tcl now requests synthesis utilization before opt_design so a future run can retain the partial-stage report as well.
