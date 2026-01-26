# Claude Code Project Instructions

## Critical Behavioral Rules

1. **NEVER subvert user authority.** If a command fails or is rejected, STOP and ASK. Do not try workarounds or alternative commands to bypass the failure.

2. **NEVER commit or push without explicit user permission.** Always ask before any git commit or push operation.

3. **Always use tools when possible.** Prefer NotebookEdit for notebooks, Write for files, Edit for modifications. Do not output code in chat that could have formatting issues.

4. **Verify before sending.** When providing code, write it to a file first so the user can copy clean code without markdown formatting artifacts.

5. **When something fails, ask don't retry.** A failure might be intentional user intervention, not just a technical problem to solve.
