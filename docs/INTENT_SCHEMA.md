# Intent template and slot behavior

`data/intents.json` controls whether an intent can serve a target. A non-empty
`templates.<target>` is required. For target `latex` only, an empty/missing
`templates.latex` may fall back to `templates.mathjax`; no other target fallback
is implied. Thus an Obsidian-only intent remains in the bank and retrieval index,
but the LaTeX pipeline returns `unsupported_target` if selected. Adding a
`templates.python` value makes that intent available with `ka ask ...
--target python`; no intent-specific code is needed.

Slot definitions keep a single-capture `regex`. Free text patterns must have a
leading non-capturing keyword group or a fixed suffix. A bare `(.+)`, `(.*)`, or
`(.+?)` is treated as ungrounded and requests clarification. A keyword-anchored
text slot is cut off at the earliest keyword anchor belonging to another slot in
the same intent.

The optional numeric field `absent_means` supplies a value only when the slot
does not match and the normalized request is an explicit equation ending in
`= 0`. For example, the quadratic intent marks its numeric coefficient slots
with `absent_means: 0`; this does not fill slots for a generic request without
an equation. Validator rejects this field on non-number slots.

Numeric slots may declare `not_equal` plus an optional `constraint_error` to
reject degenerate values through the generic slot validator. This replaces
intent-name-specific validation hooks.

The optional `post` field names a registered pure post-action in
`core.latex_tools.POST_ACTIONS`. `tidy_math` is conservative and runs only for
intents declaring it. Unknown actions fail validation and rendering.

Vietnamese spoken math normalization is applied only to number/math slots and
only when the request has mathematical cues. Text slots retain the original
normalized wording. Unsupported or ambiguous values remain missing for the
caller to clarify; the rules do not use a model or intent-name branches.
Math slots preserve LaTeX control sequences and text-like macro contents while
normalizing spacing; unsafe control words are rejected and the rendered result
still passes through the static LaTeX checker.

Templates must contain actual line breaks. A literal backslash followed by
`n` is linted because it renders as text rather than a newline; LaTeX commands
such as `\nabla` remain valid.
