# Final — copy-paste ready

Exactly what goes into the GitHub issue form, and nothing else. No checklists, no status
headers, no notes to ourselves — everything in these files is intended to be read by a
maintainer.

## Layout

Two files per issue, matching the two fields GitHub asks for:

```
<project>-<NNN>-title.txt   -> the Title field    (select all, copy, paste)
<project>-<NNN>-body.md     -> the Body field     (select all, copy, paste)
```

Split rather than one file with a separator, so each is a clean select-all with nothing to
trim afterwards.

## Relationship to `../`

The parent directory holds the **working draft**: the same text plus the triage checklist,
what was deliberately left out, what is inference versus verified, and what to do
depending on the answer. That is for us. **This directory is for them.**

A file appearing here means the draft's checklist is complete and Kevin has reviewed it.
Anything still uncertain stays in the parent.

## After filing

Record the URL and date in the parent draft, and set its status to `FILED`. Leave these
files unchanged as a record of exactly what was sent.
