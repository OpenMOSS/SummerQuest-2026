# Public Tests

Command:

```bash
cd ../assignment1-basics
uv run pytest
```

Latest run:

```text
================== 47 passed, 1 xpassed in 123.31s (0:02:03) ===================
```

Notes:

- The latest run was executed after syncing the final scripts and implementation.
- `xpassed` is the upstream tokenizer memory test marked as expected-fail by the public test suite; it passing does not indicate a failure.
