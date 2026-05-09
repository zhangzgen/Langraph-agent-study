---
name: python-debug-helper
description: Debug Python application errors in Chinese. Use when the user provides Python tracebacks, package/import errors, virtual environment issues, API client errors, LangChain or LangGraph runtime errors, or asks why a Python command failed.
---

# Python Debug Helper

## Workflow

Follow this sequence:

1. Identify the first user-code frame in the traceback.
2. Separate warnings from fatal exceptions.
3. Explain the direct cause before suggesting fixes.
4. If the error involves environment variables, check whether the variable is loaded without printing secrets.
5. If the error involves an API response, distinguish local code problems from server-side validation errors.

## Response Style

- Answer in Chinese.
- Lead with the root cause.
- Quote only the key error line, not the whole traceback.
- Give the smallest command or code change that verifies the fix.
- For this project, remind that `.env` is intentionally ignored and should not be committed.

## Common Patterns

Missing environment variable:

```text
RuntimeError: 缺少 XIAOMI_API_KEY
```

Explain that the program did not load `.env` or the variable is absent.

Unsupported model:

```text
Not supported model ...
```

Explain that the request reached the provider, but the model name is invalid for that endpoint.

LangGraph warning:

```text
LangChainPendingDeprecationWarning
```

Explain that this is not fatal unless execution stops with an exception.
