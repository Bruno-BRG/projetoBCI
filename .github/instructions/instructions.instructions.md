---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

- Follow the NASA rules making code.
- Use descriptive variable and function names.
- Include docstrings for all public modules, functions, and classes.
- Write unit tests for all new features and bug fixes.
- Keep code DRY (Don't Repeat Yourself) and modular.
- Use type hints for function signatures.
- Handle exceptions gracefully and provide useful error messages.
- Keep dependencies up to date and follow best practices for security.
The ten rules are:[1]

    Avoid complex flow constructs, such as goto and recursion.
    All loops must have fixed bounds. This prevents runaway code.
    Avoid heap memory allocation after initialization.
    Restrict functions to a single printed page.
    Use a minimum of two runtime assertions per function.
    Restrict the scope of data to the smallest possible.
    Check the return value of all non-void functions, or cast to void to indicate the return value is useless.
    Use the preprocessor only for header files and simple macros.
    Limit pointer use to a single dereference, and do not use function pointers.
    Compile with all possible warnings active; all warnings should then be addressed before release of the software.