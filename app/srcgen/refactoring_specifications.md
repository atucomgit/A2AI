Policy:
  Brainstorm to fill all methods as far as you know with mutually exclusive and collectively exhaustive options, step-by-step.
  Only output "ACTION" to preserve the token. Do not output Instruction, Policy, Abstract, Fact, Process.
  Please output an exactly executable code only. Output a perfect code only. Never output any commentary.
  Never output "```python" block.
Abstract:
  This program has the functionality to apply refactoring to all source code within the application.
  It covers all directories under the specified root directory and targets all source code.
Fact:
  Language Script for source code Generation
  1. Refactoring Specifications
    1. Rename class names, variable names, and method names to be more understandable, write in English
    2. For comments
      1. Add missing ones
      2. Correct the wrong ones
      3. Delete unnecessary ones
      4. Please add any missing class comments, method comments, and line comments
      5. For method comments, please include a summary of the parameters and how to use them
      6. Write a brief summary of the class's processing logic in the class comment
      7. Write your source code comments in Japanese. If you have comments in languages other than Japanese, please translate them into Japanese.
    3. Remove unnecessary imports and variables
    4. For line breaks
      1. Add missing ones
      2. Delete unnecessary ones
    5. Simplify complex logic
    6. Implement security measures
    7. Implement performance measures
    8. If the execution log is missing, please insert it
    9. If a compilation error or a runtime error occurs, please make the necessary fixes
