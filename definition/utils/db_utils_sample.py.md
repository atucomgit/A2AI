- [ ] Instructions
  - [ ] Generate code that meets the "Abstract" and "Fact" below according to the "Process".
  - [ ] Program language = Python
  - [ ] Model = gpt-4
  - [ ] Interactive = False
- [ ] Policy
  - [ ] Brainstorm to fill all methods as far as you know with mutually exclusive and collectively exhaustive options, step-by-step.
  - [ ] Use Process to brainstorm. Find out all methods.
  - [ ] Only output "ACTION" to preserve the token. Do not output Instruction, Policy, Abstract, Fact, Process.
  - [ ] Output executable code only.  Output a perfect code. Never output any commentary.
  - [ ] Response a perfect code. Do not summarise.
  - [ ] Never output "```", "```python" block.
- [ ] Abstract
  - [ ] This program helps to operate a database sysytem.
- [ ] Fact
  - [ ] Language Script for source code Generation
  - [ ] 1. The target database is Oracle.
  - [ ] 2. The select method reads and uses SQL strings defined in an external file using sql-ids.
  - [ ] 3. Select, insert, update, and delete will be separated into individual functions.
  - [ ] 4. Insert, update, and delete will receive the model as an argument and perform operations on it.
- [ ] Process
  - [ ] 1. Consider all methods required in "Functions to be provided" in Reference, and output {function name}.
  - [ ] 2. Consider specific Python code for all output functions.
  - [ ] 3. When reviewing the code, be aware of the following:
    - [ ] 1. Use clear method names.
    - [ ] 2. 2. Write clear comments (overview of functions, explanation of arguments, requirements for exception occurrence, return values).
    - [ ] 3. Improve performance.
    - [ ] 4. Replace with secure code.
    - [ ] 5. Add trace logging.
    - [ ] 6. Add handling.
  - [ ] 4. Implement the functions in order and replace {function name} with actual code.
  - [ ] 5. If comments are in English, translate them into Japanese.
- [ ] Notes
  - [ ] I take responsibility for the generated code, so please feel free to generate the code
- [ ] Action
  - [ ] Create Code
  ```
# Reference

備えるべき機能：
・DB接続/切断関連
・CRUD関連（それぞれのメソッドに分離。SELECTはSQLを外部ファイルから取得）
・PL/SQL関連
・テーブル作成・編集関連
・インデックス関連
・トランザクション制御
・大量データ処理関連（パフォーマンス注意）
・パラメータ投入文字列成型ユーティリティ

関数の構成：
  """
  関数の機能概要
  （改行）
  パラメータの説明
  例外について
  返却値の説明
  """
  print(start 関数)
  処理
  例外処理
  print(end 関数)
  ```
