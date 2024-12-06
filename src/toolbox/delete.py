# 元のファイルのパス
input_file = "requirements.txt"
# 修正後に保存するファイルのパス
output_file = "output.txt"

# ファイルを開いて読み込む
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# '=='より前の部分を抽出
processed_lines = [line.split("==")[0].strip() + "\n" for line in lines]

# 修正した内容を別ファイルに保存
with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(processed_lines)

print(f"処理が完了しました。修正後の内容は '{output_file}' に保存されました。")
