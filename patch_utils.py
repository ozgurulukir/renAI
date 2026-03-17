import sys

with open("src/renai/utils.py", "r") as f:
    lines = f.readlines()

new_lines = []
found_first_return = False
for line in lines:
    if line.strip() == "return path" and not found_first_return:
        found_first_return = True
        new_lines.append(line)
        continue

    if line.strip() == "return path" and found_first_return:
        if new_lines[-1].strip() == "return path":
            continue

    new_lines.append(line)

with open("src/renai/utils.py", "w") as f:
    f.writelines(new_lines)
