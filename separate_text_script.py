def split_file(input_file, output_prefix, lines_per_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    for i in range(0, total_lines, lines_per_file):
        output_file = f"{output_prefix}_{i}_{i + lines_per_file}.txt"
        with open(output_file, 'w') as outfile:
            outfile.writelines(lines[i:i + lines_per_file])
        print(f"Created {output_file}")

# Parameters
input_file = 'Data/HateCheck_test_suite_cases.txt'
output_prefix = 'Data/HateCheck_test_suite_cases'
lines_per_file = 20

# Execute the function
split_file(input_file, output_prefix, lines_per_file)
