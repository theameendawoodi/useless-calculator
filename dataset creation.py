import csv
import random

def get_result(op1, op2, operator):
    if operator == 1:      # +
        return op1 + op2
    elif operator == 2:    # -
        return op1 - op2
    elif operator == 3:    # *
        return op1 * op2
    elif operator == 4:    # /
        return round(op1 / op2, 4)  # Round for cleaner float results

def generate_dataset(filename, num_records=100000):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['operand1', 'operand2', 'operator', 'target'])
        
        for _ in range(num_records):
            operand1 = random.randint(1, 1000)
            operand2 = random.randint(1, 1000)
            operator = random.randint(1, 4)

            # Ensure no division by zero
            if operator == 4:
                while operand2 == 0:
                    operand2 = random.randint(1, 1000)

            result = get_result(operand1, operand2, operator)
            writer.writerow([operand1, operand2, operator, result])

    print(f"Dataset with {num_records} records saved to '{filename}'.")

# Run it
generate_dataset('arithmetic_dataset.csv')
