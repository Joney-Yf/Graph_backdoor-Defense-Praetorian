import argparse
import re
import sys

def parse_logs(log_file):
    """
    Parses the log file to extract metrics before and after defense,
    including Trigger Precision, Trigger Recall, Victim Precision, Victim Recall,
    ASR, and CA.

    Parameters:
    - log_file (str): Path to the log file.

    Returns:
    - dict: A dictionary containing lists of extracted metrics.
    """
    metrics = {
        'trigger_precision': [],
        'trigger_recall': [],
        'victim_precision': [],
        'victim_recall': [],
        'asr_before': [],
        'ca_before': [],
        'asr_after': [],
        'ca_after': []
    }

    # Regular expression patterns
    trigger_precision_pattern = re.compile(r'trigger precision is\s*([0-9.]+)', re.IGNORECASE)
    trigger_recall_pattern = re.compile(r'trigger recall is\s*([0-9.]+)', re.IGNORECASE)
    victim_precision_pattern = re.compile(r'victim precision is\s*([0-9.]+)', re.IGNORECASE)
    victim_recall_pattern = re.compile(r'victim recall is\s*([0-9.]+)', re.IGNORECASE)
    asr_pattern = re.compile(r'The ASR is\s*([0-9.]+)', re.IGNORECASE)
    ca_pattern = re.compile(r'The CA is\s*([0-9.]+)', re.IGNORECASE)

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found. Please check the file path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading the file: {e}")
        sys.exit(1)

    i = 0
    flag = False
    while i < len(lines):
        line = lines[i].strip()

        if "########################## overall loss rank #################################" in line:
            flag = True
        # Extract Victim Precision
        if flag:
            victim_prec_match = victim_precision_pattern.search(line)
            if victim_prec_match:
                victim_precision = float(victim_prec_match.group(1))
                metrics['victim_precision'].append(victim_precision)
                i += 1
                continue

            # Extract Victim Recall
            victim_recall_match = victim_recall_pattern.search(line)
            if victim_recall_match:
                victim_recall = float(victim_recall_match.group(1))
                metrics['victim_recall'].append(victim_recall)
                i += 1
                continue

            # Extract Trigger Precision
            trigger_prec_match = trigger_precision_pattern.search(line)
            if trigger_prec_match:
                trigger_precision = float(trigger_prec_match.group(1))
                metrics['trigger_precision'].append(trigger_precision)
                i += 1
                continue

            # Extract Trigger Recall
            trigger_recall_match = trigger_recall_pattern.search(line)
            if trigger_recall_match:
                trigger_recall = float(trigger_recall_match.group(1))
                metrics['trigger_recall'].append(trigger_recall)
                i += 1
                continue

        if "------ Use best configs ------" in line:
            flag = False

        # Check for Before Defense section
        if "Before Defense, the Attack Performance" in line:
            flag=False
            # Expect the next two lines to contain ASR and CA
            if i + 2 < len(lines):
                asr_line = lines[i + 1].strip()
                ca_line = lines[i + 2].strip()

                asr_match = asr_pattern.search(asr_line)
                ca_match = ca_pattern.search(ca_line)

                if asr_match and ca_match:
                    asr_before = float(asr_match.group(1))
                    ca_before = float(ca_match.group(1))
                    metrics['asr_before'].append(asr_before)
                    metrics['ca_before'].append(ca_before)
                    i += 3  # Move past the ASR and CA lines
                    continue

        # Check for After Defense section
        if "After Defense, the Attack Performance" in line:
            flag=False
            # Expect the next two lines to contain ASR and CA
            if i + 2 < len(lines):
                asr_line = lines[i + 1].strip()
                ca_line = lines[i + 2].strip()

                asr_match = asr_pattern.search(asr_line)
                ca_match = ca_pattern.search(ca_line)

                if asr_match and ca_match:
                    asr_after = float(asr_match.group(1))
                    ca_after = float(ca_match.group(1))
                    metrics['asr_after'].append(asr_after)
                    metrics['ca_after'].append(ca_after)
                    i += 3  # Move past the ASR and CA lines
                    continue

        i += 1  # Move to the next line if no patterns matched

    return metrics

def compute_averages(metrics):
    """
    Computes the average of each metric.

    Parameters:
    - metrics (dict): Dictionary containing lists of metrics.

    Returns:
    - dict: Dictionary containing average values of metrics.
    """
    averages = {}
    for key, values in metrics.items():
        if values:
            averages[key] = sum(values) / len(values)
        else:
            averages[key] = 0.0
    return averages

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract and compute average metrics from log files.")
    parser.add_argument('--log_file', type=str, help='Path to the log file.')
    parser.add_argument('--output', type=str, default=None, help='Path to save the results (optional).')

    args = parser.parse_args()

    # Parse the log file
    metrics = parse_logs(args.log_file)
    print(metrics)

    # Compute averages
    averages = compute_averages(metrics)

    # Prepare the output
    num_experiments = len(metrics['asr_before'])  # Assuming each experiment has a before defense ASR
    print(f"Total experiments parsed: {num_experiments}")

    if num_experiments > 0:
        print(f"Average Victim Precision: {averages['victim_precision']:.5f}")
        print(f"Average Victim Recall: {averages['victim_recall']:.5f}")
        print(f"Average Trigger Precision: {averages['trigger_precision']:.5f}")
        print(f"Average Trigger Recall: {averages['trigger_recall']:.5f}")
        print(f"Average ASR Before Defense: {averages['asr_before']:.5f}")
        print(f"Average CA Before Defense: {averages['ca_before']:.5f}")
        print(f"Average ASR After Defense: {averages['asr_after']:.5f}")
        print(f"Average CA After Defense: {averages['ca_after']:.5f}")
    else:
        print(f"Average Victim Precision: {averages['victim_precision']:.5f}")
        print(f"Average Victim Recall: {averages['victim_recall']:.5f}")
        print(f"Average Trigger Precision: {averages['trigger_precision']:.5f}")
        print(f"Average Trigger Recall: {averages['trigger_recall']:.5f}")
        print("No valid experiment data found in the log file.")

    # If output file is specified, write the results to the file
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as out_file:
                out_file.write(f"Total experiments parsed: {num_experiments}\n")
                if num_experiments > 0:
                    out_file.write(f"Average Victim Precision: {averages['victim_precision']:.5f}\n")
                    out_file.write(f"Average Victim Recall: {averages['victim_recall']:.5f}\n")
                    out_file.write(f"Average Trigger Precision: {averages['trigger_precision']:.5f}\n")
                    out_file.write(f"Average Trigger Recall: {averages['trigger_recall']:.5f}\n")
                    out_file.write(f"Average ASR Before Defense: {averages['asr_before']:.5f}\n")
                    out_file.write(f"Average CA Before Defense: {averages['ca_before']:.5f}\n")
                    out_file.write(f"Average ASR After Defense: {averages['asr_after']:.5f}\n")
                    out_file.write(f"Average CA After Defense: {averages['ca_after']:.5f}\n")
                else:
                    out_file.write("No valid experiment data found in the log file.\n")
            print(f"Results have been saved to '{args.output}'.")
        except Exception as e:
            print(f"Error while writing to the output file: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
