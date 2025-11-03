import csv
import re

# Parse the original trace
original_data = []
with open('integration_tests/out/original_trace.txt', 'r') as f:
    content = f.read()
    
    # Split by time steps
    time_steps = content.split('==== Time step')
    
    for step in time_steps[1:]:  # Skip first empty element
        if not step.strip():
            continue
            
        lines = step.strip().split('\n')
        step_num = int(lines[0].split()[0])
        
        data = {'time_step': step_num}
        
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = float(value.strip())
                data[key] = value
        
        # Calculate internal state from the original
        if 'output_state_(o)' in data and 'internal_state_(o)' in data:
            data['internal_state_e'] = data['internal_state_(o)']
            data['output_state_o'] = data['output_state_(o)']
        
        original_data.append(data)

# Parse CSV data
current_data = []
with open('integration_tests/out/pyuri_dynamics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['time_step'] = int(row['time_step'])
        for key in row:
            if key != 'time_step':
                try:
                    row[key] = float(row[key])
                except:
                    pass
        current_data.append(row)

# Compare the traces
print("=" * 80)
print("TRACE COMPARISON ANALYSIS")
print("=" * 80)

# Find divergences
divergences = []
for orig, curr in zip(original_data, current_data):
    t = orig['time_step']
    
    # Compare key metrics (with some tolerance for floating point differences)
    orig_int = orig.get('internal_state_e', 0)
    curr_int = curr.get('internal_state_E', 0)
    
    orig_out = orig.get('output_state_o', 0)
    curr_out = curr.get('output_state_O', 0)
    
    # Check if diverged significantly (> 0.1 difference)
    if abs(orig_int - curr_int) > 0.1 or abs(orig_out - curr_out) > 0.1:
        divergences.append({
            'time_step': t,
            'orig_internal': orig_int,
            'curr_internal': curr_int,
            'orig_output': orig_out,
            'curr_output': curr_out,
            'difference': abs(orig_int - curr_int)
        })

print(f"\nFound {len(divergences)} time steps with significant divergence (>0.1)")
print("\nFirst 20 divergences:")
print("-" * 80)
print(f"{'Step':<6} {'Orig E':<12} {'Curr E':<12} {'Orig O':<12} {'Curr O':<12}")
print("-" * 80)

for d in divergences[:20]:
    print(f"{d['time_step']:<6} {d['orig_internal']:>10.2f}  {d['curr_internal']:>10.2f}  {d['orig_output']:>10.2f}  {d['curr_output']:>10.2f}")

if len(divergences) > 20:
    print(f"... and {len(divergences) - 20} more")

# Analyze specific time windows
print("\n" + "=" * 80)
print("KEY TIME WINDOW ANALYSIS")
print("=" * 80)

# First 10 steps (inhibitory)
print("\nTime steps 0-9 (Inhibitory input):")
print(f"Original E: {[d['internal_state_e'] for d in original_data[0:10]]}")
print(f"Current E:   {[d['internal_state_E'] for d in current_data[0:10]]}")

# Steps 10-19 (excitatory)
print("\nTime steps 10-19 (Excitatory input):")
print(f"Original E: {[d['internal_state_e'] for d in original_data[10:20]]}")
print(f"Current E:   {[d['internal_state_E'] for d in current_data[10:20]]}")

# Steps 20-29 (gap junction)
print("\nTime steps 20-29 (Gap junction input):")
print(f"Original E: {[d['internal_state_e'] for d in original_data[20:30]]}")
print(f"Current E:   {[d['internal_state_E'] for d in current_data[20:30]]}")

# Look at step 20 specifically
if len(original_data) > 20 and len(current_data) > 20:
    print("\nStep 20 detailed comparison:")
    orig_20 = original_data[20]
    curr_20 = current_data[20]
    print(f"Original - gj inf: {orig_20.get('gj_inf', 'N/A')}, stimulus: {orig_20.get('current_stimulus_(instate_+_ij)', 'N/A')}, E: {orig_20.get('internal_state_e', 'N/A')}")
    print(f"Current  - gj inf: {curr_20.get('gj_inf', 'N/A')}, stimulus: {curr_20.get('current_stimulus', 'N/A')}, E: {curr_20.get('internal_state_E', 'N/A')}")

