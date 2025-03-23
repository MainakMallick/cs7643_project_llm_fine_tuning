import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs to a dictionary of pandas DataFrames."""
    data = {}
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    for event_file in event_files:
        run_name = os.path.dirname(event_file).split('/')[-1]
        print(f"Processing run: {run_name}")
        
        # Load event data
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get available tags (metrics)
        tags = ea.Tags()['scalars']
        
        # Extract data for each tag
        for tag in tags:
            events = ea.Scalars(tag)
            
            # Convert to pandas DataFrame
            df = pd.DataFrame({
                'step': [e.step for e in events],
                'value': [e.value for e in events],
                'wall_time': [e.wall_time for e in events]
            })
            
            # Store in dictionary with tag as key
            if tag not in data:
                data[tag] = df
            else:
                data[tag] = pd.concat([data[tag], df])
    
    return data

def plot_training_metrics(data, save_dir='./plots'):
    """Create plots for each metric in the data."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot each metric
    for tag, df in data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['value'])
        plt.title(tag)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True)
        
        # Save the plot
        safe_tag = tag.replace('/', '_')
        plt.savefig(os.path.join(save_dir, f"{safe_tag}.png"))
        plt.close()
        
        print(f"Created plot for {tag}")

def create_training_report(data, save_dir='./plots'):
    """Create an HTML report with all training metrics."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Metrics Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .plot-container { margin-bottom: 30px; }
            img { max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Training Metrics Report</h1>
    """
    
    # Add each metric plot to the report
    for tag in data.keys():
        safe_tag = tag.replace('/', '_')
        html_content += f"""
        <div class="plot-container">
            <h2>{tag}</h2>
            <img src="{safe_tag}.png" alt="{tag}">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(os.path.join(save_dir, 'training_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML report at {os.path.join(save_dir, 'training_report.html')}")

def main():
    # Path to your TensorBoard logs
    log_dir = "./logs"
    
    # Extract data from TensorBoard logs
    data = extract_tensorboard_data(log_dir)
    
    if not data:
        print("No TensorBoard data found!")
        return
    
    # Create plots
    plot_training_metrics(data)
    
    # Create HTML report
    create_training_report(data)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()