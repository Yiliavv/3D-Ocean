import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os
import glob

from src.config.params import PROJECT_PATH

# Set style to match the professional appearance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Configure font settings
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['font.weight'] = 'normal'

# Import additional modules
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.ticker import LogFormatterMathtext
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
plt.ioff()  # Turn off interactive mode, don't display plots

def ensure_output_dir():
    """Ensure output directory exists"""
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def scan_training_files():
    """Scan training output directories to find all available files"""
    base_dir = f"{PROJECT_PATH}/train_output"
    models = ['LSTM', 'ConvLSTM', 'Transformer']
    available_files = {}
    
    print("🔍 Scanning training output directories...")
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        if os.path.exists(model_dir):
            json_files = glob.glob(os.path.join(model_dir, '*.json'))
            available_files[model] = []
            
            for file_path in json_files:
                filename = os.path.basename(file_path)
                uid = filename.replace('.json', '')
                available_files[model].append(uid)
                
            print(f"  {model}: {len(available_files[model])} files found")
            for uid in available_files[model]:
                print(f"    - {uid}")
        else:
            available_files[model] = []
            print(f"  {model}: Directory not found")
    
    return available_files

def get_common_uids(available_files):
    """Find UIDs that are common across multiple models"""
    all_uids = set()
    for model_uids in available_files.values():
        all_uids.update(model_uids)
    
    uid_counts = {}
    for uid in all_uids:
        count = sum(1 for model_uids in available_files.values() if uid in model_uids)
        uid_counts[uid] = count
    
    # Sort by number of models that have this UID (descending)
    sorted_uids = sorted(uid_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Found {len(all_uids)} unique parameter sets:")
    for uid, count in sorted_uids:
        models_with_uid = [model for model, uids in available_files.items() if uid in uids]
        print(f"  {uid[:8]}... ({count}/3 models): {', '.join(models_with_uid)}")
    
    return sorted_uids

def load_training_data(uid, available_files=None):
    """Load training data from JSON files for available models"""
    if available_files is None:
        available_files = scan_training_files()
    
    models = ['Transformer', 'ConvLSTM', 'LSTM']
    data = {}
    
    print(f"\n📂 Loading data for parameter set: {uid[:8]}...")
    
    for model in models:
        if uid in available_files.get(model, []):
            base_dir = f"{PROJECT_PATH}/train_output"
            file_path = os.path.join(base_dir, model, f'{uid}.json')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                    data[model] = model_data
                    print(f"  ✅ {model}: {len(model_data.get('val_loss', []))} training records loaded")
            except Exception as e:
                print(f"  ❌ {model}: Failed to load - {e}")
        else:
            print(f"  ⚠️  {model}: File not available")
    
    return data

def load_all_training_data():
    """Load all available training data for cross-parameter analysis"""
    available_files = scan_training_files()
    common_uids = get_common_uids(available_files)
    
    if not common_uids:
        print("❌ No training files found!")
        return {}, []
    
    all_data = {}
    uids = []
    
    # Load data for UIDs that have the most models available
    max_uids_to_load = min(3, len(common_uids))  # Load up to 3 parameter sets
    
    for uid, model_count in common_uids[:max_uids_to_load]:
        if model_count >= 1:  # At least one model has data for this UID
            try:
                data = load_training_data(uid, available_files)
                if data:  # Only add if we successfully loaded some data
                    all_data[uid] = data
                    uids.append(uid)
                    print(f"✅ Successfully loaded parameter set {uid[:8]}... with {len(data)} model(s)")
                else:
                    print(f"⚠️  No data loaded for parameter set {uid[:8]}...")
            except Exception as e:
                print(f"❌ Failed to load parameter set {uid[:8]}...: {e}")
    
    print(f"\n📊 Total parameter sets loaded: {len(uids)}")
    return all_data, uids

def get_first_available_uid():
    """Get the first available UID for single analysis"""
    available_files = scan_training_files()
    common_uids = get_common_uids(available_files)
    
    if not common_uids:
        print("❌ No training files found!")
        return None
    
    # Return the UID with the most models available
    return common_uids[0][0]

def process_loss_data(loss_data, epochs=150):
    """Process loss data to get per-epoch averages"""
    entries_per_epoch = len(loss_data) // epochs
    epoch_losses = []
    
    for i in range(epochs):
        start_idx = i * entries_per_epoch
        end_idx = (i + 1) * entries_per_epoch
        if end_idx > len(loss_data):
            end_idx = len(loss_data)
        
        epoch_loss = np.mean(loss_data[start_idx:end_idx])
        epoch_losses.append(epoch_loss)
    
    return np.array(epoch_losses)

def smooth_curve(y, factor=0.9):
    """Apply exponential smoothing to curves"""
    smoothed = [y[0]]
    for i in range(1, len(y)):
        smoothed.append(smoothed[-1] * factor + y[i] * (1 - factor))
    return smoothed

def calculate_advanced_metrics(val_loss, train_loss):
    """Calculate advanced performance metrics"""
    metrics = {}
    
    # Final performance
    metrics['final_val_loss'] = np.mean(val_loss[-10:])
    metrics['final_train_loss'] = np.mean(train_loss[-10:])
    
    # Overfitting indicator
    metrics['overfitting'] = metrics['final_val_loss'] - metrics['final_train_loss']
    
    # Convergence rate
    early_loss = np.mean(val_loss[10:20])
    late_loss = np.mean(val_loss[-20:-10])
    metrics['convergence_rate'] = (early_loss - late_loss) / early_loss * 100
    
    # Stability (based on coefficient of variation in last 30 epochs)
    final_30_std = np.std(val_loss[-30:])
    final_30_mean = np.mean(val_loss[-30:])
    metrics['stability'] = final_30_mean / (final_30_std + 1e-8)
    
    # Best performance point
    metrics['best_val_loss'] = np.min(val_loss)
    metrics['best_epoch'] = np.argmin(val_loss) + 1
    
    # Early stopping indicator (consecutive epochs without improvement)
    best_loss = np.inf
    no_improve_count = 0
    max_no_improve = 0
    for loss in val_loss:
        if loss < best_loss:
            best_loss = loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            max_no_improve = max(max_no_improve, no_improve_count)
    metrics['max_plateau'] = max_no_improve
    
    # Training efficiency (epochs needed to reach threshold)
    threshold = metrics['final_val_loss'] * 1.1
    efficiency_epoch = len(val_loss)
    for i, loss in enumerate(val_loss):
        if loss <= threshold:
            efficiency_epoch = i + 1
            break
    metrics['training_efficiency'] = efficiency_epoch
    
    # Learning curve slope
    if len(val_loss) >= 20:
        recent_slope = np.polyfit(range(len(val_loss)-20, len(val_loss)), val_loss[-20:], 1)[0]
        metrics['learning_slope'] = recent_slope
    else:
        metrics['learning_slope'] = 0
    
    return metrics

def analyze_cross_parameter_performance():
    """Analyze model performance across different training parameters - Optimized Version"""
    all_data, uids = load_all_training_data()
    
    # Check if we have enough data for comparison
    if len(uids) < 1:
        print("❌ No parameter sets available for analysis")
        return {}
    
    print(f"📊 Found {len(uids)} parameter set(s) for analysis")
    
    # Create optimized figure with better layout - now 2x2 grid without convergence chart
    fig = plt.figure(figsize=(20, 12), dpi=300)
    
    # Adjust layout - remove convergence chart, use 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1, 1.2])
    comparison_mode = len(uids) > 1
    
    colors = {'Transformer': '#2E86C1', 'ConvLSTM': '#E74C3C', 'LSTM': '#F39C12'}
    models = ['LSTM', 'ConvLSTM', 'Transformer']
    
    # Calculate all metrics
    all_metrics = {}
    for uid in uids:
        all_metrics[uid] = {}
        for model in models:
            if model in all_data[uid]:  # Check if model data exists for this UID
                data = all_data[uid][model]
                val_loss = process_loss_data(data['val_loss'], epochs=data['epoch'])
                train_loss = process_loss_data(data['train_loss'], epochs=data['epoch'])
                all_metrics[uid][model] = calculate_advanced_metrics(val_loss, train_loss)
    
    # 1. Performance Overview - Top Left
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(models))
    width = 0.35 if comparison_mode else 0.6
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for i, uid in enumerate(uids):
        uid_losses = []
        for model in models:
            if model in all_metrics[uid]:
                uid_losses.append(all_metrics[uid][model]['final_val_loss'])
            else:
                uid_losses.append(float('inf'))  # Use inf for missing data
        plot_data.append(uid_losses)
        labels.append(f'Parameter Set {i+1}')
    
    # Plot bars
    if comparison_mode and len(plot_data) >= 2:
        bars1 = ax1.bar(x - width/2, plot_data[0], width, 
                       label=labels[0], color=[colors[m] for m in models], 
                       alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax1.bar(x + width/2, plot_data[1], width,
                       label=labels[1], color=[colors[m] for m in models], 
                       alpha=0.6, edgecolor='white', linewidth=1, hatch='///')
        
        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if plot_data[0][i] != float('inf'):
                label1 = f'{plot_data[0][i]:.3f}' if plot_data[0][i] <= 1 else f'{plot_data[0][i]:.2f}'
                ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() * 1.05,
                        label1, ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            if plot_data[1][i] != float('inf'):
                label2 = f'{plot_data[1][i]:.3f}' if plot_data[1][i] <= 1 else f'{plot_data[1][i]:.2f}'
                ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() * 1.05,
                        label2, ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        # Single parameter set
        bars = ax1.bar(x, plot_data[0], width, 
                      label=labels[0], color=[colors[m] for m in models], 
                      alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            if plot_data[0][i] != float('inf'):
                label = f'{plot_data[0][i]:.3f}' if plot_data[0][i] <= 1 else f'{plot_data[0][i]:.2f}'
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                        label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Final Validation Loss', fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Training Dynamics - Top Right
    ax2 = fig.add_subplot(gs[0, 1])
    
    for i, uid in enumerate(uids):
        for model in models:
            if model in all_data[uid]:
                data = all_data[uid][model]
                val_loss = process_loss_data(data['val_loss'], epochs=data['epoch'])
                epochs_range = range(1, len(val_loss) + 1)
                
                linestyle = '-' if i == 0 else '--'
                alpha = 0.9 if i == 0 else 0.7
                linewidth = 2.5 if i == 0 else 2
                
                ax2.plot(epochs_range, smooth_curve(val_loss, 0.95), 
                        color=colors[model], linestyle=linestyle, 
                        alpha=alpha, linewidth=linewidth)
    
    # Add legend with cleaner format
    legend_elements = []
    for model in models:
        legend_elements.append(plt.Line2D([0], [0], color=colors[model], lw=2.5, label=model))
    
    if comparison_mode and len(uids) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label='Param Set 1'))
        legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Param Set 2'))
    
    ax2.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.set_xlabel('Training Epochs', fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontweight='bold')
    ax2.set_title('Training Dynamics', fontweight='bold', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Comprehensive Performance Matrix (Bottom - spans both columns)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create comprehensive performance matrix
    metric_names = ['Final Loss', 'Convergence', 'Efficiency', 'Stability', 'Best Loss']
    
    # Prepare data matrix
    performance_matrix = []
    row_labels = []
    
    for i, uid in enumerate(uids):
        for model in models:
            if model in all_metrics[uid]:
                m = all_metrics[uid][model]
                
                # Normalize metrics for better visualization (0-1 scale, higher is better)
                final_loss_norm = 1 / (1 + m['final_val_loss'])  # Invert: lower loss = better
                convergence_norm = max(0, min(1, abs(m['convergence_rate']) / 100))  # 0-100% -> 0-1
                efficiency_norm = max(0, min(1, (300 - m['training_efficiency']) / 300))  # Invert: fewer epochs = better
                stability_norm = min(1, m['stability'] / 100)  # Cap at 1
                best_loss_norm = 1 / (1 + m['best_val_loss'])  # Invert: lower loss = better
                
                performance_matrix.append([
                    final_loss_norm,
                    convergence_norm, 
                    efficiency_norm,
                    stability_norm,
                    best_loss_norm
                ])
                
                row_labels.append(f'{model}\nSet {i+1}')
    
    if not performance_matrix:
        # Handle case where no data is available
        ax4.text(0.5, 0.5, 'No performance data available', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=16, fontweight='bold')
        ax4.set_title('Performance Matrix - No Data Available', fontweight='bold', fontsize=16)
    else:
        performance_matrix = np.array(performance_matrix)
        
        # Create enhanced heatmap without colorbar
        im = ax4.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        
        # Customize the heatmap
        ax4.set_xticks(range(len(metric_names)))
        ax4.set_xticklabels(metric_names, fontweight='bold', fontsize=12)
        ax4.set_yticks(range(len(row_labels)))
        ax4.set_yticklabels(row_labels, fontweight='bold', fontsize=11)
        
        # Add performance scores as text with uniform white background and black text
        for i in range(len(row_labels)):
            for j in range(len(metric_names)):
                score = performance_matrix[i, j]
                text = f'{score:.2f}'
                    
                ax4.text(j, i, text, ha="center", va="center", 
                        color='black', fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                 alpha=0.9, edgecolor='gray', linewidth=0.5))
        
        ax4.set_title('Comprehensive Performance Matrix (Normalized Scores)', 
                     fontweight='bold', fontsize=16, pad=20)
        
        # Remove colorbar completely
        
        # Add grid lines for better readability
        ax4.set_xticks(np.arange(-0.5, len(metric_names), 1), minor=True)
        ax4.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax4.grid(which='minor', color='white', linestyle='-', linewidth=2, alpha=0.8)
        ax4.tick_params(which='minor', size=0)
        
        # Style the borders
        for spine in ax4.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
        
        # Remove summary statistics box - commented out
        # textstr = 'Best Overall Performance:\n'
        # best_scores = {}
        # for i, uid in enumerate(uids):
        #     available_models = [m for m in models if m in all_metrics[uid]]
        #     if available_models:
        #         best_model = min(available_models, key=lambda m: all_metrics[uid][m]['final_val_loss'])
        #         best_scores[f'Set {i+1}'] = f'{best_model} ({all_metrics[uid][best_model]["final_val_loss"]:.3f})'
        # 
        # for set_name, best_info in best_scores.items():
        #     textstr += f'{set_name}: {best_info}\n'
        # 
        # # Position summary box to avoid overlapping with the matrix
        # props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy', linewidth=1)
        # # Place at top right, outside the main matrix area
        # ax4.text(1.02, 0.98, textstr, transform=ax4.transAxes, fontsize=10,
        #         verticalalignment='top', horizontalalignment='left', bbox=props, fontweight='bold')
    
    # Overall styling
    fig.patch.set_facecolor('#f8f9fa')
    
    title_suffix = "Comprehensive Evaluation" if comparison_mode else "Single Parameter Set Analysis"
    plt.suptitle(f'Sea Surface Temperature Model Performance Analysis\nCross-Parameter {title_suffix}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save optimized image
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, 'cross_parameter_analysis_optimized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f"Optimized cross-parameter analysis saved to: {output_path}")
    
    plt.close()  # Close chart without displaying
    
    return all_metrics

def analyze_model_performance(uid=None):
    """Main analysis function - now accepts optional UID parameter"""
    if uid is None:
        uid = get_first_available_uid()
        if uid is None:
            print("❌ No training data available for analysis!")
            return None, None, None, None
    
    print(f"\n🔬 Starting analysis for parameter set: {uid[:8]}...")
    
    # Load data
    data = load_training_data(uid)
    
    if not data:
        print(f"❌ No data loaded for UID {uid[:8]}...")
        return None, None, None, None
    
    # Create figure with subplots matching reference style
    fig = plt.figure(figsize=(24, 16), dpi=300)
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)
    
    # Enhanced color scheme with better contrast
    colors = {
        'Transformer': '#1f77b4',  # Professional blue
        'ConvLSTM': '#d62728',     # Deep red
        'LSTM': '#ff7f0e'          # Orange
    }
    
    # English labels mapping
    english_labels = {
        'Transformer': 'Transformer',
        'ConvLSTM': 'ConvLSTM', 
        'LSTM': 'LSTM'
    }
    
    epochs = 150
    epoch_range = range(1, epochs + 1)
    
    # Process data for all available models
    processed_data = {}
    for model_name, model_data in data.items():
        val_loss_per_epoch = process_loss_data(model_data['val_loss'], epochs)
        train_loss_per_epoch = process_loss_data(model_data['train_loss'], epochs)
        processed_data[model_name] = {
            'val_loss': val_loss_per_epoch,
            'train_loss': train_loss_per_epoch,
            'batch_size': model_data['batch_size'],
            'epochs': epochs
        }
    
    if not processed_data:
        print(f"❌ No processed data available for UID {uid[:8]}...")
        return None, None, None, None
    
    # 1. Training Loss Curves (Top Left)
    ax1 = fig.add_subplot(gs[0, :2])
    
    for model_name, model_data in processed_data.items():
        val_loss = model_data['val_loss']
        train_loss = model_data['train_loss']
        
        # Plot validation loss
        smoothed_val = smooth_curve(val_loss, factor=0.95)
        ax1.plot(epoch_range, val_loss, alpha=0.15, color=colors[model_name], linewidth=0.8)
        ax1.plot(epoch_range, smoothed_val, label=f'{english_labels[model_name]} (Validation)', 
                color=colors[model_name], linewidth=3.5, alpha=0.9)
        
        # Plot training loss with dashed line
        smoothed_train = smooth_curve(train_loss, factor=0.95)
        ax1.plot(epoch_range, smoothed_train, label=f'{english_labels[model_name]} (Training)', 
                color=colors[model_name], linewidth=2.8, linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Training Epochs', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=15, fontweight='bold')
    ax1.set_title('Training Loss vs Validation Loss Comparison', fontsize=17, fontweight='bold', pad=30)
    
    # Enhanced legend
    legend1 = ax1.legend(frameon=True, fancybox=True, shadow=True, ncol=2, 
                        fontsize=13, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.95)
    legend1.get_frame().set_edgecolor('gray')
    legend1.get_frame().set_linewidth(0.8)
    
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
    ax1.set_yscale('log')
    ax1.set_xlim(1, epochs)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Loss Distribution Heatmap (Top Right)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    segment_size = 10
    n_segments = epochs // segment_size
    model_list = list(processed_data.keys())
    heatmap_data = np.zeros((len(model_list), n_segments))
    
    for i, model_name in enumerate(model_list):
        val_loss = processed_data[model_name]['val_loss']
        for j in range(n_segments):
            start_idx = j * segment_size
            end_idx = (j + 1) * segment_size
            segment_loss = np.mean(val_loss[start_idx:end_idx])
            heatmap_data[i, j] = segment_loss
    
    heatmap_data_log = np.log10(heatmap_data + 1e-8)
    
    heatmap_data_normalized = np.zeros_like(heatmap_data_log)
    for i in range(heatmap_data_log.shape[0]):
        row_min = np.min(heatmap_data_log[i, :])
        row_max = np.max(heatmap_data_log[i, :])
        if row_max > row_min:
            heatmap_data_normalized[i, :] = (heatmap_data_log[i, :] - row_min) / (row_max - row_min)
        else:
            heatmap_data_normalized[i, :] = 0.5
    
    im = ax2.imshow(heatmap_data_normalized, cmap='plasma', aspect='auto', 
                   interpolation='bilinear', vmin=0, vmax=1)
    
    for i in range(len(model_list)):
        for j in range(n_segments):
            loss_val = heatmap_data[i, j]
            if loss_val > 1:
                text = f'{loss_val:.1f}'
            elif loss_val > 0.1:
                text = f'{loss_val:.2f}'
            else:
                text = f'{loss_val:.3f}'
            
            text_color = 'white' if heatmap_data_normalized[i, j] > 0.5 else 'black'
            ax2.text(j, i, text, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color=text_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             alpha=0.7, edgecolor='none'))
    
    ax2.set_xlabel('Training Stage (Epoch Groups)', fontsize=15, fontweight='bold')
    ax2.set_title('Validation Loss Distribution During Training\n(Normalized by Model for Better Contrast)', 
                 fontsize=17, fontweight='bold', pad=30)
    
    ax2.set_yticks(range(len(model_list)))
    ax2.set_yticklabels([english_labels[model] for model in model_list], fontsize=13)
    ax2.set_xticks(range(n_segments))
    ax2.set_xticklabels([f'{i*segment_size+1}-{(i+1)*segment_size}' for i in range(n_segments)], 
                       rotation=45, fontsize=11, ha='right')
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Relative Loss Intensity\n(Normalized per Model)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Lowest', 'Low', 'Medium', 'High', 'Highest'])
    
    ax2.set_xticks(np.arange(-0.5, n_segments, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(model_list), 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=1.5, alpha=0.8)
    ax2.tick_params(which='minor', size=0)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    # 3. Final Performance Comparison (Bottom Left)
    ax3 = fig.add_subplot(gs[1, :2])
    
    final_val_losses = {}
    final_train_losses = {}
    convergence_rates = {}
    
    for model_name, model_data in processed_data.items():
        val_loss = model_data['val_loss']
        train_loss = model_data['train_loss']
        
        final_val_losses[model_name] = np.mean(val_loss[-10:])
        final_train_losses[model_name] = np.mean(train_loss[-10:])
        
        early_loss = np.mean(val_loss[10:20])
        late_loss = np.mean(val_loss[-20:-10])
        convergence_rates[model_name] = (early_loss - late_loss) / early_loss * 100
    
    models = list(final_val_losses.keys())
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, [final_train_losses[m] for m in models], width, 
                   label='Training Loss', color=[colors[m] for m in models], alpha=0.7, 
                   edgecolor='black', linewidth=1.5, hatch='//')
    bars2 = ax3.bar(x + width/2, [final_val_losses[m] for m in models], width,
                   label='Validation Loss', color=[colors[m] for m in models], alpha=0.95, 
                   edgecolor='black', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.03,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                             edgecolor='gray', linewidth=0.8))
    
    ax3.set_ylabel('Final Loss Value (Last 10 Epochs)', fontsize=15, fontweight='bold')
    ax3.set_title('Model Final Performance Comparison', fontsize=17, fontweight='bold', pad=30)
    ax3.set_xticks(x)
    ax3.set_xticklabels([english_labels[m] for m in models], fontsize=13)
    
    legend3 = ax3.legend(fontsize=13, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                        frameon=True, fancybox=True, shadow=True)
    legend3.get_frame().set_facecolor('white')
    legend3.get_frame().set_alpha(0.95)
    legend3.get_frame().set_edgecolor('gray')
    legend3.get_frame().set_linewidth(0.8)
    
    ax3.set_ylim(0, max(max(final_val_losses.values()), max(final_train_losses.values())) * 1.4)
    
    ax3.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.6)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Convergence Analysis (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 2:])
    
    stability_scores = {}
    overfitting_scores = {}
    
    for model_name, model_data in processed_data.items():
        val_loss = model_data['val_loss']
        train_loss = model_data['train_loss']
        
        final_30_std = np.std(val_loss[-30:])
        final_30_mean = np.mean(val_loss[-30:])
        stability_scores[model_name] = final_30_mean / (final_30_std + 1e-8)
        
        overfitting_scores[model_name] = final_val_losses[model_name] - final_train_losses[model_name]
    
    conv_rates = [convergence_rates[m] for m in models]
    stab_scores = [stability_scores[m] for m in models]
    
    scatter = ax4.scatter(conv_rates, stab_scores, 
                         c=[colors[m] for m in models], 
                         s=500, alpha=0.8, edgecolors='black', linewidth=2.8)
    
    for i, model in enumerate(models):
        ax4.annotate(english_labels[model], (conv_rates[i], stab_scores[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                             edgecolor='gray', linewidth=1.0),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                   color='gray', alpha=0.7))
    
    ax4.set_xlabel('Convergence Rate (%)', fontsize=15, fontweight='bold')
    ax4.set_ylabel('Stability Score', fontsize=15, fontweight='bold')
    ax4.set_title('Model Convergence vs Stability Analysis', fontsize=17, fontweight='bold', pad=30)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.6)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    fig.patch.set_facecolor('#fafafa')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save image to output folder
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, f'model_performance_analysis_{uid[:8]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    print(f"Analysis chart saved to: {output_path}")
    
    plt.close()  # Close chart without displaying
    
    return processed_data, final_val_losses, convergence_rates, stability_scores

if __name__ == "__main__":
    print("Starting Sea Surface Temperature Model Training Analysis...")
    print("=" * 60)
    
    # 1. Cross-parameter analysis
    print("\n🔬 Executing Cross-Parameter Analysis")
    print("-" * 40)
    cross_metrics = analyze_cross_parameter_performance()
    
    # 2. Get available UIDs and analyze them
    available_files = scan_training_files()
    common_uids = get_common_uids(available_files)
    
    if not common_uids:
        print("❌ No training data files found! Please check the train_output directory.")
    else:
        # Analyze the top UIDs (those with most models available)
        max_analyses = min(3, len(common_uids))  # Analyze up to 3 parameter sets
        
        analysis_results = []
        
        for i, (uid, model_count) in enumerate(common_uids[:max_analyses]):
            print(f"\n🎯 Analyzing Parameter Set {i+1}: {uid[:8]}...")
            print(f"   Available models: {model_count}/3")
            print("-" * 50)
            
            try:
                result = analyze_model_performance(uid)
                if result[0] is not None:  # Check if analysis was successful
                    analysis_results.append((uid, result))
                    
                    processed_data, final_val_losses, convergence_rates, stability_scores = result
                    
                    print(f"\n✅ Analysis completed for {uid[:8]}...")
                    if final_val_losses:
                        best_model = min(final_val_losses, key=final_val_losses.get)
                        print(f"   Best Model (Lowest Validation Loss): {best_model}")
                        print("   Final Validation Losses:")
                        for model, loss in final_val_losses.items():
                            print(f"     {model}: {loss:.4f}")
                else:
                    print(f"⚠️  Analysis failed for {uid[:8]}...")
                    
            except Exception as e:
                print(f"❌ Error analyzing {uid[:8]}...: {e}")
        
        # Summary
        print(f"\n📊 Analysis Summary")
        print("=" * 50)
        
        if cross_metrics:
            print(f"Cross-Parameter Analysis: ✅ Completed")
            print(f"Parameter sets analyzed: {len(cross_metrics)}")
            
            for uid in cross_metrics.keys():
                print(f"\n📈 Parameter Set {uid[:8]}...:")
                available_models = list(cross_metrics[uid].keys())
                for model in available_models:
                    m = cross_metrics[uid][model]
                    print(f"  {model}:")
                    print(f"    Final Validation Loss: {m['final_val_loss']:.4f}")
                    print(f"    Convergence Rate: {m['convergence_rate']:.2f}%")
                    print(f"    Training Efficiency: {m['training_efficiency']} epochs")
                    print(f"    Overfitting Degree: {m['overfitting']:.4f}")
        
        print(f"\n🎉 Analysis completed! Generated files:")
        print(f"   - output/cross_parameter_analysis_optimized.png")
        for uid, _ in analysis_results:
            print(f"   - output/model_performance_analysis_{uid[:8]}.png") 