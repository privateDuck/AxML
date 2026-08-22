import numpy as np
import matplotlib.pyplot as plt
import os

def load_cpp_predictions(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as f:
        data = f.read().strip().split(',')
        return np.array([int(x) for x in data])

def plot_side_by_side(config_id, xx, yy, X_train, y_train, sk_preds, cpp_preds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Decision Boundary: {config_id}', fontsize=16)

    # Plot Scikit-learn
    sk_Z = sk_preds.reshape(xx.shape)
    ax1.contourf(xx, yy, sk_Z, alpha=0.4, cmap='viridis')
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', cmap='viridis')
    ax1.set_title('Scikit-Learn')

    # Plot C++
    if cpp_preds is not None:
        cpp_Z = cpp_preds.reshape(xx.shape)
        ax2.contourf(xx, yy, cpp_Z, alpha=0.4, cmap='viridis')
        ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', cmap='viridis')

        match_pct = np.mean(sk_preds == cpp_preds) * 100
        ax2.set_title(f'AxML (Agreement: {match_pct:.1f}%)')
    else:
        ax2.set_title('AxML (Missing CSV Data)')
        ax2.text(0.5, 0.5, "Awaiting C++ CSV", ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(f'plots/boundary_{config_id}.png')
    plt.close() # Close to prevent memory warnings if lots of configs

def main():
    data = np.load("sklearn_boundary_data.npz")
    xx, yy = data['xx'], data['yy']
    X_train, y_train = data['X_train'], data['y_train']

    # Get all keys except the structural data
    config_ids = [k for k in data.files if k not in ['xx', 'yy', 'X_train', 'y_train']]

    for config_id in config_ids:
        sk_preds = data[config_id]
        csv_filename = f"data/cpp_preds_{config_id}.csv"
        cpp_preds = load_cpp_predictions(csv_filename)

        print(f"Generating plot for {config_id}...")
        plot_side_by_side(config_id, xx, yy, X_train, y_train, sk_preds, cpp_preds)

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()