import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

# Load real errors
orig_errs = np.load('outputs/baseline_errors.npy')
refined_errs = np.load('outputs/refined_errors.npy')

plt.figure(figsize=(10, 6))
sns.kdeplot(orig_errs, color="red", fill=True, alpha=0.3, label=f"Baseline (MAE: {orig_errs.mean():.2f}°)")
sns.kdeplot(refined_errs, color="green", fill=True, alpha=0.3, label=f"Refined Pipeline (MAE: {refined_errs.mean():.2f}°)")

# Plot vertical lines for MAE
plt.axvline(x=refined_errs.mean(), color='green', linestyle='--', alpha=0.7)
plt.axvline(x=orig_errs.mean(), color='red', linestyle='--', alpha=0.7)

plt.xlim(0, 15)
plt.xlabel("Angle Error (Degrees)")
plt.ylabel("Density")
plt.title("Verified 5-Fold Error Density: Baseline vs. Refined Pipeline")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('outputs/error_kde_comparison.png', dpi=300)
print("Saved authentic KDE plot to outputs/error_kde_comparison.png")
