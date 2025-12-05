import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_customers = 200

customer_id = range(1, n_customers + 1)
gender = np.random.choice(['Male', 'Female'], n_customers, p=[0.44, 0.56])
age = np.random.randint(18, 70, n_customers)

cluster_centers = [
    (25, 20),
    (25, 80),
    (85, 20),
    (85, 85),
    (55, 50)
]

annual_income = []
spending_score = []

for i in range(n_customers):
    center = cluster_centers[i % 5]
    income = center[0] + np.random.normal(0, 10)
    spending = center[1] + np.random.normal(0, 10)
    annual_income.append(np.clip(income, 15, 137))
    spending_score.append(np.clip(spending, 1, 99))

df = pd.DataFrame({
    'CustomerID': customer_id,
    'Gender': gender,
    'Age': age,
    'Annual Income (k$)': annual_income,
    'Spending Score (1-100)': spending_score
})

print("=" * 70)
print("Q7: MALL CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
print("=" * 70)

print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print(df.head(10))

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Gender Distribution ---")
print(df['Gender'].value_counts())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 70)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 70)

K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))

print("\n--- Elbow Method Results ---")
elbow_df = pd.DataFrame({
    'K': list(K_range),
    'Inertia': inertias,
    'Silhouette Score': silhouette_scores
})
print(elbow_df.to_string(index=False))

optimal_k = 5
print(f"\nOptimal K selected: {optimal_k}")

print("\n" + "=" * 70)
print(f"K-MEANS CLUSTERING WITH K={optimal_k}")
print("=" * 70)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

print("\n--- Cluster Centroids (Original Scale) ---")
centroids_df = pd.DataFrame(centroids_original, 
                            columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centroids_df['Cluster'] = range(optimal_k)
centroids_df = centroids_df[['Cluster', 'Annual Income (k$)', 'Spending Score (1-100)']]
print(centroids_df.to_string(index=False))

print("\n--- Cluster Statistics ---")
cluster_stats = df.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Age': 'mean'
}).round(2)
cluster_stats.columns = ['Count', 'Income_Mean', 'Income_Std', 'Spending_Mean', 'Spending_Std', 'Age_Mean']
print(cluster_stats)

final_silhouette = silhouette_score(X_scaled, df['Cluster'])
print(f"\nFinal Silhouette Score: {final_silhouette:.4f}")

print("\n" + "=" * 70)
print("CLUSTER INTERPRETATION")
print("=" * 70)

cluster_labels = {}
for cluster in range(optimal_k):
    income = centroids_original[cluster][0]
    spending = centroids_original[cluster][1]
    
    if income < 40 and spending < 40:
        label = "Low Income, Low Spending (Careful)"
        description = "Budget-conscious customers, price-sensitive"
    elif income < 40 and spending > 60:
        label = "Low Income, High Spending (Careless)"
        description = "Overspenders, potential credit risks"
    elif income > 70 and spending < 40:
        label = "High Income, Low Spending (Sensible)"
        description = "Wealthy savers, quality over quantity"
    elif income > 70 and spending > 60:
        label = "High Income, High Spending (Target)"
        description = "Premium customers, ideal for luxury marketing"
    else:
        label = "Average Income, Average Spending (Standard)"
        description = "Mainstream customers, general marketing"
    
    cluster_labels[cluster] = {'label': label, 'description': description}
    print(f"\nCluster {cluster}: {label}")
    print(f"  Income: ${income:.1f}k, Spending Score: {spending:.1f}")
    print(f"  Description: {description}")
    print(f"  Customer Count: {len(df[df['Cluster'] == cluster])}")

df['Cluster_Label'] = df['Cluster'].map(lambda x: cluster_labels[x]['label'])

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax1 = axes[0, 0]
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax3.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                c=colors[cluster], label=f'Cluster {cluster}', s=100, alpha=0.7, edgecolors='black')

ax3.scatter(centroids_original[:, 0], centroids_original[:, 1],
            c='black', marker='X', s=300, linewidths=2, label='Centroids')

ax3.set_xlabel('Annual Income (k$)', fontsize=12)
ax3.set_ylabel('Spending Score (1-100)', fontsize=12)
ax3.set_title('Customer Segments', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
cluster_counts = df['Cluster'].value_counts().sort_index()
bars = ax4.bar(range(optimal_k), cluster_counts.values, color=colors, edgecolor='black', linewidth=2)
ax4.set_xlabel('Cluster', fontsize=12)
ax4.set_ylabel('Number of Customers', fontsize=12)
ax4.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(range(optimal_k))

for bar, count in zip(bars, cluster_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('q7_kmeans_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(14, 10))

for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    ax.scatter(cluster_data['Annual Income (k$)'], 
               cluster_data['Spending Score (1-100)'],
               c=colors[cluster], s=150, alpha=0.7, edgecolors='black', linewidth=1.5)

ax.scatter(centroids_original[:, 0], centroids_original[:, 1],
           c='black', marker='X', s=400, linewidths=3, zorder=5)

for cluster in range(optimal_k):
    ax.annotate(cluster_labels[cluster]['label'].split('(')[1].replace(')', ''),
                (centroids_original[cluster, 0], centroids_original[cluster, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Annual Income (k$)', fontsize=14)
ax.set_ylabel('Spending Score (1-100)', fontsize=14)
ax.set_title('Mall Customer Segmentation\n(K-Means Clustering with K=5)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

legend_labels = [f"Cluster {i}: {cluster_labels[i]['label'].split(',')[0]}" for i in range(optimal_k)]
legend_elements = [plt.scatter([], [], c=colors[i], s=100, label=legend_labels[i]) for i in range(optimal_k)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('q7_customer_segments.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("MARKETING RECOMMENDATIONS")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    CUSTOMER SEGMENT STRATEGIES                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  CLUSTER 0: Low Income, Low Spending (CAREFUL)                        ║
║  Strategy: Discount promotions, loyalty programs, budget products     ║
║  Focus: Value for money, bulk deals, clearance sales                  ║
║                                                                        ║
║  CLUSTER 1: Low Income, High Spending (CARELESS)                      ║
║  Strategy: Credit options, installment plans, targeted offers         ║
║  Focus: Prevent customer debt, offer smart spending tips              ║
║                                                                        ║
║  CLUSTER 2: High Income, Low Spending (SENSIBLE)                      ║
║  Strategy: Quality emphasis, exclusive products, premium services     ║
║  Focus: Long-term value, investment pieces, VIP experiences           ║
║                                                                        ║
║  CLUSTER 3: High Income, High Spending (TARGET)                       ║
║  Strategy: Premium marketing, luxury products, personalized service   ║
║  Focus: Exclusive access, new arrivals, concierge services            ║
║  ★ PRIORITY SEGMENT - Highest revenue potential                       ║
║                                                                        ║
║  CLUSTER 4: Average Income, Average Spending (STANDARD)               ║
║  Strategy: Mainstream marketing, seasonal campaigns, variety          ║
║  Focus: Broad appeal, trending products, social proof                 ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Final Results:
- Optimal Clusters: {optimal_k}
- Silhouette Score: {final_silhouette:.4f}
- Total Customers: {len(df)}

Cluster Distribution:
{df['Cluster'].value_counts().sort_index().to_string()}
""")
