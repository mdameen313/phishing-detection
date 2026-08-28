"""
CBA Rule Analysis - CORRECTED
Shows how each rule affects the URL and increases risk score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ASSOCIATIVE CLASSIFICATION (CBA) RULE ANALYSIS")
print("="*70)

# Load data
train_df = pd.read_csv('processed_data/train_enhanced.csv')
X = train_df.drop('label', axis=1)
y = train_df['label']

print(f"\nLoaded {len(train_df)} training samples")
print(f"Features: {X.columns.tolist()}")

# ============================================================
# 1. Define CBA Rules (with correct column names)
# ============================================================

rules = [
    {
        'id': 1,
        'name': 'Suspicious TLD',
        'condition': "X['has_suspicious_tld'] == 1",
        'confidence': 0.85,
        'description': 'Domain ends with .xyz, .tk, .ml, .top, .club'
    },
    {
        'id': 2,
        'name': 'IP Address',
        'condition': "X['has_ip'] == 1",
        'confidence': 0.95,
        'description': 'URL contains IP address instead of domain'
    },
    {
        'id': 3,
        'name': '@ Symbol',
        'condition': "X['has_@'] == 1",
        'confidence': 0.90,
        'description': 'URL contains @ symbol (credential phishing)'
    },
    {
        'id': 4,
        'name': 'Shortened URL',
        'condition': "X['is_shortened'] == 1",
        'confidence': 0.70,
        'description': 'URL is shortened (bit.ly, tinyurl, etc.)'
    },
    {
        'id': 5,
        'name': 'Brand Misuse',
        'condition': "(X['num_sensitive_words'] > 0) & (X['has_suspicious_tld'] == 1)",
        'confidence': 0.88,
        'description': 'Brand name used with sensitive keywords'
    },
    {
        'id': 6,
        'name': 'Long URL',
        'condition': "X['url_length'] > 100",
        'confidence': 0.75,
        'description': 'URL length > 100 characters'
    },
    {
        'id': 7,
        'name': 'Multiple Subdomains',
        'condition': "X['num_dots'] > 3",
        'confidence': 0.75,
        'description': 'Multiple subdomains (domain spoofing)'
    },
    {
        'id': 8,
        'name': 'HTTP for Sensitive Page',
        'condition': "(X['has_https'] == 0) & (X['num_sensitive_words'] > 0)",
        'confidence': 0.70,
        'description': 'HTTP (insecure) for login/secure page'
    }
]

# ============================================================
# 2. Calculate Rule Impact
# ============================================================
print("\n" + "="*70)
print("2. RULE IMPACT ANALYSIS")
print("="*70)

rule_impact = []
base_phishing_ratio = y.mean()

for rule in rules:
    try:
        # Apply condition using eval
        rule_indices = eval(rule['condition']).values
        rule_applies = sum(rule_indices)
        
        if rule_applies == 0:
            continue
        
        # Get indices where rule applies
        rule_indices_bool = eval(rule['condition'])
        rule_indices = rule_indices_bool[rule_indices_bool].index
        
        # Calculate phishing ratio when rule applies
        phishing_ratio_rule = y.loc[rule_indices].mean() if len(rule_indices) > 0 else 0
        
        # Calculate impact
        impact = phishing_ratio_rule - base_phishing_ratio
        
        # Calculate support
        support = rule_applies / len(X)
        
        # Calculate lift
        lift = phishing_ratio_rule / base_phishing_ratio if base_phishing_ratio > 0 else 0
        
        rule_impact.append({
            'Rule_ID': rule['id'],
            'Rule_Name': rule['name'],
            'Description': rule['description'],
            'Confidence': rule['confidence'],
            'Support': support,
            'Phishing_Ratio_When_Applies': phishing_ratio_rule,
            'Base_Phishing_Ratio': base_phishing_ratio,
            'Impact': impact,
            'Lift': lift,
            'Times_Applied': rule_applies
        })
    except Exception as e:
        print(f"Error with rule {rule['name']}: {e}")

if len(rule_impact) == 0:
    print("\n⚠️ No rules applied. Try adjusting conditions.")
else:
    rule_impact_df = pd.DataFrame(rule_impact).sort_values('Impact', ascending=False)

    print("\nRule Impact on Phishing Detection:")
    print(rule_impact_df[['Rule_Name', 'Confidence', 'Support', 'Impact', 'Lift']].to_string(index=False))

    # ============================================================
    # 3. Risk Score Contribution by Rule
    # ============================================================
    print("\n" + "="*70)
    print("3. RISK SCORE CONTRIBUTION BY RULE")
    print("="*70)

    # Weights
    w1, w2, w3 = 0.40, 0.35, 0.25

    risk_contributions = []
    for _, row in rule_impact_df.iterrows():
        # P_RF (Random Forest contribution) - fixed baseline
        p_rf = 0.70  # Baseline RF probability
        
        # P_AC (Rule confidence)
        p_ac = row['Confidence']
        
        # F_critical (Critical indicators)
        f_critical = row['Impact'] * 10  # Scale impact to 0-1 range
        f_critical = min(max(f_critical, 0), 0.25)
        
        # Risk Score
        risk_score = w1 * p_rf + w2 * p_ac + w3 * f_critical
        
        # Contribution analysis
        rf_contribution = w1 * p_rf
        ac_contribution = w2 * p_ac
        critical_contribution = w3 * f_critical
        
        risk_contributions.append({
            'Rule_Name': row['Rule_Name'],
            'P_RF': p_rf,
            'P_AC': p_ac,
            'F_Critical': f_critical,
            'Risk_Score': risk_score,
            'RF_Contribution': rf_contribution,
            'AC_Contribution': ac_contribution,
            'Critical_Contribution': critical_contribution
        })

    risk_contrib_df = pd.DataFrame(risk_contributions)

    print("\nRisk Score Contribution by Rule:")
    print(risk_contrib_df[['Rule_Name', 'Risk_Score', 'RF_Contribution', 'AC_Contribution', 'Critical_Contribution']].to_string(index=False))

    # ============================================================
    # 4. Visualization
    # ============================================================
    print("\n" + "="*70)
    print("4. GENERATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Rule Impact
    ax1 = axes[0, 0]
    top_rules = rule_impact_df.head(8)
    colors = ['#d32f2f' if i < 3 else '#ff9800' for i in range(len(top_rules))]
    ax1.barh(top_rules['Rule_Name'], top_rules['Impact']*100, color=colors)
    ax1.set_xlabel('Impact on Phishing Detection (%)')
    ax1.set_title('Rule Impact on Phishing Detection', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    for i, v in enumerate(top_rules['Impact']*100):
        ax1.text(v + 1, i, f'{v:.1f}%', va='center')

    # Subplot 2: Rule Support vs Confidence
    ax2 = axes[0, 1]
    ax2.scatter(rule_impact_df['Support'], rule_impact_df['Confidence'], s=200, c='#1a73e8', alpha=0.7)
    for _, row in rule_impact_df.iterrows():
        ax2.annotate(row['Rule_Name'][:12], (row['Support'], row['Confidence']), fontsize=8)
    ax2.set_xlabel('Support (Frequency in Dataset)')
    ax2.set_ylabel('Confidence (Rule Accuracy)')
    ax2.set_title('Rule Support vs Confidence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Risk Score Contribution
    ax3 = axes[1, 0]
    x = np.arange(len(risk_contrib_df))
    width = 0.25
    ax3.bar(x - width, risk_contrib_df['RF_Contribution'], width, label='RF Contribution', color='#1a73e8')
    ax3.bar(x, risk_contrib_df['AC_Contribution'], width, label='AC Contribution', color='#ff9800')
    ax3.bar(x + width, risk_contrib_df['Critical_Contribution'], width, label='Critical Contribution', color='#4caf50')
    ax3.set_xlabel('Rule')
    ax3.set_ylabel('Contribution')
    ax3.set_title('Risk Score Contribution Breakdown', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(risk_contrib_df['Rule_Name'], rotation=45, ha='right', fontsize=8)
    ax3.legend()

    # Subplot 4: Lift Analysis
    ax4 = axes[1, 1]
    ax4.barh(rule_impact_df['Rule_Name'], rule_impact_df['Lift'], color='#7b1fa2')
    ax4.axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Random)')
    ax4.set_xlabel('Lift (Rule Effectiveness)')
    ax4.set_title('Rule Lift Analysis', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.legend()

    plt.tight_layout()
    plt.savefig('cba_rule_analysis.png', dpi=200, bbox_inches='tight')
    print("✅ Saved: cba_rule_analysis.png")

    # ============================================================
    # 5. Summary Report
    # ============================================================
    print("\n" + "="*70)
    print("5. CBA RULES SUMMARY")
    print("="*70)

    print(f"""
CBA RULES SUMMARY
=================
Total Rules Defined: {len(rules)}
Rules Applied in Dataset: {len(rule_impact_df)}

Rule Rankings (by Impact):
{rule_impact_df[['Rule_Name', 'Impact', 'Support', 'Lift']].to_string(index=False)}

Risk Score Contribution:
{risk_contrib_df[['Rule_Name', 'Risk_Score']].to_string(index=False)}

Key Findings:
1. Most Impactful Rule: {rule_impact_df.iloc[0]['Rule_Name']} (Impact: {rule_impact_df.iloc[0]['Impact']*100:.2f}%)
2. Highest Confidence Rule: {rule_impact_df.iloc[rule_impact_df['Confidence'].idxmax()]['Rule_Name']} ({rule_impact_df['Confidence'].max():.2f}%)
3. Most Frequent Rule: {rule_impact_df.iloc[rule_impact_df['Support'].idxmax()]['Rule_Name']} ({rule_impact_df['Support'].max():.2f}%)
""")

    print("\n✅ CBA Rule Analysis Complete!")