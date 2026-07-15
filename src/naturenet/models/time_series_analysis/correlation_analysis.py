
#taken from TRACES_ES library
 
# Step 1 of 6: Setup and Environment Configuration

"""TRACES Setup and Environment Configuration.
This module initializes the TRACES framework environment, loads required libraries,
and sets up core data structures for time series relationship analysis.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import List, Dict, Tuple, Optional, Union

import sys

# Core configuration parameters
PARENT_CHILD_MAPPING: Dict[str, List[str]] = {} #'season' : ['month']}
CONFIG = {
    'rolling_window': 20,       # Window size for rolling correlations
    'max_lag': 8,             # Maximum lag for time-delayed analysis
    'significance_level': 0.05, # Statistical significance threshold
    'min_correlation': 0.3      # Minimum correlation strength threshold
}

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    notebook_dir = Path(os.getcwd())
    if notebook_dir.name == 'notebooks':
        return notebook_dir.parent
    return notebook_dir.parent.parent

def get_sample_data_path(filename: str) -> Path:
    """Get the full path to a sample data file.
    
    Args:
        filename: Name of the sample data file
    
    Returns:
        Path: Full path to the sample data file
    """
    return get_project_root() / 'data' / 'examples' / filename

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate the loaded DataFrame meets TRACES requirements.
    
    Args:
        df: DataFrame to validate
    
    Raises:
        ValueError: If DataFrame doesn't meet requirements
    """
    if len(df) < 3:
        raise ValueError("DataFrame must contain at least 3 rows of data")
    
    if not all(df.iloc[:, 1:].dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All columns except the first must contain numeric data")


def load_and_prepare_data(data_dir, file_path): #: Union[str, Path]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Load and prepare time series data for relationship analysis.
    
    Loads time series data from an Excel file and generates valid comparison pairs,
    excluding defined parent-child relationships.
    
    Args:
        file_path: Path to Excel file (.xlsx) containing time series data.
                  First column must contain time intervals.
                  Other columns contain series data with headers as series names.
    
    Returns:
        DataFrame: Processed time series data
        List[Tuple[str, str]]: Valid comparison pairs, excluding parent-child relationships
    
    Example:
        df, pairs = load_and_prepare_data("path/to/data.xlsx")
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If data format is invalid
    """
    #if not os.path.exists(file_path):
    #    raise FileNotFoundError(f"Data file not found: {file_path}")
    
 
    df = None
    for f in range(len(file_path)):
        print(file_path[f])
        df_sub = pd.read_excel(os.path.join(data_dir, file_path[f]), header=0)
        print(df_sub.keys())
        #df_sub.drop('season', axis=1, inplace=True)
        df_sub.drop('seasaon', axis=1, inplace=True)
 
        print(len(df_sub))
        if len(df_sub) < 4:
            continue
        if df is None:
            df = df_sub
        else:
            df = pd.concat([df, df_sub])

    print(df)
    validate_dataframe(df)
    
    all_columns = [col for col in df.columns if col != 'Time']
    valid_pairs = []
    
    for i, col1 in enumerate(all_columns):
        for col2 in all_columns[i+1:]:
            is_parent_child = False
            for parent, children in PARENT_CHILD_MAPPING.items():
                if (col1 == parent and col2 in children) or \
                   (col2 == parent and col1 in children):
                    is_parent_child = True
                    break
            
            if not is_parent_child and ("reward" in col1 or "reward" in col2 or "latent" in col1 or "latent" in col2):
                valid_pairs.append((col1, col2))
    
    return df, valid_pairs

def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a time series to zero mean and unit variance.
    
    Args:
        series: Input time series data
    
    Returns:
        Normalized series (mean=0, std=1)
    """
    if series.std() == 0:
        raise ValueError("Cannot normalize series with zero standard deviation")
    return (series - series.mean()) / series.std()


# Step 2 of 6: Core Correlation Functions

"""TRACES Core Correlation Functions.

Implements core correlation analysis methods including Pearson, Spearman, and Kendall
correlations, with rolling window analysis and method comparison capabilities.
"""

def calculate_basic_correlations(series1: pd.Series, series2: pd.Series) -> Dict:
    """Calculate standard correlation measures between two time series.

    Computes Pearson, Spearman, and Kendall correlations with significance testing.

    Args:
        series1: First time series data
        series2: Second time series data

    Returns:
        Dictionary of correlation results for each method:
        {method_name: {correlation, p_value, significant}}
    """
    s1_norm = normalize_series(series1)
    s2_norm = normalize_series(series2)
    
    pearson_corr, pearson_p = pearsonr(s1_norm, s2_norm)
    spearman_corr, spearman_p = spearmanr(s1_norm, s2_norm)
    kendall_corr, kendall_p = kendalltau(s1_norm, s2_norm)
    
    return {
        'pearson': {
            'correlation': pearson_corr,
            'p_value': pearson_p,
            'significant': pearson_p < CONFIG['significance_level']
        },
        'spearman': {
            'correlation': spearman_corr,
            'p_value': spearman_p,
            'significant': spearman_p < CONFIG['significance_level']
        },
        'kendall': {
            'correlation': kendall_corr,
            'p_value': kendall_p,
            'significant': kendall_p < CONFIG['significance_level']
        }
    }

def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series) -> Dict:
    """Calculate rolling window correlations between time series.

    Args:
        series1: First time series data
        series2: Second time series data

    Returns:
        Dictionary containing rolling correlation statistics:
        {values, mean, std, max, min}
    """
    s1_norm = normalize_series(series1)
    s2_norm = normalize_series(series2)
    
    rolling_pearson = pd.Series(s1_norm).rolling(window=CONFIG['rolling_window'])\
        .corr(pd.Series(s2_norm))
    
    return {
        'rolling_correlation': {
            'values': rolling_pearson,
            'mean': rolling_pearson.mean(),
            'std': rolling_pearson.std(),
            'max': rolling_pearson.max(),
            'min': rolling_pearson.min()
        }
    }

def identify_best_correlation_method(results: Dict) -> Tuple[str, float]:
    """Determine the correlation method showing strongest relationship.

    Args:
        results: Dictionary of correlation results from calculate_basic_correlations()

    Returns:
        (method_name, correlation_value) of strongest correlation
    """
    methods = {
        'pearson': abs(results['pearson']['correlation']),
        'spearman': abs(results['spearman']['correlation']),
        'kendall': abs(results['kendall']['correlation'])
    }
    
    best_method = max(methods.items(), key=lambda x: x[1])
    return best_method[0], best_method[1]


# Step 3 of 6: Advanced Correlation Methods and CCF Analysis

"""TRACES Advanced Correlation Analysis.

Implements advanced time series correlation methods including Cross-Correlation Function (CCF)
and time-delayed correlation analysis with comprehensive relationship metrics.
"""

def calculate_ccf(series1: pd.Series, series2: pd.Series, max_lag: int = None) -> Dict:
    """Calculate Cross Correlation Function between time series.

    Args:
        series1: First time series data
        series2: Second time series data
        max_lag: Maximum lag to consider (defaults to CONFIG['max_lag'])

    Returns:
        Dictionary containing CCF analysis:
        {correlation, optimal_lag, zero_lag_correlation, 
         all_correlations, all_lags, lag_strength_ratio}
    """
    if max_lag is None:
        max_lag = CONFIG['max_lag']
    
    s1_norm = normalize_series(series1)
    s2_norm = normalize_series(series2)
    
    correlation = signal.correlate(s1_norm, s2_norm, mode='full')
    lags = signal.correlation_lags(len(s1_norm), len(s2_norm))
    
    max_corr_idx = np.argmax(np.abs(correlation))
    max_corr = correlation[max_corr_idx]
    max_lag_found = lags[max_corr_idx]
    
    central_idx = len(correlation) // 2
    zero_lag_corr = correlation[central_idx]
    
    valid_range = (lags >= -max_lag) & (lags <= max_lag)
    filtered_corr = correlation[valid_range]
    filtered_lags = lags[valid_range]

    max_corr_idx = np.argmax(np.abs(filtered_corr))
    max_corr = filtered_corr[max_corr_idx]
    max_lag_found = filtered_lags[max_corr_idx]   

 
    return {
        'ccf': {
            'correlation': max_corr,
            'optimal_lag': max_lag_found,
            'zero_lag_correlation': zero_lag_corr,
            'all_correlations': filtered_corr,
            'all_lags': filtered_lags,
            'lag_strength_ratio': abs(max_corr / zero_lag_corr) if zero_lag_corr != 0 else np.inf
        }
    }

def calculate_time_delayed_correlations(series1: pd.Series, series2: pd.Series, 
                                    max_lag: int = None) -> Dict:
    """Calculate correlations at different time delays.

    Args:
        series1: First time series data
        series2: Second time series data
        max_lag: Maximum lag to consider (defaults to CONFIG['max_lag'])

    Returns:
        Dictionary of correlation results for each lag:
        {lag_value: {method: {correlation, p_value}}}
    """
    if max_lag is None:
        max_lag = CONFIG['max_lag']
    
    results = {'delayed_correlations': {}}
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s1 = series1.iloc[abs(lag):]
            s2 = series2.iloc[:lag]
        elif lag > 0:
            s1 = series1.iloc[:-lag]
            s2 = series2.iloc[lag:]
        else:
            s1 = series1
            s2 = series2
            
        pearson_corr, pearson_p = pearsonr(s1, s2)
        spearman_corr, spearman_p = spearmanr(s1, s2)
        kendall_corr, kendall_p = kendalltau(s1, s2)
        
        results['delayed_correlations'][lag] = {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
        }
    
    return results

def combine_correlation_analyses(series1: pd.Series, series2: pd.Series) -> Dict:
    """Combine all correlation analyses into comprehensive results.

    Args:
        series1: First time series data
        series2: Second time series data

    Returns:
        Dictionary containing all correlation analyses:
        {basic_correlations, rolling_correlation, ccf, delayed_correlations}
    """
    basic_results = calculate_basic_correlations(series1, series2)
    rolling_results = calculate_rolling_correlation(series1, series2)
    ccf_results = calculate_ccf(series1, series2)
    delayed_results = calculate_time_delayed_correlations(series1, series2)
    
    return {
        'basic_correlations': basic_results,
        'rolling_correlation': rolling_results,
        'ccf': ccf_results,
        'delayed_correlations': delayed_results
    }



# Step 4 of 6: Analysis Framework and Method Comparison

"""TRACES Analysis Framework.

Implements relationship classification and method comparison logic for time series pairs,
providing automated relationship type detection and confidence scoring.
"""

def analyze_relationship_type(results: Dict) -> Dict:
    """Classify relationship type between time series variables.

    Args:
        results: Combined correlation results containing:
                basic_correlations, ccf, rolling_correlation, delayed_correlations

    Returns:
        Classification results dictionary:
        {primary_type, confidence, supporting_metrics, method_recommendations}
    """
    basic = results['basic_correlations']
    ccf = results['ccf']
    rolling = results['rolling_correlation']
    delayed = results['delayed_correlations']
    
    pearson_spearman_diff = abs(basic['pearson']['correlation'] - 
                               basic['spearman']['correlation'])
    rolling_std = rolling['rolling_correlation']['std']
    lag_impact = ccf['ccf']['lag_strength_ratio']
    
    classification = {
        'primary_type': None,
        'confidence': 0.0,
        'supporting_metrics': {},
        'method_recommendations': []
    }
    
    if pearson_spearman_diff < 0.1 and rolling_std < 0.2:
        classification['primary_type'] = 'linear'
        classification['method_recommendations'].append('pearson')
    elif pearson_spearman_diff > 0.2:
        classification['primary_type'] = 'non_linear'
        classification['method_recommendations'].extend(['spearman', 'kendall'])
    elif lag_impact > 1.2:
        classification['primary_type'] = 'lagged'
        classification['method_recommendations'].append('ccf')
    else:
        classification['primary_type'] = 'complex'
        classification['method_recommendations'].extend(['ccf', 'spearman'])
    
    classification['confidence'] = calculate_confidence(results)
    
    return classification

def calculate_confidence(results: Dict) -> float:
    """Calculate confidence score for relationship classification.

    Args:
        results: Combined correlation results

    Returns:
        Confidence score (0-1) based on significance tests and correlation strengths
    """
    basic = results['basic_correlations']
    significant_count = sum([1 for method in basic.values() if method['significant']])
    
    confidence = (significant_count / 3) * \
                 max(abs(basic['pearson']['correlation']),
                     abs(basic['spearman']['correlation']),
                     abs(basic['kendall']['correlation']))
    
    return round(confidence, 3)

def create_summary_table(series1_name: str, series2_name: str, 
                        results: Dict, classification: Dict) -> pd.DataFrame:
    """Create comprehensive summary of correlation analyses.

    Args:
        series1_name: Name of first time series
        series2_name: Name of second time series
        results: Combined correlation results
        classification: Relationship classification results

    Returns:
        DataFrame containing correlation analysis summary
    """
    summary = {
        'Series 1': series1_name,
        'Series 2': series2_name,
        'Relationship Type': classification['primary_type'],
        'Confidence': classification['confidence'],
        'Best Method': ', '.join(classification['method_recommendations']),
        'Pearson': results['basic_correlations']['pearson']['correlation'],
        'Spearman': results['basic_correlations']['spearman']['correlation'],
        'Kendall': results['basic_correlations']['kendall']['correlation'],
        'Max CCF': results['ccf']['ccf']['correlation'],
        'Optimal Lag': results['ccf']['ccf']['optimal_lag'],
        'Rolling Mean': results['rolling_correlation']['rolling_correlation']['mean']
    }
    
    return pd.DataFrame([summary])


# Step 5 of 6: Visualization Functions

"""TRACES Visualization Suite.

Implements comprehensive visualization functions for time series relationship analysis,
including correlation comparisons, relationship matrices, and CCF patterns.
"""

def plot_correlation_comparison(out_dir, full_results: pd.DataFrame, figsize=(15, 10)) -> None:
    """Plot comparative visualization of correlation methods.

    Generates grouped bar plot comparing Pearson, Spearman, and Kendall
    correlations for top relationships.

    Args:
        full_results: DataFrame containing analysis results
        figsize: Figure dimensions (width, height)
    """
    plt.figure(figsize=figsize)
    
    plot_data = full_results[['Series 1', 'Series 2', 'Pearson', 'Spearman', 'Kendall']].copy()
    plot_data.loc[:, 'Pair'] = plot_data['Series 1'] + ' - ' + plot_data['Series 2']
    
    plot_data_melted = pd.melt(
        plot_data,
        id_vars=['Pair'],
        value_vars=['Pearson', 'Spearman', 'Kendall'],
        var_name='Method',
        value_name='Correlation'
    )
    
    ax = sns.barplot(
        data=plot_data_melted,
        x='Pair',
        y='Correlation',
        hue='Method',
        palette='coolwarm'
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Comparison of Correlation Methods Across Top Relationships')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_analysis.png"), dpi=400)
 
    #plt.show()

def plot_relationship_matrix(out_dir, full_results: pd.DataFrame, figsize=(12, 8)) -> None:
    """Plot matrix visualization of relationship types and confidence scores.

    Note: This visualization is designed for the top N strongest relationships 
    (where N is an even number, default from previous is 24, but customizable).
    The matrix will naturally contain blank/NaN cells due to the cartesian product of
    unique series pairs, which is expected behavior amidst the other strongest relationship patterns.

    Args:
        full_results: DataFrame containing top N analysis results (N should be even)
        figsize: Figure dimensions (width, height)
    """
    sorted_results = full_results.sort_values('Confidence', ascending=False)
    pairs = list(zip(sorted_results['Series 1'], sorted_results['Series 2']))
    
    series1_ordered = []
    series2_ordered = []
    for s1, s2 in pairs:
        if s1 not in series1_ordered:
            series1_ordered.append(s1)
        if s2 not in series2_ordered:
            series2_ordered.append(s2)
    
    matrix_data = pd.DataFrame(np.nan, 
                             index=series1_ordered,
                             columns=series2_ordered)
    
    type_matrix = pd.DataFrame('',
                             index=series1_ordered,
                             columns=series2_ordered)
    
    for _, row in sorted_results.iterrows():
        matrix_data.loc[row['Series 1'], row['Series 2']] = row['Confidence']
        type_matrix.loc[row['Series 1'], row['Series 2']] = row['Relationship Type']
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.isnan(matrix_data)
    
    sns.heatmap(
        matrix_data,
        annot=type_matrix,
        fmt='',
        ax=ax,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        mask=mask,
        cbar_kws={'label': 'Confidence Score'}
    )
    
    plt.title('Top Relationships by Confidence Score')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "matrix_analysis.png"), dpi=400)
 
    #plt.show()
    
def plot_method_performance(out_dir, full_results: pd.DataFrame, figsize=(15, 6)) -> None:
    """Plot method performance across relationship types.

    Args:
        full_results: DataFrame containing analysis results
        figsize: Figure dimensions (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    method_counts = pd.DataFrame(full_results.groupby('Relationship Type')['Best Method'].value_counts())
    method_counts = method_counts.unstack(fill_value=0)
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(method_counts.columns)))
    method_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors
    )
    
    plt.title('Method Performance by Relationship Type')
    plt.xlabel('Relationship Type')
    plt.ylabel('Count')
    plt.legend(title='Best Method', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "method_perf.png"), dpi=400)
    #plt.show()
 
def plot_ccf_analysis(out_dir, full_results: pd.DataFrame, figsize=(12, 6)) -> None:
    """Plot CCF analysis showing correlation strength vs lag patterns.

    Args:
        full_results: DataFrame containing analysis results
        figsize: Figure dimensions (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = plt.scatter(
        full_results['Optimal Lag'],
        full_results['Max CCF'].abs(),
        c=full_results['Confidence'],
        cmap='coolwarm',
        s=100
    )
    
    for idx, row in full_results.iterrows():
        plt.annotate(
            row['Relationship Type'],
            (row['Optimal Lag'], abs(row['Max CCF'])),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.colorbar(scatter, label='Confidence Score')
    plt.title('CCF Analysis: Maximum Correlation vs Optimal Lag')
    plt.xlabel('Optimal Lag')
    plt.ylabel('|Maximum CCF|')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "CCF_analysis.png"), dpi=400)
    #plt.show()
 
# Step 6 of 6: Results Processing and Full Dataset Analysis

"""TRACES Results Processing Module.

Implements comprehensive dataset analysis pipeline, including correlation processing,
summary statistics generation, and detailed results reporting by relationship type.
"""

def analyze_full_dataset(df: pd.DataFrame, 
                        valid_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Process all valid series pairs through correlation analysis pipeline.

    Args:
        df: DataFrame containing time series data
        valid_pairs: List of valid series pairs for analysis

    Returns:
        DataFrame containing comprehensive analysis results
    """
    results_list = []
    
    for pair in valid_pairs:
        series1 = df[pair[0]]
        series2 = df[pair[1]]
        
        results = combine_correlation_analyses(series1, series2)
        classification = analyze_relationship_type(results)
        
        summary = {
            'Series 1': pair[0],
            'Series 2': pair[1],
            'Relationship Type': classification['primary_type'],
            'Confidence': classification['confidence'],
            'Best Method': ', '.join(classification['method_recommendations']),
            'Pearson': results['basic_correlations']['pearson']['correlation'],
            'P-Value': results['basic_correlations']['pearson']['p_value'],
            'Spearman': results['basic_correlations']['spearman']['correlation'],
            'P-Value': results['basic_correlations']['spearman']['p_value'],
            'Kendall': results['basic_correlations']['kendall']['correlation'],
            'P-Value': results['basic_correlations']['kendall']['p_value'],
            'Max CCF': results['ccf']['ccf']['correlation'],
            'Optimal Lag': results['ccf']['ccf']['optimal_lag'],
            'Rolling Mean': results['rolling_correlation']['rolling_correlation']['mean'],
            'Abs_Max_Corr': max(abs(results['basic_correlations']['pearson']['correlation']),
                               abs(results['basic_correlations']['spearman']['correlation']),
                               abs(results['basic_correlations']['kendall']['correlation']))
        }
        results_list.append(summary)
    
    results_df = pd.DataFrame(results_list)
    return results_df.sort_values('Abs_Max_Corr', ascending=False)

def generate_summary_statistics(results_df: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics from analysis results.

    Args:
        results_df: DataFrame containing analysis results

    Returns:
        Dictionary of summary statistics including relationship types,
        confidence scores, and correlation strength distributions
    """
    relationship_types = results_df['Relationship Type'].value_counts().to_dict()
    avg_confidence = results_df['Confidence'].mean()
    
    method_counts = {}
    for methods in results_df['Best Method']:
        for method in methods.split(', '):
            method_counts[method] = method_counts.get(method, 0) + 1
    
    strong_correlations = len(results_df[results_df['Abs_Max_Corr'] > 0.7])
    moderate_correlations = len(results_df[
        (results_df['Abs_Max_Corr'] >= 0.3) & 
        (results_df['Abs_Max_Corr'] <= 0.7)
    ])
    weak_correlations = len(results_df[results_df['Abs_Max_Corr'] < 0.3])
    
    return {
        'relationship_types': relationship_types,
        'avg_confidence': avg_confidence,
        'method_counts': method_counts,
        'strong_correlations': strong_correlations,
        'moderate_correlations': moderate_correlations,
        'weak_correlations': weak_correlations
    }

def print_grouped_results(results_df: pd.DataFrame) -> None:
    """Print detailed analysis results grouped by relationship type.

    Args:
        results_df: DataFrame containing analysis results
    """
    grouped = results_df.groupby('Relationship Type')
    
    for rel_type, group in grouped:
        print(f"\n=== {rel_type.upper()} RELATIONSHIPS ===")
        print(f"Number of pairs: {len(group)}")
        
        top_pairs = group.nlargest(10, 'Abs_Max_Corr')
        
        print("\nTop 10 strongest correlations:")
        for _, row in top_pairs.iterrows():
            print(f"\n{row['Series 1']} vs {row['Series 2']}:")
            print(f"  Absolute Max Correlation: {row['Abs_Max_Corr']:.3f}")
            print(f"  Best Method(s): {row['Best Method']}")
            print(f"  Confidence: {row['Confidence']:.3f}")
            if abs(row['Optimal Lag']) > 0:
                print(f"  Optimal Lag: {row['Optimal Lag']}")
        
        print(f"\nGroup Statistics:")
        print(f"  Mean Confidence: {group['Confidence'].mean():.3f}")
        print(f"  Mean Abs Correlation: {group['Abs_Max_Corr'].mean():.3f}")
        print(f"  Most Common Best Method: {group['Best Method'].mode().iloc[0]}")

def run_full_analysis(df: pd.DataFrame, 
                     valid_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Execute complete TRACES analysis pipeline on dataset.

    Args:
        df: DataFrame containing time series data
        valid_pairs: List of valid series pairs for analysis

    Returns:
        DataFrame containing complete analysis results
    """
    print("\nInitiating TRACES Correlation Analysis...")
    
    try:
        full_results = analyze_full_dataset(df, valid_pairs)
        print(f"Completed analysis of {len(full_results)} pairs")
        
        print("\nGenerating summary statistics...")
        summary_stats = generate_summary_statistics(full_results)
        
        print("\nANALYSIS SUMMARY:")
        print(f"Total pairs analyzed: {len(full_results)}")
        print("\nRelationship Types Distribution:")
        for rel_type, count in summary_stats['relationship_types'].items():
            print(f"{rel_type}: {count}")
        
        print("\nAverage Confidence Score:", 
              f"{summary_stats['avg_confidence']:.3f}")
        
        print("\nCorrelation Strength Distribution:")
        print(f"Strong correlations (>0.7): {summary_stats['strong_correlations']}")
        print(f"Moderate correlations (0.3-0.7): "
              f"{summary_stats['moderate_correlations']}")
        print(f"Weak correlations (<0.3): {summary_stats['weak_correlations']}")
        
        print("\nPrinting detailed results by group...")
        print_grouped_results(full_results)
        
        return full_results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

main_dir = "/data/nlahaye/NatureNet/Blue_Whale_v1/"

second_dir = ["s1_complex", "s1_simple", "s2_complex", "s2_simple"]

keys = ["full", "males", "females", "unknown"]

data_slices = {
"full": [
"2017CA-Bmu-00826_time_series_pd.xlsx",
"2017CA-Bmu-05679_time_series_pd.xlsx",
"2017CA-Bmu-00827_time_series_pd.xlsx",
"2017CA-Bmu-05803_time_series_pd.xlsx",
"2017CA-Bmu-00831_time_series_pd.xlsx",
"2017CA-Bmu-05840_time_series_pd.xlsx",
"2017CA-Bmu-00835_time_series_pd.xlsx",
"2017CA-Bmu-05910_time_series_pd.xlsx", #7
"2017CA-Bmu-00841_time_series_pd.xlsx",
"2017CA-Bmu-05921_time_series_pd.xlsx",
"2017CA-Bmu-00845_time_series_pd.xlsx",
"2017CA-Bmu-10826_time_series_pd.xlsx",
"2017CA-Bmu-00847_time_series_pd.xlsx",
"2017CA-Bmu-10830_time_series_pd.xlsx",
"2017CA-Bmu-01385_time_series_pd.xlsx", #15
"2017CA-Bmu-10831_time_series_pd.xlsx",
"2017CA-Bmu-04176_time_series_pd.xlsx",  
"2017CA-Bmu-10840_time_series_pd.xlsx",

"2017CA-Bmu-05648_time_series_pd.xlsx",  
"2017CA-Bmu-23031_time_series_pd.xlsx", #20
"2017CA-Bmu-05670_time_series_pd.xlsx",
],


"males": [
"2017CA-Bmu-00827_time_series_pd.xlsx",
"2017CA-Bmu-10840_time_series_pd.xlsx",
"2017CA-Bmu-23031_time_series_pd.xlsx",
"2017CA-Bmu-00826_time_series_pd.xlsx",
"2017CA-Bmu-00841_time_series_pd.xlsx",
"2017CA-Bmu-05648_time_series_pd.xlsx",
"2017CA-Bmu-05670_time_series_pd.xlsx",
"2017CA-Bmu-05803_time_series_pd.xlsx",
"2017CA-Bmu-05921_time_series_pd.xlsx",
"2017CA-Bmu-10826_time_series_pd.xlsx",
],

"females": [
"2017CA-Bmu-00831_time_series_pd.xlsx",
"2017CA-Bmu-00835_time_series_pd.xlsx",
"2017CA-Bmu-00845_time_series_pd.xlsx",
"2017CA-Bmu-04176_time_series_pd.xlsx",
"2017CA-Bmu-05910_time_series_pd.xlsx",
],

"unknown": [
"2017CA-Bmu-01385_time_series_pd.xlsx",
"2017CA-Bmu-05840_time_series_pd.xlsx",
"2017CA-Bmu-10830_time_series_pd.xlsx",
"2017CA-Bmu-10831_time_series_pd.xlsx",
"2017CA-Bmu-00847_time_series_pd.xlsx",
"2017CA-Bmu-05679_time_series_pd.xlsx",
]
}

# Data loading validation


try:
 
    #file_path = get_sample_data_path(data_files[2])
    #print(file_path, "HERE")
    #"2017CA-Bmu-00826_time_series_pd.xlsx")  #'TRACES_sample_52x10_dataset_A1.xlsx')

    for ind in range(len(second_dir)):
        for key in data_slices.keys():
            dat = data_slices[key]
    
            out_dir = os.path.join(main_dir, second_dir[ind], key)
            os.makedirs(out_dir, exist_ok=True)
            df, valid_pairs = load_and_prepare_data(os.path.join(main_dir, second_dir[ind]), dat) #file_path)
            print(f"{second_dir[ind]} {key} Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
            print(f"{key} Generated {len(valid_pairs)} valid comparison pairs")
            
            print("\n{key} First 10 comparison pairs:")
            for pair in valid_pairs[:10]:
                print(pair)
            
            print("\n{key} Columns in dataset:")
            print(df.columns.tolist())
            #except Exception as e:
            #print(f"Error loading data: {str(e)}")
            #raise  # Re-raise the exception to ensure the notebook shows the error
        
            # Results compilation and analysis
            print("{key} Testing correlation functions across all series pairs...")
        
            summary_results = []
        
            for pair in valid_pairs:
                series1 = df[pair[0]]
                series2 = df[pair[1]]
                print(pair)   
             
                basic_results = calculate_basic_correlations(series1, series2)
                rolling_results = calculate_rolling_correlation(series1, series2)
                best_method, best_value = identify_best_correlation_method(basic_results)
                
                summary_results.append({
                    'Series 1': pair[0],
                    'Series 2': pair[1],
                    'Pearson': basic_results['pearson']['correlation'],
                    #'P-Value_P': basic_results['pearson']['p_value'],
                    'Pearson_Sig': basic_results['pearson']['significant'],
                    'Spearman': basic_results['spearman']['correlation'],
                    #'P-Value_S': basic_results['spearman']['p_value'],
                    'Spearman_Sig': basic_results['spearman']['significant'],
                    'Kendall': basic_results['kendall']['correlation'],
                    #'P-Value_K': basic_results['kendall']['p_value'],
                    'Kendall_Sig': basic_results['kendall']['significant'],
                    'Rolling_Mean': rolling_results['rolling_correlation']['mean'],
                    'Rolling_Std': rolling_results['rolling_correlation']['std'],
                    'Best_Method': best_method,
                    'Best_Value': best_value
                })
            
            # Results analysis and display
            results_df = pd.DataFrame(summary_results)
            results_df['Abs_Best_Value'] = abs(results_df['Best_Value'])
            results_df = results_df.sort_values('Abs_Best_Value', ascending=False)
            
            print("\n{key} Top 10 Strongest Correlations:")
            print(results_df[['Series 1', 'Series 2', 'Best_Method', 'Best_Value']]) #, 'P-Value_P', 'P-Value_S', 'P-Value_K']])
            
            print("\n{key} Correlation Method Distribution:")
            print(results_df['Best_Method'].value_counts())
            
            print("\n{key} Significant Correlations Count:")
            print(f"{key} Pearson: {results_df['Pearson_Sig'].sum()}")
            print(f"{key} Spearman: {results_df['Spearman_Sig'].sum()}")
            print(f"{key} Kendall: {results_df['Kendall_Sig'].sum()}")
            
            
            # Analysis execution and results compilation
            print("{key} Testing advanced correlation methods across all series pairs...")
            
            summary_results = []
            
            for pair in valid_pairs:
                series1 = df[pair[0]]
                series2 = df[pair[1]]
                
                results = combine_correlation_analyses(series1, series2)
                
                summary = {
                    'Series 1': pair[0],
                    'Series 2': pair[1],
                    'CCF_Max_Corr': results['ccf']['ccf']['correlation'],
                    'CCF_Optimal_Lag': results['ccf']['ccf']['optimal_lag'],
                    'CCF_Zero_Lag': results['ccf']['ccf']['zero_lag_correlation'],
                    'Best_Delayed_Lag': max(
                        results['delayed_correlations']['delayed_correlations'].items(),
                        key=lambda x: abs(x[1]['pearson']['correlation'])
                    )[0]
                }
                summary_results.append(summary)
            
            # Results analysis
            results_df = pd.DataFrame(summary_results)
            results_df = results_df.iloc[results_df['CCF_Max_Corr'].abs().argsort()[::-1]]
            
            print("\n{key} Advanced Correlation Analysis Results:")
            print(results_df)
            
            print("\n{key} Summary Statistics:")
            print(f"{key} Average Optimal Lag: {results_df['CCF_Optimal_Lag'].mean():.2f}")
            print(f"{key} Max CCF Correlation: {results_df['CCF_Max_Corr'].max():.4f}")
            
            
            # Analysis execution and results compilation
            print("Testing analysis framework across all series pairs...")
            
            all_summaries = []
            
            for pair in valid_pairs:
                series1 = df[pair[0]]
                series2 = df[pair[1]]
                
                results = combine_correlation_analyses(series1, series2)
                classification = analyze_relationship_type(results)
                
                summary = create_summary_table(pair[0], pair[1], results, classification)
                all_summaries.append(summary)
            
            # Results analysis
            full_results = pd.concat(all_summaries, ignore_index=True)
            full_results['max_correlation'] = full_results[['Pearson', 'Spearman', 'Kendall']].abs().max(axis=1)
            full_results = full_results.nlargest(24, 'max_correlation')
            full_results = full_results.drop('max_correlation', axis=1)
            
            print("\nRelationship Analysis Results:")
            print(full_results.to_string())
            
            print("\nRelationship Type Distribution:")
            print(full_results['Relationship Type'].value_counts())
            
            print("\nConfidence Statistics:")
            print(f"Mean Confidence: {full_results['Confidence'].mean():.3f}")
            print(f"Max Confidence: {full_results['Confidence'].max():.3f}")
            
            print("\nRecommended Methods Distribution:")
            full_results['Best Method'].value_counts().to_frame()
            
            # Visualization generation
            print("{key} Generating visualization suite...")
            
            plot_correlation_comparison(out_dir, full_results)
            plot_relationship_matrix(out_dir, full_results)
            plot_method_performance(out_dir, full_results)
            plot_ccf_analysis(out_dir, full_results)
            
            print("\nVisualization suite complete. Plot descriptions:")
            print("1. Bar plot: Correlation method comparison across relationships")
            print("2. Matrix: Relationship types and confidence scores")
            print("3. Bar chart: Method effectiveness by relationship type")
            print("4. Scatter plot: CCF patterns and lag relationships")
            
            # Execute full analysis pipeline
            final_results = run_full_analysis(df, valid_pairs)


except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise  # Re-raise the exception to ensure the notebook shows the error 
   
