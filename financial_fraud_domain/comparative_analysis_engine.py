"""
Comparative Analysis Engine for Multi-Model Statistical Comparison
Advanced statistical methods for model comparison, significance testing, and ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
import logging
from sklearn.linear_model import LinearRegression


class ComparativeAnalysisEngine:
    """
    Engine for performing complex comparative statistical analysis.
    Handles multi-model comparisons, significance testing, and effect size calculations.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the comparative analysis engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def perform_pairwise_comparisons(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic pairwise comparisons between all model pairs"""
        pairwise_results = {}
        
        for model1, model2 in prepared_data["comparison_pairs"]:
            pair_key = f"{model1}_vs_{model2}"
            pairwise_results[pair_key] = {}
            
            for metric_name in prepared_data["models"][model1].keys():
                if metric_name in prepared_data["models"][model2]:
                    values1 = prepared_data["metrics"][metric_name][model1]
                    values2 = prepared_data["metrics"][metric_name][model2]
                    
                    if isinstance(values1, np.ndarray) and isinstance(values2, np.ndarray):
                        if len(values1) >= 2 and len(values2) >= 2:
                            # Parametric test (t-test)
                            t_stat, t_p_value = stats.ttest_ind(values1, values2)
                            
                            # Non-parametric test (Mann-Whitney U)
                            u_stat, u_p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                            
                            # Calculate mean difference
                            mean_diff = np.mean(values1) - np.mean(values2)
                            
                            pairwise_results[pair_key][metric_name] = {
                                "parametric": {
                                    "test": "independent_t_test",
                                    "statistic": float(t_stat),
                                    "p_value": float(t_p_value),
                                    "significant": t_p_value < 0.05
                                },
                                "non_parametric": {
                                    "test": "mann_whitney_u",
                                    "statistic": float(u_stat),
                                    "p_value": float(u_p_value),
                                    "significant": u_p_value < 0.05
                                },
                                "mean_difference": float(mean_diff),
                                "sample_sizes": {
                                    model1: len(values1),
                                    model2: len(values2)
                                }
                            }
        
        return pairwise_results
    
    def perform_bonferroni_correction(self, pairwise_results: Dict[str, Any], 
                                     prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons"""
        bonferroni_results = {
            "method": "bonferroni",
            "corrected_alpha": 0.0,
            "total_comparisons": 0,
            "corrected_results": {},
            "summary": {}
        }
        
        # Count total comparisons
        total_comparisons = 0
        for pair_results in pairwise_results.values():
            total_comparisons += len(pair_results)
        
        bonferroni_results["total_comparisons"] = total_comparisons
        bonferroni_results["corrected_alpha"] = 0.05 / total_comparisons if total_comparisons > 0 else 0.05
        
        # Apply correction to all p-values
        for pair_key, metrics in pairwise_results.items():
            bonferroni_results["corrected_results"][pair_key] = {}
            
            for metric_name, test_results in metrics.items():
                # Correct both parametric and non-parametric p-values
                corrected_p_parametric = min(1.0, test_results["parametric"]["p_value"] * total_comparisons)
                corrected_p_nonparametric = min(1.0, test_results["non_parametric"]["p_value"] * total_comparisons)
                
                bonferroni_results["corrected_results"][pair_key][metric_name] = {
                    "original_p_values": {
                        "parametric": test_results["parametric"]["p_value"],
                        "non_parametric": test_results["non_parametric"]["p_value"]
                    },
                    "corrected_p_values": {
                        "parametric": float(corrected_p_parametric),
                        "non_parametric": float(corrected_p_nonparametric)
                    },
                    "significant_after_correction": {
                        "parametric": corrected_p_parametric < 0.05,
                        "non_parametric": corrected_p_nonparametric < 0.05
                    },
                    "mean_difference": test_results["mean_difference"]
                }
        
        # Generate summary
        significant_count = 0
        for pair_results in bonferroni_results["corrected_results"].values():
            for metric_results in pair_results.values():
                if metric_results["significant_after_correction"]["parametric"]:
                    significant_count += 1
        
        bonferroni_results["summary"] = {
            "total_tests": total_comparisons,
            "corrected_alpha": bonferroni_results["corrected_alpha"],
            "significant_after_correction": significant_count,
            "correction_factor": total_comparisons
        }
        
        return bonferroni_results
    
    def perform_tukey_hsd_test(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Tukey's HSD test for multiple comparisons"""
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
        except ImportError:
            self.logger.warning("statsmodels not available for Tukey HSD test")
            return {"status": "error", "error": "statsmodels package required for Tukey HSD test"}
        
        tukey_results = {
            "method": "tukey_hsd",
            "results_by_metric": {},
            "summary": {}
        }
        
        for metric_name, model_values in prepared_data["aligned_data"].items():
            if len(model_values) >= 3:  # Need at least 3 groups
                try:
                    # Prepare data for Tukey HSD
                    all_values = []
                    all_labels = []
                    
                    for model_id, values in model_values.items():
                        if isinstance(values, np.ndarray) and len(values) > 0:
                            all_values.extend(values)
                            all_labels.extend([model_id] * len(values))
                    
                    if len(set(all_labels)) >= 3:  # Ensure we have at least 3 different groups
                        # Perform Tukey HSD
                        tukey_result = pairwise_tukeyhsd(all_values, all_labels, alpha=0.05)
                        
                        # Extract results
                        comparisons = []
                        for i in range(len(tukey_result.summary().data) - 1):  # Skip header
                            row = tukey_result.summary().data[i + 1]
                            comparisons.append({
                                "group1": row[0],
                                "group2": row[1],
                                "mean_diff": float(row[2]),
                                "p_adj": float(row[5]),
                                "lower_ci": float(row[3]),
                                "upper_ci": float(row[4]),
                                "reject_null": bool(row[6]),
                                "significant": bool(row[6])
                            })
                        
                        tukey_results["results_by_metric"][metric_name] = {
                            "comparisons": comparisons,
                            "alpha": 0.05,
                            "groups_compared": len(set(all_labels)),
                            "total_observations": len(all_values)
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Tukey HSD failed for {metric_name}: {e}")
                    tukey_results["results_by_metric"][metric_name] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Generate summary
        total_comparisons = 0
        significant_comparisons = 0
        
        for metric_results in tukey_results["results_by_metric"].values():
            if "comparisons" in metric_results:
                for comp in metric_results["comparisons"]:
                    total_comparisons += 1
                    if comp["significant"]:
                        significant_comparisons += 1
        
        tukey_results["summary"] = {
            "total_comparisons": total_comparisons,
            "significant_comparisons": significant_comparisons,
            "metrics_tested": len(tukey_results["results_by_metric"]),
            "family_wise_error_rate": 0.05
        }
        
        return tukey_results
    
    def perform_friedman_test(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Friedman test for repeated measures across multiple groups"""
        friedman_results = {
            "method": "friedman",
            "results_by_metric": {},
            "post_hoc_tests": {},
            "summary": {}
        }
        
        for metric_name, model_values in prepared_data["aligned_data"].items():
            if len(model_values) >= 3:  # Need at least 3 groups
                try:
                    # Prepare data matrix for Friedman test
                    model_ids = list(model_values.keys())
                    min_length = min(len(v) for v in model_values.values() if isinstance(v, np.ndarray))
                    
                    if min_length >= 3:  # Need sufficient observations
                        # Create matrix where rows are observations, columns are models
                        data_matrix = []
                        for i in range(min_length):
                            row = [model_values[model_id][i] for model_id in model_ids]
                            data_matrix.append(row)
                        
                        data_matrix = np.array(data_matrix)
                        
                        # Perform Friedman test
                        stat, p_value = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(len(model_ids))])
                        
                        friedman_results["results_by_metric"][metric_name] = {
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                            "df": len(model_ids) - 1,
                            "models": model_ids,
                            "observations": min_length
                        }
                        
                        # If significant, perform post-hoc tests
                        if p_value < 0.05:
                            post_hoc = self._perform_friedman_post_hoc(data_matrix, model_ids)
                            friedman_results["post_hoc_tests"][metric_name] = post_hoc
                            
                except Exception as e:
                    self.logger.warning(f"Friedman test failed for {metric_name}: {e}")
                    friedman_results["results_by_metric"][metric_name] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        # Generate summary
        significant_count = sum(1 for r in friedman_results["results_by_metric"].values() 
                               if isinstance(r, dict) and r.get("significant", False))
        
        friedman_results["summary"] = {
            "metrics_tested": len(friedman_results["results_by_metric"]),
            "significant_results": significant_count,
            "post_hoc_performed": len(friedman_results["post_hoc_tests"])
        }
        
        return friedman_results
    
    def _perform_friedman_post_hoc(self, data_matrix: np.ndarray, model_ids: List[str]) -> Dict[str, Any]:
        """Perform post-hoc tests for Friedman test using Nemenyi test"""
        from scipy.stats import rankdata
        
        post_hoc_results = {
            "method": "nemenyi",
            "comparisons": []
        }
        
        # Rank data for each observation
        ranked_data = np.array([rankdata(row) for row in data_matrix])
        mean_ranks = np.mean(ranked_data, axis=0)
        
        # Calculate critical difference
        n_models = len(model_ids)
        n_observations = len(data_matrix)
        
        # Nemenyi critical value (approximate)
        q_alpha = 2.569  # For alpha=0.05 and k=3-10 groups
        cd = q_alpha * np.sqrt((n_models * (n_models + 1)) / (6 * n_observations))
        
        # Perform pairwise comparisons
        for i in range(n_models):
            for j in range(i + 1, n_models):
                rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                post_hoc_results["comparisons"].append({
                    "model1": model_ids[i],
                    "model2": model_ids[j],
                    "mean_rank1": float(mean_ranks[i]),
                    "mean_rank2": float(mean_ranks[j]),
                    "rank_difference": float(rank_diff),
                    "critical_difference": float(cd),
                    "significant": rank_diff > cd
                })
        
        post_hoc_results["critical_difference"] = float(cd)
        post_hoc_results["mean_ranks"] = {model_ids[i]: float(mean_ranks[i]) for i in range(n_models)}
        
        return post_hoc_results
    
    def perform_wilcoxon_signed_rank_tests(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Wilcoxon signed-rank tests for paired comparisons"""
        wilcoxon_results = {
            "method": "wilcoxon_signed_rank",
            "results": {},
            "summary": {}
        }
        
        # Perform tests for each pair using aligned data
        for model1, model2 in prepared_data["comparison_pairs"]:
            pair_key = f"{model1}_vs_{model2}"
            wilcoxon_results["results"][pair_key] = {}
            
            for metric_name in prepared_data["aligned_data"]:
                if model1 in prepared_data["aligned_data"][metric_name] and \
                   model2 in prepared_data["aligned_data"][metric_name]:
                    
                    values1 = prepared_data["aligned_data"][metric_name][model1]
                    values2 = prepared_data["aligned_data"][metric_name][model2]
                    
                    if len(values1) == len(values2) and len(values1) >= 6:  # Minimum for Wilcoxon
                        try:
                            # Perform Wilcoxon signed-rank test
                            stat, p_value = stats.wilcoxon(values1, values2, alternative='two-sided')
                            
                            # Calculate effect size (r = Z / sqrt(N))
                            n = len(values1)
                            z_score = stats.norm.ppf(1 - p_value/2) if p_value < 1 else 0
                            effect_size = abs(z_score) / np.sqrt(n)
                            
                            # Calculate median difference
                            differences = values1 - values2
                            median_diff = np.median(differences)
                            
                            wilcoxon_results["results"][pair_key][metric_name] = {
                                "statistic": float(stat),
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "effect_size": float(effect_size),
                                "effect_interpretation": self._interpret_effect_size_r(effect_size),
                                "median_difference": float(median_diff),
                                "n_pairs": n,
                                "n_zero_differences": int(np.sum(differences == 0))
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"Wilcoxon test failed for {pair_key} - {metric_name}: {e}")
                            wilcoxon_results["results"][pair_key][metric_name] = {
                                "status": "error",
                                "error": str(e)
                            }
        
        # Generate summary
        total_tests = 0
        significant_tests = 0
        
        for pair_results in wilcoxon_results["results"].values():
            for test_result in pair_results.values():
                if isinstance(test_result, dict) and "p_value" in test_result:
                    total_tests += 1
                    if test_result["significant"]:
                        significant_tests += 1
        
        wilcoxon_results["summary"] = {
            "total_tests": total_tests,
            "significant_tests": significant_tests,
            "test_type": "paired_non_parametric"
        }
        
        return wilcoxon_results
    
    def calculate_comparative_effect_sizes(self, prepared_data: Dict[str, Any], 
                                          pairwise_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive effect sizes for all comparisons"""
        effect_sizes = {
            "pairwise_effects": {},
            "overall_effects": {},
            "effect_interpretations": {}
        }
        
        # Calculate pairwise effect sizes
        for pair_key, metrics in pairwise_results.items():
            effect_sizes["pairwise_effects"][pair_key] = {}
            models = pair_key.split("_vs_")
            
            for metric_name, test_results in metrics.items():
                if "mean_difference" in test_results:
                    # Get the actual data
                    values1 = prepared_data["metrics"][metric_name][models[0]]
                    values2 = prepared_data["metrics"][metric_name][models[1]]
                    
                    if isinstance(values1, np.ndarray) and isinstance(values2, np.ndarray):
                        # Cohen's d
                        cohens_d = self._calculate_cohens_d(values1, values2)
                        
                        # Hedges' g
                        hedges_g = self._calculate_hedges_g(values1, values2)
                        
                        # Common Language Effect Size
                        cles = self._calculate_cles(values1, values2)
                        
                        # Rank-biserial correlation (for Mann-Whitney)
                        u_stat = test_results["non_parametric"]["statistic"]
                        n1, n2 = len(values1), len(values2)
                        r_rb = 1 - (2 * u_stat) / (n1 * n2)
                        
                        effect_sizes["pairwise_effects"][pair_key][metric_name] = {
                            "cohens_d": float(cohens_d),
                            "hedges_g": float(hedges_g),
                            "cles": float(cles),
                            "rank_biserial": float(r_rb),
                            "interpretations": {
                                "cohens_d": self._interpret_cohens_d(cohens_d),
                                "hedges_g": self._interpret_cohens_d(hedges_g),
                                "cles": f"{cles:.1%} probability of superiority",
                                "rank_biserial": self._interpret_rank_biserial(r_rb)
                            }
                        }
        
        # Calculate overall effect sizes (eta-squared for multiple groups)
        for metric_name, model_values in prepared_data["metrics"].items():
            if len(model_values) >= 3:
                # Calculate eta-squared
                all_values = []
                group_means = []
                group_sizes = []
                
                for values in model_values.values():
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        all_values.extend(values)
                        group_means.append(np.mean(values))
                        group_sizes.append(len(values))
                
                if len(group_means) >= 3:
                    grand_mean = np.mean(all_values)
                    
                    # Between-group sum of squares
                    ss_between = sum(n * (mean - grand_mean)**2 
                                   for n, mean in zip(group_sizes, group_means))
                    
                    # Total sum of squares
                    ss_total = np.sum((np.array(all_values) - grand_mean)**2)
                    
                    # Eta-squared
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    # Omega-squared (less biased)
                    k = len(group_means)
                    N = sum(group_sizes)
                    ms_between = ss_between / (k - 1)
                    ms_within = (ss_total - ss_between) / (N - k)
                    omega_squared = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0
                    omega_squared = max(0, omega_squared)  # Can't be negative
                    
                    effect_sizes["overall_effects"][metric_name] = {
                        "eta_squared": float(eta_squared),
                        "omega_squared": float(omega_squared),
                        "interpretations": {
                            "eta_squared": self._interpret_eta_squared(eta_squared),
                            "omega_squared": self._interpret_eta_squared(omega_squared)
                        }
                    }
        
        return effect_sizes
    
    def generate_significance_matrix(self, prepared_data: Dict[str, Any], 
                                    results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate significance matrix for all model comparisons"""
        model_ids = list(prepared_data["models"].keys())
        n_models = len(model_ids)
        
        significance_matrix = {
            "models": model_ids,
            "matrices_by_metric": {},
            "summary_matrix": {}
        }
        
        # Get all metrics
        metrics = list(prepared_data["metrics"].keys())
        
        for metric_name in metrics:
            # Initialize matrix
            matrix = np.zeros((n_models, n_models))
            p_value_matrix = np.ones((n_models, n_models))
            
            # Fill matrix with pairwise comparison results
            for i, model1 in enumerate(model_ids):
                for j, model2 in enumerate(model_ids):
                    if i != j:
                        # Find the comparison result
                        pair_key = f"{model1}_vs_{model2}" if i < j else f"{model2}_vs_{model1}"
                        
                        if pair_key in results["pairwise_comparisons"]:
                            if metric_name in results["pairwise_comparisons"][pair_key]:
                                p_value = results["pairwise_comparisons"][pair_key][metric_name]["parametric"]["p_value"]
                                p_value_matrix[i, j] = p_value
                                
                                # Check if significant (considering multiple comparison corrections if available)
                                if "bonferroni" in results["multiple_comparisons"]:
                                    corrected_results = results["multiple_comparisons"]["bonferroni"]["corrected_results"]
                                    if pair_key in corrected_results and metric_name in corrected_results[pair_key]:
                                        is_significant = corrected_results[pair_key][metric_name]["significant_after_correction"]["parametric"]
                                        matrix[i, j] = 1 if is_significant else 0
                                else:
                                    matrix[i, j] = 1 if p_value < 0.05 else 0
            
            significance_matrix["matrices_by_metric"][metric_name] = {
                "significance_matrix": matrix.tolist(),
                "p_value_matrix": p_value_matrix.tolist(),
                "interpretation": self._interpret_significance_matrix(matrix, model_ids)
            }
        
        # Create summary matrix (significant in any metric)
        summary_matrix = np.zeros((n_models, n_models))
        for metric_matrices in significance_matrix["matrices_by_metric"].values():
            summary_matrix = np.maximum(summary_matrix, np.array(metric_matrices["significance_matrix"]))
        
        significance_matrix["summary_matrix"] = {
            "matrix": summary_matrix.tolist(),
            "interpretation": self._interpret_significance_matrix(summary_matrix, model_ids)
        }
        
        return significance_matrix
    
    def calculate_model_rankings(self, prepared_data: Dict[str, Any], 
                                results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive model rankings based on all analyses"""
        rankings = {
            "by_metric": {},
            "overall_rankings": {},
            "ranking_methods": {}
        }
        
        model_ids = list(prepared_data["models"].keys())
        
        # Rank by each metric
        for metric_name, model_values in prepared_data["metrics"].items():
            metric_rankings = []
            
            for model_id in model_ids:
                if model_id in model_values:
                    values = model_values[model_id]
                    if isinstance(values, np.ndarray) and len(values) > 0:
                        metric_rankings.append({
                            "model": model_id,
                            "mean": float(np.mean(values)),
                            "median": float(np.median(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        })
            
            # Sort by mean performance
            metric_rankings.sort(key=lambda x: x["mean"], reverse=True)
            
            # Add ranks
            for i, ranking in enumerate(metric_rankings):
                ranking["rank"] = i + 1
            
            rankings["by_metric"][metric_name] = metric_rankings
        
        # Calculate overall rankings using different methods
        
        # Method 1: Average rank across metrics
        model_ranks = {model_id: [] for model_id in model_ids}
        for metric_rankings in rankings["by_metric"].values():
            for ranking in metric_rankings:
                model_ranks[ranking["model"]].append(ranking["rank"])
        
        average_ranks = []
        for model_id, ranks in model_ranks.items():
            if ranks:
                average_ranks.append({
                    "model": model_id,
                    "average_rank": float(np.mean(ranks)),
                    "rank_std": float(np.std(ranks)) if len(ranks) > 1 else 0.0,
                    "best_rank": min(ranks),
                    "worst_rank": max(ranks)
                })
        
        average_ranks.sort(key=lambda x: x["average_rank"])
        for i, ranking in enumerate(average_ranks):
            ranking["final_rank"] = i + 1
        
        rankings["ranking_methods"]["average_rank"] = average_ranks
        
        # Method 2: Win-rate based ranking (from pairwise comparisons)
        if "pairwise_comparisons" in results:
            win_rates = {model_id: {"wins": 0, "total": 0} for model_id in model_ids}
            
            for pair_key, metrics in results["pairwise_comparisons"].items():
                models = pair_key.split("_vs_")
                if len(models) == 2:
                    model1, model2 = models
                    
                    for metric_results in metrics.values():
                        if "mean_difference" in metric_results:
                            win_rates[model1]["total"] += 1
                            win_rates[model2]["total"] += 1
                            
                            if metric_results["mean_difference"] > 0:
                                win_rates[model1]["wins"] += 1
                            else:
                                win_rates[model2]["wins"] += 1
            
            win_rate_rankings = []
            for model_id, stats in win_rates.items():
                if stats["total"] > 0:
                    win_rate_rankings.append({
                        "model": model_id,
                        "win_rate": float(stats["wins"] / stats["total"]),
                        "wins": stats["wins"],
                        "total_comparisons": stats["total"]
                    })
            
            win_rate_rankings.sort(key=lambda x: x["win_rate"], reverse=True)
            for i, ranking in enumerate(win_rate_rankings):
                ranking["rank"] = i + 1
            
            rankings["ranking_methods"]["win_rate"] = win_rate_rankings
        
        # Method 3: Significance-based ranking (how many models each significantly outperforms)
        if "significance_matrix" in results:
            significance_scores = {model_id: 0 for model_id in model_ids}
            
            summary_matrix = np.array(results["significance_matrix"]["summary_matrix"]["matrix"])
            for i, model_id in enumerate(model_ids):
                # Count how many models this model significantly outperforms
                significance_scores[model_id] = int(np.sum(summary_matrix[i, :]))
            
            significance_rankings = [
                {"model": model_id, "significance_score": score, "outperforms_count": score}
                for model_id, score in significance_scores.items()
            ]
            
            significance_rankings.sort(key=lambda x: x["significance_score"], reverse=True)
            for i, ranking in enumerate(significance_rankings):
                ranking["rank"] = i + 1
            
            rankings["ranking_methods"]["significance_based"] = significance_rankings
        
        # Combine rankings for overall assessment
        overall_scores = {}
        ranking_methods = ["average_rank", "win_rate", "significance_based"]
        
        for model_id in model_ids:
            overall_scores[model_id] = {
                "ranks": {},
                "scores": {}
            }
            
            for method in ranking_methods:
                if method in rankings["ranking_methods"]:
                    for ranking in rankings["ranking_methods"][method]:
                        if ranking["model"] == model_id:
                            if method == "average_rank":
                                overall_scores[model_id]["ranks"][method] = ranking["final_rank"]
                                overall_scores[model_id]["scores"][method] = 1 / ranking["average_rank"]
                            else:
                                overall_scores[model_id]["ranks"][method] = ranking.get("rank", 999)
                                if method == "win_rate":
                                    overall_scores[model_id]["scores"][method] = ranking.get("win_rate", 0)
                                elif method == "significance_based":
                                    overall_scores[model_id]["scores"][method] = ranking.get("significance_score", 0)
        
        # Calculate consensus ranking
        consensus_rankings = []
        for model_id, scores in overall_scores.items():
            if scores["ranks"]:
                consensus_rankings.append({
                    "model": model_id,
                    "average_rank_across_methods": float(np.mean(list(scores["ranks"].values()))),
                    "rank_agreement": float(1 - np.std(list(scores["ranks"].values())) / np.mean(list(scores["ranks"].values()))) if np.mean(list(scores["ranks"].values())) > 0 else 0,
                    "method_ranks": scores["ranks"]
                })
        
        consensus_rankings.sort(key=lambda x: x["average_rank_across_methods"])
        for i, ranking in enumerate(consensus_rankings):
            ranking["consensus_rank"] = i + 1
        
        rankings["overall_rankings"] = consensus_rankings
        
        return rankings
    
    # Helper methods for effect size calculations
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)"""
        cohens_d = self._calculate_cohens_d(group1, group2)
        
        # Correction factor
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1)
        
        return cohens_d * correction
    
    def _calculate_cles(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Common Language Effect Size"""
        # Probability that a randomly selected score from group1 > group2
        comparisons = 0
        total = 0
        
        for val1 in group1:
            for val2 in group2:
                total += 1
                if val1 > val2:
                    comparisons += 1
                elif val1 == val2:
                    comparisons += 0.5
        
        return comparisons / total if total > 0 else 0.5
    
    # Interpretation helper methods
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_rank_biserial(self, r_rb: float) -> str:
        """Interpret rank-biserial correlation"""
        abs_r = abs(r_rb)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta squared effect size"""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """Interpret correlation-based effect size"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_significance_matrix(self, matrix: np.ndarray, model_ids: List[str]) -> Dict[str, Any]:
        """Interpret significance matrix patterns"""
        interpretation = {
            "significant_pairs": [],
            "model_dominance": {},
            "clustering": []
        }
        
        # Find significant pairs
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                if matrix[i, j] > 0 or matrix[j, i] > 0:
                    interpretation["significant_pairs"].append({
                        "model1": model_ids[i],
                        "model2": model_ids[j]
                    })
        
        # Calculate model dominance (how many other models each model significantly differs from)
        for i, model_id in enumerate(model_ids):
            dominance_score = np.sum(matrix[i, :]) + np.sum(matrix[:, i])
            interpretation["model_dominance"][model_id] = {
                "score": int(dominance_score),
                "percentage": float(dominance_score / (2 * (len(model_ids) - 1)))
            }
        
        return interpretation