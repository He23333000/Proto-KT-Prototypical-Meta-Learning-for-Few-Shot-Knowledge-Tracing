# Experimental Plan for Proto-KT

**Status**: Pipeline Running  


---

## 🎯 Research Questions (from main.tex)

### Q1: Main Performance
**Does Proto-KT significantly outperform single-initialization MAML?**

- **Hypothesis**: Conditional multi-prototype initialization will outperform single universal initialization
- **Expected**: Proto-KT > MAML-SAKT > SAKT Baseline
- **Metrics**: AUC, Accuracy across windows (1-10, 1-20, 1-50, overall)

### Q2: Ablation Analysis
**How many prototypes are necessary?**

- **Hypothesis**: Performance improves from k=1 to k=8, then plateaus
- **Test**: k ∈ {1, 2, 4, 8, 16}
- **Expected**: Diminishing returns beyond k=8

### Q3: Interpretability
**Do learned prototypes correspond to meaningful student behaviors?**

- **Hypothesis**: Prototypes align with educationally meaningful archetypes
- **Analysis**: UMAP visualization, prototype characterization
- **Expected**: 3 clusters (high performers, medium learners, strugglers)

---

## 📊 Complete Experimental Protocol

### Phase 1: Model Training (3-5 days with 5% data)

#### Baselines:
1. **SAKT Baseline**
   - Standard SAKT pre-trained on meta-training set
   - Fine-tuned on each test student
   - Purpose: Establishes performance without meta-learning

2. **MAML-SAKT** 
   - MAML with single universal initialization
   - Purpose: Direct comparison to single-initialization paradigm

#### Main Model:
3. **Proto-KT (k=8)**
   - Multi-prototype conditional initialization
   - Purpose: Proposed method

#### Ablation Variants:
4-8. **Proto-KT (k=1, 2, 4, 16)**
   - Purpose: Validate necessity of multiple prototypes

### Phase 2: Evaluation (1 day)

For each model:
- Evaluate on held-out test students (227 students)
- Metrics per interaction window:
  - 1-10 interactions (critical cold-start)
  - 1-20 interactions
  - 1-50 interactions  
  - Overall (2-51)
- Compute:
  - AUC (primary metric)
  - Accuracy
  - BCE Loss
  - ECE (calibration)

### Phase 3: Statistical Analysis (1 day)

- Bootstrap confidence intervals (1000 samples)
- Paired t-tests (Proto-KT vs baselines)
- Bonferroni correction for multiple comparisons
- Effect size computation (Cohen's d)

### Phase 4: Interpretability (1 day)

For Proto-KT (k=8):
- Compute student→prototype assignments
- UMAP projection of context embeddings
- Characterize each prototype:
  - Number of assigned students
  - Average initial accuracy
  - Learning rate patterns
  - Semantic interpretation

---

## 📁 Expected Outputs

### Tables (LaTeX format, ready for main.tex):

**Table 1: Main Results Comparison**
```latex
\begin{table}[t]
\centering
\caption{Few-shot performance comparison...}
\label{tab:main_results}
\begin{tabular}{l|cc|cc|cc|cc}
\toprule
Method & \multicolumn{2}{c|}{1-10} & \multicolumn{2}{c|}{1-20} & 
         \multicolumn{2}{c|}{1-50} & \multicolumn{2}{c}{Overall} \\
       & AUC & Acc & AUC & Acc & AUC & Acc & AUC & Acc \\
\midrule
SAKT Baseline  & ... \\
MAML-SAKT      & ... \\
Proto-KT (k=8) & ... \\
\bottomrule
\end{tabular}
\end{table}
```

**Table 2: Ablation Study**
```latex
\begin{table}[t]
\centering
\caption{Effect of prototype count...}
\label{tab:ablation}
\begin{tabular}{l|ccc}
\toprule
\# Prototypes & AUC@10 & AUC@20 & AUC (Overall) \\
\midrule
k=1 (MAML) & ... \\
k=2        & ... \\
k=4        & ... \\
k=8        & ... \\
k=16       & ... \\
\bottomrule
\end{tabular}
\end{table}
```

**Table 3: Prototype Characteristics**
```latex
\begin{table}[t]
\centering
\caption{Learned prototype characteristics...}
\label{tab:prototypes}
\begin{tabular}{l|ccl}
\toprule
Prototype & \# Students & Init. Acc. & Interpretation \\
\midrule
Prototype 1 & ... & ... & High performers \\
Prototype 2 & ... & ... & Medium learners \\
Prototype 3 & ... & ... & Strugglers \\
\bottomrule
\end{tabular}
\end{table}
```

### Figures (PDF format):

**Figure 1: Learning Curves**
- X-axis: Number of interactions (1-51)
- Y-axis: AUC
- Lines: SAKT (orange), MAML (purple), Proto-KT (blue)
- Shows: Proto-KT's superior cold-start adaptation

**Figure 2: Prototype Visualization**
- UMAP 2D projection
- Points: Test students colored by assigned prototype
- Stars: Prototype centers
- Shows: Clear cluster separation

---

## 🎯 Success Criteria

### Minimum (for submission):
- [ ] Proto-KT > MAML > SAKT across all windows
- [ ] Statistical significance (p < 0.05)
- [ ] Ablation shows k=8 is near-optimal
- [ ] Prototypes show clear clustering

### Ideal (strong accept):
- [ ] Proto-KT improves AUC@10 by >10% vs MAML
- [ ] Consistent across multiple datasets
- [ ] Prototypes align with known educational theory
- [ ] Code released, fully reproducible

---

## 🚀 Current Status

**Running**: Complete experimental pipeline on 5% data subset  
**ETA**: 3-5 hours  
**Log**: `proto_kt/results/neurips_5pct/experiment_log.json`  
**Checkpoints**: `proto_kt/checkpoints/5pct/`

**Pipeline Steps**:
1. ✅ Fix MAML training bug (shape mismatch)
2. 🔄 Train SAKT baseline
3. ⏳ Train MAML-SAKT
4. ⏳ Train Proto-KT (k=1,2,4,8,16)
5. ⏳ Run evaluations
6. ⏳ Generate tables/figures
7. ⏳ Statistical analysis

**Monitor Progress**:
```bash
# Check running status
python -c "import json; print(json.dumps(json.load(open('proto_kt/results/neurips_5pct/experiment_log.json')), indent=2))"

# Or tail the terminal log
Get-Content terminals/5.txt -Wait -Tail 20
```

---

## 📝 Integration with main.tex

Once experiments complete, update main.tex:

### Section 4.3: Main Results
- Replace simulated numbers in narrative
- Copy `table_1_main_results.tex` to `proto_kt/results/`
- Update learning curves figure path

### Section 4.4: Ablation Study
- Replace simulated k values
- Copy `table_2_ablation.tex` to `proto_kt/results/`
- Update analysis text with actual trends

### Section 4.5: Interpretability
- Replace prototype counts/characteristics
- Copy `table_3_prototypes.tex` and `prototype_visualization.pdf`
- Update semantic interpretations based on actual clusters

### Section 5: Limitations
- Add disclaimer if using subset (<100% data)
- Note any unexpected findings
- Discuss computational requirements

##  Reviewer Expectations

**What reviewers will look for**:
1. ✅ Novel contribution (multi-prototype initialization)
2. ✅ Strong empirical validation (real data, multiple baselines)
3. ✅ Rigorous evaluation (statistical tests, ablations)
4. ✅ Interpretability (not black box)
5. ✅ Reproducibility (complete specs)
6. ✅ Writing quality (clear, well-motivated)

**Potential concerns** (already addressed):
- "Only on one dataset" → Plan to acknowledge in limitations
- "Computational cost" → Provide first-order MAML option
- "Prototype interpretability" → Full visualization + analysis
- "Statistical significance" → Bootstrap CI + paired tests
- "Reproducibility" → Code release + complete specs

---

**Last Updated**: Nov 30, 2025, 10:30 PM  
**Next Milestone**: Complete 5% validation run (~4 hours)  
**Critical Decision**: After validation, choose 10% or 100% for final results

