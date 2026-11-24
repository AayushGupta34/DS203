# DS203 one-day exam review

Use the companion **DS203_Crash_Course.ipynb** to learn and run the examples. Use this sheet for written recall and the mock exam. These priorities come from the supplied coursework. The actual exam weighting is unknown.

| Time | Study block | Produce from memory before moving on |
|---|---|---|
| 75 min | Python, NumPy, pandas, plotting and exploratory data analysis | Read a table, filter it, group it, plot it and explain what you found |
| 60 min | Statistics, simulation, the CLT and Bayes | Calculate a posterior and explain how the spread of sample means changes with sample size |
| 110 min | OLS, inference, regularization, cross-validation and pipelines | Calculate a line by hand and write a train/CV/test workflow |
| 95 min | Classification, LDA, KNN, metrics and SVM | Classify a point by hand and calculate metrics from a confusion matrix |
| 55 min | K-means and PCA | Do one Lloyd iteration and interpret explained variance |
| 75 min | Closed-note mock and corrections | Rewrite every missed answer without looking at the key |

This is about eight hours of focused work. Add breaks and leave time to sleep. If you have less time, do the coding workflow and the regression and classification calculations first. Reproduce each example once with the code hidden and change one input to check your understanding.

## Python, data and statistics

- Python starts indexing at zero. `a[1:4]` selects positions 1, 2, and 3. `a[-1]` selects the last item. `//` floors the quotient, so `-3 // 2 == -2`. Two names can reference the same mutable list. Use `a.copy()` for a shallow copy. `==` compares values and `is` compares identity. Use `None` as a function default when you need a fresh list on each call.
- NumPy arrays support elementwise arithmetic. An array of shape `(n,p)` has n rows and p columns. `axis=0` reduces down rows to give one result per column. `axis=1` gives one result per row. Boolean masks select entries. The mean of a Boolean mask is the fraction that passed. `*` multiplies elementwise and `@` performs matrix multiplication. Broadcasting aligns trailing dimensions whose sizes agree or are 1.
- Python strings are immutable. `line.strip().lower().split()` cleans the ends, lowercases and splits on whitespace. `counts[w] = counts.get(w,0)+1` counts tokens. Use `with open(path, encoding="utf-8") as f:` to close a file automatically.
- A pandas Series is a labeled one-dimensional column. A DataFrame is a labeled table. `df.loc[mask, "col"]` uses labels and `df.iloc[:3, :2]` uses positions. Pandas aligns Series by index in arithmetic. Inspect with `head`, `info`, `describe`, `isna().sum()` and `value_counts`. `df.groupby("group")["value"].mean()` calculates each group's mean. Use `df.loc[mask,"col"] = value` for assignment. Use `merge` to join by a key and `concat` to stack compatible tables.
- Use a histogram for a distribution, a box plot for median and spread across groups, a scatter plot for two numerical variables, and a bar chart for category counts. Check types, duplicates, missingness and units. For an IQR flag use bounds Q1 − 1.5 IQR and Q3 + 1.5 IQR. A flagged value may be valid. Pearson correlation measures linear association, Spearman measures monotonic association using ranks, and neither establishes causality. An aggregate trend can reverse inside groups.
- Nominal categories have no order. Ordinal categories have an order with potentially unequal gaps. Interval data have meaningful differences and an arbitrary zero. Ratio data have a meaningful zero. One-hot encode nominal categories, use meaningful order for ordinal categories, and scale numeric features when units affect distances or penalties. Median imputation is less sensitive to extremes than mean imputation. Mode imputation is one option for categoricals. A missingness indicator can carry information. Fit data-dependent preprocessing on training data.
- Sample mean is sum/n. Sample variance uses denominator n−1. NumPy defaults to denominator n unless you use `ddof=1`. Standard deviation describes spread among observations. Standard error of the sample mean estimates how much the mean varies across repeated samples, approximately s/√n for independent observations. Expected value for a discrete variable is ΣxP(X=x). Var(X) = E[X²] − E[X]².
- P(A | B) = P(A ∩ B)/P(B). If A and B are independent their joint probability is P(A)P(B). A Bernoulli(p) variable is 0 or 1 with mean p and variance p(1−p). A Binomial(n,p) variable counts successes in n independent Bernoulli trials with common p, so its mean and variance are np and np(1−p). A density integrates to probability over an interval. The probability at any one point of a continuous distribution is zero. Standardizing a normal observation by (x−μ)/σ gives a standard normal.
- A confidence interval is constructed by a procedure that covers the fixed population parameter in the stated fraction of repeated samples under its assumptions. For the mean under the usual normal-sample model with unknown variance, use x̄ ± t* s/√n. For a Monte Carlo estimate, simulate independent repetitions, record the statistic or event each time, and summarize those repetitions.
- LLN gives convergence of sample averages to the expectation under usual i.i.d. finite-mean conditions. The classical i.i.d. CLT says the centered and standardized sample mean approaches a standard normal when variance is finite and positive. Raw observations can stay highly skewed. For Pareto shape b>0, mean is finite for b>1, variance for b>2, skewness for b>3. The classical finite-variance CLT therefore applies for b>2. The assignment contains a reversed inequality in one sentence.
- Bayes' rule is posterior = likelihood × prior / evidence. For a grid, normalize the products to sum to one when treating grid points as probability masses. With a Beta(a,b) prior and h heads, t tails, the posterior is Beta(a+h,b+t). Under squared error, the Bayes estimate is the posterior mean. A zero prior stays zero after updating. The current posterior becomes the prior for the next independent batch conditional on the parameter.

## Regression and model selection

- A supervised model learns to predict a supplied target. Regression predicts a quantity, classification a class. Unsupervised methods can discover clusters or directions of variation without a supplied target. Parameters are learned during fitting. Hyperparameters, such as K or alpha, are set outside that fitting and can be selected by validation.
- OLS fits ŷ = b₀ + b₁x by minimizing SSE = Σ(y − ŷ)². For one predictor, b₁ = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)² and b₀ = ȳ − b₁x̄. In matrix form XᵀXb = Xᵀy with a ones column for an intercept. The inverse formula needs full column rank. In code use a library or `np.linalg.lstsq`. A residual is observed minus fitted.
- MSE = SSE/n, RMSE = √MSE in target units, MAE = mean absolute error. R² = 1 − SSE/SST with SST = Σ(y−ȳ)². It can be negative on a test set. Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1), with p predictors excluding the intercept. It penalizes adding predictors. Compare held-out scores for predictive performance.
- A coefficient is the expected change in predicted y per unit x holding other modeled features fixed, subject to interactions. In ŷ = 2 + 3x + 5G − 2xG, G=0 gives 2+3x and G=1 gives 7+x. A dummy coefficient compares to the omitted category at x=0 when there is an interaction. Polynomial regression uses powers as features and stays linear in the coefficients. For log(y) on x, increasing x by one multiplies the fitted response on the exponentiated scale by exp(b).
- Classical OLS conditions include a correct linear-in-parameters conditional mean, zero conditional mean error, no perfect collinearity, and constant variance with uncorrelated errors for the standard Gauss–Markov result. Normal errors support exact classical small-sample t/F inference. In residual-versus-fitted plots, curvature suggests a missing nonlinear term and a funnel suggests changing variance. A Q–Q plot checks the normality approximation.
- To test one coefficient, H₀: βⱼ=0 and t = estimate/SE. A 95% CI is estimate ± critical t × SE. If its two-sided p-value is below 0.05, reject H₀ at 5%. A p-value is computed assuming H₀, from how extreme the observed statistic is. It does not measure effect size. An F-test can test several coefficients together using nested models. VIFⱼ = 1/(1−Rⱼ²), where Rⱼ² comes from regressing predictor j on the other predictors. High VIF signals imprecise individual coefficients under the classical model.
- Ridge adds αΣbⱼ², lasso adds αΣ|bⱼ| and can set coefficients exactly to zero. Elastic net combines both. Standardize predictors so the penalty treats them on comparable scales. More regularization usually raises bias and lowers variance. For squared prediction error, expected error decomposes into squared bias, variance, and irreducible noise.
- Split off the test set first. In K-fold CV, fit on K−1 training folds and score on the remaining fold for each candidate. Average the scores, choose the best candidate, refit on all training rows, and evaluate on the test set once. Use stratified folds for ordinary classification when class balance matters. Time-ordered or grouped observations need a split that respects their structure.
- Learn scaling, imputing, encoding and PCA separately inside every training fold. Put transformations in a pipeline before the estimator, then wrap the entire pipeline in `GridSearchCV`. Parameters use `step__parameter`. For `neg_mean_squared_error`, a higher score is better and negating it gives MSE. Very good train scores with poor validation scores suggest high variance. Poor scores on both suggest high bias. A validation curve varies model complexity and a learning curve varies training sample size.

## Classification and unsupervised learning

- Bayes classifier compares P(class) × p(x | class) under equal misclassification costs. LDA assumes class-conditional multivariate normals with class-specific means and a shared covariance. It estimates priors, means, and a pooled covariance from training data. Its score is δₖ(x)=xᵀΣ⁻¹μₖ−½μₖᵀΣ⁻¹μₖ+log πₖ. Pick the largest. Shared covariance makes pairwise decision boundaries linear. Class-specific covariances yield the quadratic version, QDA.
- KNN computes distances to training rows, finds K closest, and predicts the majority class or mean response. Euclidean distance is √Σ(xⱼ−zⱼ)², Manhattan is Σ|xⱼ−zⱼ|. Scaling keeps a variable with large numerical units from dominating. Small K tends toward low bias and high variance. Large K smooths more and can underfit. In high dimensions, distance often becomes less informative.
- Logistic regression uses p=1/(1+exp(−z)), z=b₀+xᵀb. A unit increase in xⱼ changes log odds by bⱼ holding others fixed and multiplies odds by exp(bⱼ). Predict a positive when p meets the chosen threshold. The default threshold is a starting point and can be tuned on validation data for the desired costs or metrics.
- A linear classifier separates classes with wᵀx+b=0. Predict according to the sign of the score. An SVM seeks a large margin. Support vectors determine the boundary. Hard margin requires separation. Soft margin permits violations and C controls their penalty. Larger C penalizes violations more strongly. A kernel supplies similarities corresponding to a feature space. RBF K(x,z)=exp(−γ‖x−z‖²). Larger gamma makes each point's influence more local. Tune C and gamma by CV after scaling.
- Confusion matrix with rows actual [0,1] and columns predicted [0,1] is [[TN,FP],[FN,TP]]. Accuracy=(TP+TN)/total, precision=TP/(TP+FP), recall/TPR=TP/(TP+FN), specificity=TN/(TN+FP), FPR=FP/(FP+TN), F1=2TP/(2TP+FP+FN). ROC plots TPR vs FPR over thresholds. AUC measures ranking ability and can be interpreted as the chance a random positive outranks a random negative, with half credit for ties. With a rare positive class, examine precision and recall as well as accuracy or ROC-AUC.
- K-means minimizes within-cluster squared Euclidean distances to centroids. Lloyd's algorithm alternates between assignment to the nearest centroid and replacing each centroid with the mean of its assigned points. Initialization matters because the objective has local optima. K-means++ and multiple starts help. Inertia decreases or stays flat as K increases. Look for an elbow or compare silhouette scores (higher is better). Scaling matters.
- PCA centers the features and finds orthogonal directions in decreasing order of sample variance. The first loading is the unit eigenvector of the covariance matrix with largest eigenvalue. Scores are projections of centered rows onto the directions. Explained variance ratio for component j is λⱼ/Σλ. Standardize first when units or scale would otherwise dominate. PCA creates components, and K-means assigns cluster labels. Fit PCA inside the pipeline if the features go into a supervised model.

## Closed-note mock exam

Set 55 minutes. Spend about 25 minutes coding and 30 minutes writing. Grade and correct afterwards.

**Coding A.** Use `df = pd.DataFrame({"team":["A","B","A","B","A","B"],"runs":[0,12,50,None,80,45]})`. Report null counts, then drop rows with missing runs for this descriptive task. Compute by team the number of observed rows, average runs, and fraction with at least 50 runs. Plot one chart that compares the teams.

**Coding B.** Use `sklearn.datasets.load_breast_cancer(as_frame=True)`. Make a stratified train/test split. Write a pipeline for median imputation, scaling, and KNN. Tune K among 3, 5, 9 with three-fold CV. Report the selected K, test confusion matrix and test F1. Briefly justify where you fit the scaler. The dataset encodes malignant as 0, so report F1 for class 0.

**Written 1.** A disease affects 10%. A positive test occurs in 80% of disease cases and 20% of healthy cases. Find P(disease | positive) and the equal-cost Bayes decision. What changes if a false negative costs five times a false positive?

**Written 2.** OLS on points (1,2), (2,3), (3,7). Find slope, intercept, residuals, SSE and R². Explain what a residual funnel suggests. Compute VIF for an auxiliary R² of 0.90.

**Written 3.** For LDA in one dimension use equal priors, means 2 and 6, and common variance 4. Find the boundary and class at x=5. For 1-NN and 3-NN with points (1,A), (3,A), (5,B), (8,B), classify x=4.2. For linear SVM, explain the effect of increasing C.

**Written 4.** For TN=80, FP=10, FN=5, TP=5, calculate accuracy, precision, recall, FPR, and F1. Explain what happens to TPR and FPR when the threshold is lowered.

**Written 5.** Do one K-means assignment/update with points 1,2,7,10 and starting centers 1,7. If PCA eigenvalues are 5,3,2, how many components explain at least 75%? If the Pareto shape is 1.5, what can you say about the LLN and the classical CLT?

**Written 6.** A student fills missing values and scales the whole table, splits it, tries five different alphas, and reports whichever alpha has the lowest test MSE. Identify the two ways the test set influenced the result and write a corrected procedure.

## Answer key

**A.** One missing value in runs. After dropping it, A has n=3, mean=43⅓ and fraction 2/3. B has n=2, mean=28.5 and fraction 0. Example after importing pandas and matplotlib:

```python
print(df.isna().sum())
observed = df.dropna(subset=["runs"]).copy()
observed["fifty"] = observed["runs"] >= 50
summary = observed.groupby("team").agg(
    n=("runs", "size"), mean_runs=("runs", "mean"), p50=("fifty", "mean")
)
print(summary)
summary["mean_runs"].plot.bar(ylabel="Mean runs")
plt.tight_layout()
plt.show()
```

**B.** The scaler learns one mean and scale per feature from the training portion of each fold. The pipeline is refit on all training rows after selecting K. The numeric scores depend on the split and metric.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])
search = GridSearchCV(
    pipeline, {"knn__n_neighbors": [3, 5, 9]}, cv=3,
    scoring=make_scorer(f1_score, pos_label=0)
)
search.fit(X_train, y_train)
pred = search.predict(X_test)
print(search.best_params_)
print(confusion_matrix(y_test, pred, labels=[1, 0]))
print(f1_score(y_test, pred, pos_label=0))
```

**1.** 0.08/(0.08+0.18)=4/13≈0.308. Predict healthy at equal costs. With FP cost 1 and FN cost 5, expected costs of predicting disease and healthy are 9/13 and 20/13. Predict disease.

**2.** x̄=2, ȳ=4. Slope 5/2=2.5, intercept −1. Predictions 1.5,4,6.5. Residuals 0.5,−1,0.5. SSE=1.5, SST=14, R²=25/28≈0.893. A funnel suggests nonconstant error variance. VIF=10.

**3.** Equal priors and shared variance place the boundary at (2+6)/2=4. x=5 is assigned to the class with mean 6. 1-NN gives B from x=5. The three nearest points are 5,3,1, so 3-NN gives A. Larger C penalizes margin violations more and can lead to a narrower margin and sensitivity to difficult observations.

**4.** Accuracy .85, precision 1/3, recall .50, FPR 1/9≈.111, F1=.40. Lowering the threshold adds positive predictions, so both TPR and FPR can rise or stay flat.

**5.** Clusters {1,2} and {7,10}, new centers 1.5 and 8.5. Two components explain 80%. Pareto b=1.5 has a finite mean, so the usual i.i.d. LLN applies. Its infinite variance prevents use of the classical finite-variance CLT.

**6.** The preprocessing learned statistics from test rows. The alpha was selected directly on test outcomes. Split first, cross-validate the entire preprocessing/model pipeline on training rows, refit the selected candidate on all training rows, and use the untouched test set for the final estimate.

## Course files to revisit after your first pass

| Topic | Course materials |
|---|---|
| Python, plots, text and pandas | `01_getting_started.ipynb`, `02_simple_plots.ipynb`, `08_text_processing.ipynb`, `09_pandas.ipynb` |
| Simulation, Bayes and statistics | `10_hello_bayes (1).ipynb`, the Bayes and Pareto assignments, and the batting assignment |
| OLS, model diagnostics and selection | `11_regression.ipynb`, `13_Regression_extended.ipynb`, `14_Cross_validation.ipynb`, the regression slides inside the course ZIP |
| Classification | `15_Bayes_LDA.ipynb`, `15_KNN.ipynb`, and the classification slides inside the course ZIP, which cover SVM |
| Clustering, PCA and data preparation | `16_Unsupervised_kmeansPCA.ipynb` and the three notebooks numbered 17 |
