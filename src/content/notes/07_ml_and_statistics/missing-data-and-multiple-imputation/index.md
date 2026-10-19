---
title: "Missing data: MCAR, MAR, MNAR and why multiple imputation exists"
slug: missing-data-and-multiple-imputation
date: 2019-10-24
topic: ML & Statistics
tags: [university-lab, missing-data, imputation, multiple-imputation, bias]
summary: "The three ways data can go missing, which of them imputation can actually handle, and why imputing several times beats filling each gap once."
sources:
  - title: "732A93 Statistical Methods (course assignment)"
    author: "Linköping University"
    type: course
  - title: "Data Analysis Using Regression and Multilevel/Hierarchical Models, Chapter 25 (Missing-data imputation)"
    author: "Andrew Gelman, Jennifer Hill"
    url: "http://www.stat.columbia.edu/~gelman/arm/missing.pdf"
    type: book
  - title: "Inference and missing data"
    author: "Donald B. Rubin"
    url: "https://doi.org/10.1093/biomet/63.3.581"
    type: paper
  - title: "Flexible Imputation of Missing Data (2nd ed.)"
    author: "Stef van Buuren"
    url: "https://stefvanbuuren.name/fimd/"
    type: book


status: draft
---

Almost every real dataset has holes. The tempting fixes are to drop incomplete rows or fill each hole with a best guess. Whether either is OK depends much less on the method than on *why* the values are missing. I wrote about this for a statistics assignment based on Gelman and Hill's chapter on imputation, and I got the key question backwards at first.

## Why values go missing

Rubin's classic terms, using "income not reported" as the example:

- **MCAR, missing completely at random**: missingness has nothing to do with anything, observed or not. A page of the survey got lost. The complete rows are a random subsample.
- **MAR, missing at random**: missingness depends only on things you *observed*. Older people skip the income question more often, and you know everyone's age. Within each age, the missing incomes look like the observed ones.
- **MNAR, missing not at random**: missingness depends on the missing value itself, or on something you didn't record. High earners hide their income. Gelman and Hill split this into two cases (depends on an unobserved predictor; depends on the value itself), which is why their list has four mechanisms.

You can't tell MAR from MNAR using the data alone, because the evidence you'd need is exactly what's missing. It's an assumption you argue for with knowledge of how the data were collected.

## Which mechanism makes imputation work?

The assignment asked which mechanism you'd prefer for a good imputation. I answered that the more systematic the missingness (MNAR), the better, because there's more pattern to model. That's wrong, and it's worth saying why.

Imputation models predict the missing values from the observed ones. That only works if, given the observed variables, the missing values behave like the observed ones. That's the definition of MAR. Under MNAR the pattern you'd need to model lives in the *unobserved* values. High earners hide their income, so the incomes you can see are biased low at every age, and a regression fitted to them inherits that bias. You can only fix MNAR with an explicit model of the missingness itself (selection models, pattern-mixture models) plus assumptions you can't check, or by collecting more variables so that MAR becomes believable.

So the ranking for imputation is: MCAR is easiest, MAR is what standard imputation is designed for, and MNAR is the hard case.

```r
set.seed(9)
n <- 5000
age    <- rnorm(n, 45, 12)
income <- 20 + 0.8 * age + rnorm(n, sd = 8)        # true mean income is 56

miss <- list(                                       # who doesn't report income?
  MCAR = runif(n) < 0.3,                            # pure chance
  MAR  = runif(n) < plogis((age - 45) / 5),         # older people; age is observed
  MNAR = runif(n) < plogis((income - 56) / 5))      # high earners; depends on income itself

sapply(miss, function(m) {
  fit <- lm(income ~ age, subset = !m)              # regression imputation from age
  imp <- income; imp[m] <- predict(fit, data.frame(age = age[m]))
  c(complete_case = mean(income[!m]), regression_imp = mean(imp))
})
#                 MCAR   MAR  MNAR
# complete_case   56.3  50.1  47.7
# regression_imp  56.2  56.1  51.2
```

Dropping rows is fine under MCAR and biased under MAR. Regression imputation repairs MAR, because age carries the information about who's missing. Under MNAR both are biased, and imputation only removes part of the gap.

## Single imputation and its problem

Two simple ways to fill one variable:

- **Simple random imputation**: replace each missing value with a randomly drawn observed value of the same variable. It keeps the variable's distribution but ignores how it relates to other variables, so correlations get diluted.
- **Regression imputation**: fit a model on the complete cases and fill in its predictions. Better, but the filled values sit exactly on the regression line. The data look less noisy than reality, variances shrink and standard errors come out too small. Adding a random residual to each prediction (stochastic regression imputation) fixes the noise part.

Even then, one filled-in dataset treats the guesses as if they were real measurements. The uncertainty about *what the missing values were* is gone.

## Multiple imputation

The fix is to impute several times, say $M = 5$ to 20, each time drawing different plausible values (with different random residuals and, ideally, model parameters drawn from their uncertainty). Run the same analysis on each completed dataset, then combine with **Rubin's rules**. If $\hat Q_m$ is the estimate from dataset $m$ and $U_m$ its squared standard error,

$$
\bar Q = \frac{1}{M}\sum_{m} \hat Q_m, \qquad \bar U = \frac{1}{M}\sum_m U_m, \qquad B = \frac{1}{M-1}\sum_m (\hat Q_m - \bar Q)^2,
$$

$$
T = \bar U + \Big(1 + \frac{1}{M}\Big) B .
$$

$\bar Q$ is the final estimate. $\bar U$ is the ordinary within-dataset uncertainty; $B$ is the between-imputation spread, the extra uncertainty from not knowing the missing values. The total variance $T$ includes both, so the standard errors are honest. In R, the `mice` package does the imputing and pooling.

Multiple imputation still assumes MAR. It fixes the "too confident" problem of single imputation, not the bias of MNAR.

## Key takeaways

- MCAR: missingness is pure chance. MAR: it depends only on observed data. MNAR: it depends on the missing values or on unrecorded variables.
- Standard imputation needs MAR; MNAR is the hardest case, not the easiest.
- You can't test MAR vs MNAR from the data; it's a judgement about how the data came to be.
- Single imputation understates uncertainty; multiple imputation with Rubin's rules adds the between-imputation variance back.

## Resources

- 732A93 Statistical Methods, Linköping University (computer assignment, autumn 2019)
- Andrew Gelman and Jennifer Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*, Chapter 25: http://www.stat.columbia.edu/~gelman/arm/missing.pdf
- Donald B. Rubin, "Inference and missing data", *Biometrika* 63(3) (1976): https://doi.org/10.1093/biomet/63.3.581
- Stef van Buuren, *Flexible Imputation of Missing Data*, 2nd ed. (free online; Chapter 2 on Rubin's rules): https://stefvanbuuren.name/fimd/
