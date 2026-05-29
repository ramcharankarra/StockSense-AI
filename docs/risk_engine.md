# StockSense AI — Quantitative Risk Analytics Specification

The **Risk Analytics** engine (`analysis/risk_analytics.py`) implements standard mathematical formulas to calculate asset volatility, benchmark relationships, and potential downside losses.

---

## 📐 Mathematical Formulations

### 1. Daily Returns ($R_t$)
Daily returns are calculated using the percentage change of adjacent close prices:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### 2. Annualized Return ($\bar{R}_{ann}$)
Annualizes the geometric mean of daily returns over a standard trading year (252 sessions):
$$\bar{R}_{ann} = \left( \prod_{t=1}^{N} (1 + R_t) \right)^{\frac{252}{N}} - 1$$

### 3. Historical Volatility ($\sigma_{ann}$)
The annualized standard deviation of daily returns:
$$\sigma_{ann} = \sqrt{\frac{1}{N-1} \sum_{t=1}^{N} (R_t - \bar{R})^2} \times \sqrt{252}$$

---

## 📊 Performance & Reward-To-Risk Ratios

### 4. Sharpe Ratio ($S$)
Measures the excess return per unit of total risk (volatility). A risk-free rate ($R_f$) of 5.0% is applied as a default:
$$S = \frac{\bar{R}_{ann} - R_f}{\sigma_{ann}}$$

### 5. Sortino Ratio ($T$)
Similar to Sharpe, but only penalizes negative volatility (downside risk), which represents actual loss. It ignores positive volatility (upside deviations):
$$T = \frac{\bar{R}_{ann} - R_f}{\sigma_d}$$

Where downside deviation ($\sigma_d$) is:
$$\sigma_d = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \min(0, R_t)^2} \times \sqrt{252}$$

---

## 📉 Downside Protection & Loss Estimations

### 6. Maximum Drawdown ($MDD$)
Calculates the largest peak-to-trough drop in portfolio equity value over a specific lookback timeline:
$$DD_t = \frac{H_t - P_t}{H_t}$$
$$MDD = \max_{t} (DD_t)$$

Where $H_t$ represents the rolling historical maximum close price:
$$H_t = \max_{\tau \le t} (P_\tau)$$

### 7. Value-at-Risk ($VaR_{95\%}$)
Quantifies the maximum expected loss at a 95% confidence level over a 1-day horizon. StockSense AI computes this using the **historical simulation approach**:
$$VaR_{95\%} = \text{Percentile}\left(\{R_1, R_2, \dots, R_N\}, 5\right)$$
*(Meaning there is a 5% probability that the asset will lose more than this percentage in a single trading session).*

### 8. Conditional Value-at-Risk ($CVaR_{95\%}$)
Also known as **Expected Shortfall (ES)**, CVaR represents the average loss when the VaR threshold is exceeded:
$$CVaR_{95\%} = \mathbb{E}\left[ R_t \;\middle|\; R_t \le VaR_{95\%} \right]$$
$$CVaR_{95\%} = \frac{1}{|R_t \le VaR_{95\%}|} \sum_{R_t \le VaR_{95\%}} R_t$$

---

## ⚖️ Benchmark Sensitivity (vs. S&P 500)

### 9. Beta ($\beta$)
Measures the asset's systematic risk relative to the market benchmark:
$$\beta = \frac{\text{Cov}(R_{asset}, R_{bench})}{\text{Var}(R_{bench})}$$

### 10. Alpha ($\alpha$)
Measures the asset's idiosyncratic excess return relative to the benchmark return adjusted for risk (Capital Asset Pricing Model):
$$\alpha = \bar{R}_{asset, ann} - \left( R_f + \beta \times (\bar{R}_{bench, ann} - R_f) \right)$$
*(A positive Alpha indicates that the quantitative strategy or stock outperformed the market on a risk-adjusted basis).*
