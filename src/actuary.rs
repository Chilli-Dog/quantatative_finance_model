pub fn decide_purchase(pred_prices: Vec<f64>) -> bool {
    let risk_factor = 1.0;

    let mut total = 0.0;
    for price in pred_prices.clone() {
        total += price;
    }
    let mean = total / pred_prices.clone().len() as f64;
    if mean > risk_factor {
        return true;
    }
    false
}