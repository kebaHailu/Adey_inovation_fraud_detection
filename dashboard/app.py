from flask import Flask, jsonify, request
import pandas as pd
from logger import SetupLogger


app = Flask(__name__)

# Load fraud data from CSV
def load_fraud_data():
    return pd.read_csv('../data/merged_fraud_data.csv')

@app.route('/fraud_summary', methods=['GET'])
def fraud_summary():
    """
    Serve summary statistics for total transactions, fraud cases, and fraud percentage.
    """
    data = load_fraud_data()
    total_transactions = data['transaction_count'].sum()
    total_fraud_cases = data['fraud_cases'].sum()
    fraud_percentage = (total_fraud_cases / total_transactions) * 100

    summary = {
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    }
    return jsonify(summary)

@app.route('/fraud_trends', methods=['GET'])
def fraud_trends():
    """
    Serve fraud trends data over time.
    """
    data = load_fraud_data()
    trend_data = data.groupby('date').agg({
        'transaction_count': 'sum',
        'fraud_cases': 'sum'
    }).reset_index()

    return jsonify(trend_data.to_dict(orient='records'))

@app.route('/fraud_by_location', methods=['GET'])
def fraud_by_location():
    """
    Serve fraud data grouped by geographic location.
    """
    data = load_fraud_data()
    location_data = data.groupby('location').agg({
        'transaction_count': 'sum',
        'fraud_cases': 'sum'
    }).reset_index()

    return jsonify(location_data.to_dict(orient='records'))

@app.route('/fraud_by_device_browser', methods=['GET'])
def fraud_by_device_browser():
    """
    Serve fraud cases data grouped by device and browser.
    """
    data = load_fraud_data()
    device_browser_data = data.groupby(['device', 'browser']).agg({
        'fraud_cases': 'sum'
    }).reset_index()

    return jsonify(device_browser_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)