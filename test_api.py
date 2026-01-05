import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

BASE_URL = "http://localhost:8000"

class FraudDetectionTester:
    """Test suite for Fraud Detection API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results = []
    
    def test_health(self):
        """Test health endpoint"""
        print("\n" + "="*80)
        print("TEST 1: Health Check")
        print("="*80)
        
        response = requests.get(f"{self.base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    
    def test_stats(self):
        """Test stats endpoint"""
        print("\n" + "="*80)
        print("TEST 2: System Statistics")
        print("="*80)
        
        response = requests.get(f"{self.base_url}/stats")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    
    def predict_transaction(self, tx: Dict, test_name: str = "") -> Dict:
        """Make a prediction for a single transaction"""
        response = requests.post(
            f"{self.base_url}/predict",
            json=tx
        )
        
        if response.status_code == 200:
            result = response.json()
            self.results.append({
                "test_name": test_name,
                **tx,
                **result
            })
            return result
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {}
    
    def test_legitimate_transaction(self):
        """Test a normal, legitimate transaction"""
        print("\n" + "="*80)
        print("Legitimate Transaction (Small Amount, Daytime)")
        print("="*80)
        
        tx = {
            "transaction_id": "test_legit_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 25.50,
            "timestamp": "2024-01-15T14:30:00"
        }
        
        print(f"Input: {json.dumps(tx, indent=2)}")
        result = self.predict_transaction(tx, "Legitimate - Small Daytime")
        
        if result:
            print(f"\nFraud Probability: {result['fraud_probability']:.4f}")
            print(f"Decision: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    
    def test_suspicious_high_amount(self):
        """Test high amount transaction"""
        print("\n" + "="*80)
        print("TEST 4: Suspicious Transaction (High Amount)")
        print("="*80)
        
        tx = {
            "transaction_id": "test_suspicious_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 500.00,
            "timestamp": "2024-01-15T14:30:00"
        }
        
        print(f"Input: {json.dumps(tx, indent=2)}")
        result = self.predict_transaction(tx, "Suspicious - High Amount")
        
        if result:
            print(f"\nFraud Probability: {result['fraud_probability']:.4f}")
            print(f"Decision: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
    
    def test_night_transaction(self):
        """Test transaction at night with high amount"""
        print("\n" + "="*80)
        print("TEST 5: Night Transaction (High Amount at 2 AM)")
        print("="*80)
        
        tx = {
            "transaction_id": "test_night_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 350.00,
            "timestamp": "2024-01-15T02:30:00"
        }
        
        print(f"Input: {json.dumps(tx, indent=2)}")
        result = self.predict_transaction(tx, "Night - High Amount")
        
        if result:
            print(f"\n🌙 Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"🌙 Decision: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'}")
            print(f"🌙 Risk Level: {result['risk_level']}")
            print(f"🌙 Processing Time: {result['processing_time_ms']:.2f}ms")
    
    def test_multiple_customers(self):
        """Test different customers to see variation"""
        print("\n" + "="*80)
        print("TEST 6: Multiple Customers (Same Transaction)")
        print("="*80)
        
        base_tx = {
            "amount": 75.00,
            "timestamp": "2024-01-15T14:30:00",
            "terminal_id": 1000
        }
        
        customer_ids = [100, 500, 1000, 2000, 3000]
        
        for cust_id in customer_ids:
            tx = {
                "transaction_id": f"test_customer_{cust_id}",
                "customer_id": cust_id,
                **base_tx
            }
            
            result = self.predict_transaction(tx, f"Customer {cust_id}")
            
            if result:
                print(f"\nCustomer {cust_id}: Fraud Prob = {result['fraud_probability']:.4f} | "
                      f"Decision = {'FRAUD' if result['is_fraud'] else 'LEGIT'} | "
                      f"Avg Spend = ${result['features_used']['customer_avg_amount_7d']:.2f}")
    
    def test_performance_consistency(self):
        """Test API performance over multiple calls"""
        print("\n" + "="*80)
        print("TEST 7: Performance Consistency (10 identical requests)")
        print("="*80)
        
        tx = {
            "transaction_id": "perf_test",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 50.00,
            "timestamp": "2024-01-15T12:00:00"
        }
        
        times = []
        
        for i in range(10):
            start = time.time()
            result = self.predict_transaction(tx, f"Performance Test {i+1}")
            end = time.time()
            
            if result:
                processing_time = result['processing_time_ms']
                times.append(processing_time)
                print(f"Request {i+1}: {processing_time:.2f}ms")
        
        if times:
            print(f"\nPerformance Summary:")
            print(f"   First request: {times[0]:.2f}ms (cold start)")
            print(f"   Average (excluding first): {sum(times[1:])/len(times[1:]):.2f}ms")
            print(f"   Min: {min(times):.2f}ms")
            print(f"   Max: {max(times):.2f}ms")
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\n" + "="*80)
        print("TEST 8: Edge Cases")
        print("="*80)
        
        test_cases = [
            {
                "name": "Very Small Amount",
                "tx": {
                    "transaction_id": "edge_tiny",
                    "customer_id": 100,
                    "terminal_id": 500,
                    "amount": 0.50,
                    "timestamp": "2024-01-15T14:30:00"
                }
            },
            {
                "name": "Very Large Amount",
                "tx": {
                    "transaction_id": "edge_huge",
                    "customer_id": 100,
                    "terminal_id": 500,
                    "amount": 5000.00,
                    "timestamp": "2024-01-15T14:30:00"
                }
            },
            {
                "name": "Weekend Transaction",
                "tx": {
                    "transaction_id": "edge_weekend",
                    "customer_id": 100,
                    "terminal_id": 500,
                    "amount": 100.00,
                    "timestamp": "2024-01-13T14:30:00"  # Saturday
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            result = self.predict_transaction(test_case['tx'], test_case['name'])
            
            if result:
                print(f"Fraud Probability: {result['fraud_probability']:.4f}")
                print(f"Decision: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE ✅'}")
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        print("\n" + "="*80)
        print("TEST 9: Batch Prediction (5 transactions at once)")
        print("="*80)
        
        transactions = [
            {
                "transaction_id": f"batch_{i}",
                "customer_id": 100 + i * 100,
                "terminal_id": 500 + i * 50,
                "amount": 50.0 + i * 20,
                "timestamp": "2024-01-15T14:30:00"
            }
            for i in range(5)
        ]
        
        print(f"Sending {len(transactions)} transactions...")
        response = requests.post(
            f"{self.base_url}/batch-predict",
            json=transactions
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nBatch processed: {result['count']} transactions")
            
            for i, pred in enumerate(result['predictions']):
                print(f"\nTransaction {i+1}:")
                print(f"  Amount: ${pred['amount']}")
                print(f"  Fraud Prob: {pred['fraud_probability']:.4f}")
                print(f"  Decision: {'FRAUD' if pred['is_fraud'] else 'LEGIT'}")
        else:
            print(f"Error: {response.status_code}")
    
    def generate_summary_report(self):
        """Generate summary report of all tests"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        if not self.results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\nTotal Predictions: {len(df)}")
        print(f"Flagged as Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
        print(f"Legitimate: {(~df['is_fraud']).sum()} ({(~df['is_fraud']).mean()*100:.1f}%)")
        
        print(f"\nFraud Probability Stats:")
        print(f"  Mean: {df['fraud_probability'].mean():.4f}")
        print(f"  Min: {df['fraud_probability'].min():.4f}")
        print(f"  Max: {df['fraud_probability'].max():.4f}")
        
        print(f"\nProcessing Time Stats:")
        print(f"  Mean: {df['processing_time_ms'].mean():.2f}ms")
        print(f"  Min: {df['processing_time_ms'].min():.2f}ms")
        print(f"  Max: {df['processing_time_ms'].max():.2f}ms")
        
        print(f"\nRisk Level Distribution:")
        print(df['risk_level'].value_counts().to_string())
        
        # Save to CSV
        df.to_csv('test_results.csv', index=False)
        print(f"\nResults saved to test_results.csv")


def main():
    """Run all tests"""
    print("FRAUD DETECTION API - COMPREHENSIVE TEST SUITE")
    
    tester = FraudDetectionTester()
    
    try:
        # Run all tests
        tester.test_health()
        tester.test_stats()
        tester.test_legitimate_transaction()
        tester.test_suspicious_high_amount()
        tester.test_night_transaction()
        tester.test_multiple_customers()
        tester.test_performance_consistency()
        tester.test_edge_cases()
        tester.test_batch_prediction()
        
        # Generate summary
        tester.generate_summary_report()
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"\nError during testing: {e}")


if __name__ == "__main__":
    main()