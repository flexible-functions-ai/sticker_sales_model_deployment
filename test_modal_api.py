import requests
import pandas as pd
import io

# Update URL to use the Feast endpoint
url = "https://flexible-functions-ai--sticker-sales-api-predict-csv-feast.modal.run"

print("🧪 Testing Feast-enabled Sticker Sales API")
print("=" * 50)

# Create a sample CSV with test data
test_data = pd.DataFrame([
    {
        "date": "2023-01-15",
        "country": "US",
        "store": "Store_001",
        "product": "Sticker_A"
    },
    {
        "date": "2023-01-15",
        "country": "Canada",
        "store": "Discount Stickers",
        "product": "Holographic Goose"
    },
    {
        "date": "2023-01-16",
        "country": "UK",
        "store": "Sticker World",
        "product": "Kaggle"
    },
    {
        "date": "2023-02-01",
        "country": "Germany",
        "store": "European Stickers",
        "product": "Data Science"
    },
    {
        "date": "2023-02-14",
        "country": "Japan",
        "store": "Tokyo Stickers",
        "product": "Machine Learning"
    }
])

print(f"📊 Test data shape: {test_data.shape}")
print(f"📝 Test data columns: {list(test_data.columns)}")
print("\n🔍 Sample test data:")
print(test_data.head())

# Save the test data to a CSV file in memory
csv_buffer = io.StringIO()
test_data.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode()

# Prepare the file for upload
files = {'file': ('test_data.csv', csv_bytes, 'text/csv')}

# Make the prediction request
print(f"\n🚀 Sending request to Feast API...")
print(f"🌐 URL: {url}")

try:
    response = requests.post(url, files=files, timeout=60)
    
    # Print the result
    print(f"\n📡 Response status code: {response.status_code}")
    
    # Try to parse the JSON response
    try:
        result = response.json()
        
        if isinstance(result, dict):
            if result.get('success', True):
                # Successful prediction
                predictions = result.get('predictions', result)
                model_info = result.get('model_info', {})
                
                print("✅ Prediction successful!")
                
                if model_info:
                    print(f"🤖 Model info:")
                    for key, value in model_info.items():
                        print(f"   {key}: {value}")
                
                if isinstance(predictions, list):
                    # Create a DataFrame with predictions
                    result_df = test_data.copy()
                    result_df['predicted_sales'] = predictions
                    
                    print(f"\n📈 Prediction results:")
                    print("=" * 80)
                    print(result_df.to_string(index=False))
                    
                    print(f"\n📊 Prediction statistics:")
                    print(f"   Count: {len(predictions)}")
                    print(f"   Min: {min(predictions):.2f}")
                    print(f"   Max: {max(predictions):.2f}")
                    print(f"   Mean: {sum(predictions)/len(predictions):.2f}")
                else:
                    print(f"📄 Raw predictions: {predictions}")
            else:
                # Error response
                print("❌ Prediction failed!")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                if 'traceback' in result:
                    print(f"   Traceback: {result['traceback'][:500]}...")
        else:
            # Direct prediction list
            predictions = result
            result_df = test_data.copy()
            result_df['predicted_sales'] = predictions
            
            print("✅ Prediction successful!")
            print(f"\n📈 Prediction results:")
            print("=" * 80)
            print(result_df.to_string(index=False))
    
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        print(f"📄 Response text (first 500 chars): {response.text[:500]}")

except requests.exceptions.RequestException as e:
    print(f"❌ Network error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n🏁 Test completed!")