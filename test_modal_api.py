import requests
import pandas as pd
import io

# The URL of your CSV prediction endpoint
url = "https://flexible-functions-ai--sticker-sales-api-predict-csv.modal.run"

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
    }
])

# Save the test data to a CSV file in memory
csv_buffer = io.StringIO()
test_data.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode()

# Prepare the file for upload
files = {'file': ('test_data.csv', csv_bytes, 'text/csv')}

# Make the prediction request
print(f"Sending request to {url}...")
response = requests.post(url, files=files)

# Print the result
print("Status code:", response.status_code)

# Try to parse the JSON response
try:
    prediction = response.json()
    print("Prediction:", prediction)
    
    # If the prediction is a list as expected
    if isinstance(prediction, list):
        # Create a DataFrame with predictions
        result_df = test_data.copy()
        result_df['predicted_sales'] = prediction
        print("\nPrediction results:")
        print(result_df)
except Exception as e:
    print("Error parsing response:", e)
    print("Response text:", response.text[:500])