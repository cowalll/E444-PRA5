import requests
import csv
import time
import re
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def send_api_calls_and_record(url, num_calls, payloads, output_filename="api_responses.csv"):
    """
    Sends API calls to a specified URL multiple times and records the response
    and timestamp into a CSV file.
    Each call uses a payload from the provided list, cycling through them.

    Args:
        url (str): The URL of the API endpoint.
        num_calls (int): The number of API calls to make.
        output_filename (str): The name of the CSV file to save the results.
    """
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['request_timestamp', 'response_timestamp', 'latency', 'payload', 'prediction', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_calls):
            error = None
            prediction = None
            request_timestamp = None
            response_timestamp = None
            latency = None

            current_payload = payloads[i % len(payloads)]

            try:
                request_timestamp = datetime.datetime.now().isoformat()
                response = requests.post(url, data=current_payload)
                response_timestamp = datetime.datetime.now().isoformat()
                status_code = response.status_code
                response_body = response.text
                if response.ok:
                    match = re.search(r'<p>Prediction: (REAL|FAKE)</p>', response_body)
                    if match:
                        prediction = match.group(1)
                    else:
                        error = "Prediction not found in response"
                else:
                    error = f"HTTP Status {status_code}"

                # Calculate latency from timestamps for consistency
                if request_timestamp and response_timestamp:
                    request_time = datetime.datetime.fromisoformat(request_timestamp)
                    response_time = datetime.datetime.fromisoformat(response_timestamp)
                    latency = (response_time - request_time).total_seconds()

            except requests.exceptions.RequestException as e:
                error = str(e)
            except Exception as e:
                error = f"An unexpected error occurred: {e}"

            writer.writerow({
                'request_timestamp': request_timestamp,
                'response_timestamp': response_timestamp,
                'latency': latency,
                'payload': current_payload,
                'prediction': prediction,
                'error': error
            })
            print(f"Call {i+1}/{num_calls} - Prediction: {prediction if prediction else 'Error'}")
            time.sleep(0.1)  # Small delay to avoid overwhelming the server

    print(f"API call results saved to {output_filename}")

def generate_latency_boxplot(csv_filename="api_responses.csv", output_image_filename="latency_boxplot.png"):
    """Reads the CSV file and generates a box plot of latencies for each payload."""
    df = pd.read_csv(csv_filename)
    df['payload_message'] = df['payload'].apply(lambda x: eval(x)['message'])

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='latency', y='payload_message', data=df)
    plt.title('API Latency by Payload')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('Payload Message')
    plt.tight_layout()
    plt.savefig(output_image_filename)
    print(f"Latency box plot saved to {output_image_filename}")
    # plt.show() # Uncomment to display the plot directly

if __name__ == "__main__":
    api_url = "http://ece444-pra5-elastic-beanstalk-env.eba-4m3pvais.us-east-2.elasticbeanstalk.com/predict-form"
    payloads = [
        {"message": "president found on moon"},         # fake
        {"message": "price of cheese falls"},           # fake
        {"message": "trump puts tariffs on canada"},    # real
        {"message": "the king is alive"}                # real
    ]
    number_of_calls = len(payloads) * 100

    csv_file = "api_responses.csv"
    send_api_calls_and_record(api_url, number_of_calls, payloads, output_filename=csv_file)

    generate_latency_boxplot(csv_filename=csv_file)