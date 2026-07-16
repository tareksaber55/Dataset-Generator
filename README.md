# Dataset Generation

Generate high-quality synthetic datasets using open-source Large Language Models (LLMs) from Hugging Face and OpenAI Compatible Models through an easy-to-use Gradio interface.

---

## Overview

Dataset Generation Studio is a Python application that enables users to generate structured synthetic datasets without manually writing prompts or interacting directly with language models.

The application allows users to:

* Generate synthetic datasets for any domain.
* Choose from multiple models.
* Define custom dataset schemas.
* Export generated datasets as CSV files.
* Use either local Hugging Face models or OpenAI-compatible APIs.

---

## Features

* **Model Selection**

  * Supports multiple Hugging Face models.
  * Compatible with OpenAI API endpoints.

* **Custom Dataset Generation**

  * Define dataset purpose.
  * Specify column names.
  * Provide sample records.
  * Choose the number of rows to generate.

* **Prompt Engineering**

  * Automatically builds optimized prompts from user inputs.
  * Ensures generated data follows the requested schema.

* **CSV Export**

  * Download generated datasets instantly.

* **Gradio Interface**

  * Simple and intuitive web UI.
  * No coding experience required.

---

## Project Structure

```text
Dataset-Generation-Studio/
│
├── app.py                 # Gradio application
├── generator.py           # functions that generate text,prompts
├── models.py              # functions that load models
└── README.md
```

## Usage

### 1. Select a Model

Choose the model you want to use.

Examples:

* Llama
* Qwen
* Mistral
* Gemma
* DeepSeek
* Any OpenAI-compatible endpoint

---

### 2. Configure the Model

Provide:

* API Key (if required)
* Base URL
* Model Name

---

### 3. Describe the Dataset

Enter:

* Dataset purpose
* Column names
* Example rows
* Number of samples

Example

Purpose

```
Taxi Trip Duration Prediction
```

Columns

```python
[
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_duration"
]
```

Example Data

```python
[
    {
        "passenger_count": 1,
        "pickup_longitude": -73.97634125,
        "pickup_latitude": 40.76363373,
        "dropoff_longitude": -73.97334290,
        "dropoff_latitude": 40.74341965,
        "trip_duration": 677
    }
]
```

Sample Size

```
1000
```

---

### 4. Generate Dataset

Click **Generate**.

The application will:

1. Build a prompt.
2. Send it to the selected model.
3. Parse the JSON output.
4. Convert it into a pandas DataFrame.
5. Display the generated dataset.

---

### 5. Export

Click **Download CSV** to save the generated dataset.

---

## Supported Models

Any model capable of following structured instructions can be used.

Examples include:

* Llama 3
* Qwen 2.5
* Mistral
* Gemma
* DeepSeek
* Phi
* Other Hugging Face instruction-tuned models

---

## Technologies Used

* Python
* Gradio
* Hugging Face Transformers
* Pandas
* PyTorch
* OpenAI Python SDK



## License

This project is released under the MIT License.

---

## Contributing

Contributions are welcome.

Feel free to open an issue or submit a pull request to improve the project.

