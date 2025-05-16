# Scoreformer Benchmark Tutorial

This tutorial guides you through running the benchmarking tool to compare Scoreformer with other recommendation algorithms.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (optional)

### Step 1: Set up the Environment

1. Clone or download this repository:
```bash
git clone https://github.com/yourusername/scoreformer-benchmark.git
cd scoreformer-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Benchmark

You can run the benchmark using one of two methods:

#### Method 1: Using the shell script

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

#### Method 2: Running Python script directly

```bash
python benchmark.py
```

The benchmark will:
- Download the MovieLens-1M dataset automatically
- Train and evaluate all models
- Generate comparison metrics and visualizations

### Step 3: View Results

After the benchmark completes, you can find the results in the `benchmark_results` directory:

```bash
ls -la benchmark_results/results/
```

This will show you:
- CSV files containing performance metrics
- JSON files with detailed results
- PNG files with performance visualization plots

### Step 4: Interpret the Results

Open the CSV file to see a performance comparison table similar to the one below:

```
Method      HR@10    NDCG@10    Precision@10    Recall@10
BPR         0.685    0.410      0.147           0.285
NCF         0.701    0.425      0.152           0.292
NGCF        0.723    0.448      0.157           0.307
LightGCN    0.738    0.459      0.160           0.318
PinSage     0.731    0.451      0.158           0.312
SASRec      0.745    0.465      0.162           0.324
BERT4Rec    0.749    0.472      0.164           0.328
GTN         0.753    0.478      0.166           0.332
DualGNN     0.757    0.483      0.168           0.337
Scoreformer 0.775    0.502      0.174           0.348
```

The metrics used are:
- **HR@10**: Hit Ratio at 10 - The percentage of test cases where the ground truth item appears in the top-10 recommendations
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10 - Measures the ranking quality, giving higher value to hits at higher positions
- **Precision@10**: The ratio of relevant items to the total number of items recommended
- **Recall@10**: The ratio of relevant items retrieved to the total number of relevant items

## Customizing the Benchmark

### Changing Hyperparameters

To modify the training process, edit `benchmark.py` and look for the `train_and_evaluate` function:

```python
def train_and_evaluate(model_class, model_name, dataset, epochs=5, batch_size=256, lr=0.001):
    # ...
```

You can change:
- `epochs`: The number of training epochs
- `batch_size`: The batch size for training
- `lr`: The learning rate

### Adding New Models

To add a new model to the benchmark:

1. Create your model class in a separate file or add it to `benchmark.py`
2. Add your model to the `models` dictionary:

```python
models = {
    'BPR': BPR,
    'NCF': NCF,
    # ...
    'YourModel': YourModelClass,
}
```

### Running a Subset of Models

To run only specific models, modify the `run_benchmark` function:

```python
# Run benchmark for specific models only
selected_models = ['BPR', 'Scoreformer']
for model_name in selected_models:
    if model_name in models and models[model_name] is not None:
        metrics = train_and_evaluate(models[model_name], model_name, dataset)
        results[model_name] = metrics
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: If you encounter memory issues, try reducing the batch size or using a smaller dataset.

2. **CUDA Out of Memory**: Try moving to CPU mode by modifying the device selection:
   ```python
   device = torch.device('cpu')
   ```

3. **Package Not Found**: Make sure you've installed all requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. **Dataset Download Failure**: If automatic download fails, manually download the MovieLens-1M dataset from https://grouplens.org/datasets/movielens/1m/ and place it in the `benchmark_results/data` directory. 