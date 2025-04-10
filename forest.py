import kagglehub

# Download latest version
path = kagglehub.dataset_download("devvratmathur/micro-expression-dataset-for-lie-detection")

print("Path to dataset files:", path)
