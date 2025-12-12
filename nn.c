/*
 * Neural Network Implementation in C
 * Supports CSV input with multiple features and binary/multiclass classification
 * 
 * Features:
 * - Multi-layer perceptron with configurable hidden layers
 * - Forward propagation
 * - Backpropagation with gradient descent
 * - ReLU and Sigmoid activation functions
 * - Softmax for multiclass classification
 * - CSV input with header row
 * - Training and prediction capabilities
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LINE_LENGTH 10000
#define MAX_FEATURES 1000
#define MAX_CLASSES 100
#define MAX_LAYERS 10

typedef struct {
    double **data;          // Feature matrix (rows x features)
    int *labels;            // Class labels
    int rows;               // Number of samples
    int features;           // Number of features
    int num_classes;        // Number of unique classes
    char **feature_names;   // Feature names from CSV header
} Dataset;

typedef struct {
    int num_layers;         // Total number of layers (including input and output)
    int *layer_sizes;       // Size of each layer
    double ***weights;      // Weights for each layer [layer][neuron][prev_neuron]
    double **biases;        // Biases for each layer [layer][neuron]
    double **activations;   // Activations for each layer (temporary)
    double **z_values;      // Pre-activation values (temporary)
    double learning_rate;   // Learning rate
    int max_epochs;         // Maximum training epochs
    int batch_size;         // Mini-batch size
} NeuralNetwork;

// Function prototypes
Dataset* load_csv(const char *filename, int is_training);
void free_dataset(Dataset *dataset);
NeuralNetwork* create_network(int *layer_sizes, int num_layers, double learning_rate, int max_epochs, int batch_size);
void free_network(NeuralNetwork *nn);
void train_network(NeuralNetwork *nn, Dataset *train_data);
int* predict_network(NeuralNetwork *nn, Dataset *test_data);
void forward_propagation(NeuralNetwork *nn, double *input, int input_size);
void backward_propagation(NeuralNetwork *nn, double *input, int true_label, int num_classes);
double relu(double x);
double relu_derivative(double x);
double sigmoid(double x);
double sigmoid_derivative(double x);
void softmax(double *input, int length, double *output);
double compute_accuracy(int *predictions, int *actual, int n);
void print_confusion_matrix(int *predictions, int *actual, int n, int num_classes);
void shuffle_data(Dataset *dataset, int *indices);

// Utility functions
int count_features(const char *line);
void parse_line(const char *line, double *features, int *label, int num_features, int is_training);
int get_num_classes(int *labels, int n);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <train_csv> <test_csv> [output_csv] [hidden_layers] [learning_rate] [epochs] [batch_size]\n", argv[0]);
        printf("\nCSV Format:\n");
        printf("  - First row: feature1,feature2,...,featureN,class\n");
        printf("  - Data rows: value1,value2,...,valueN,class_label\n");
        printf("  - Class labels should be integers (0, 1, 2, ...)\n");
        printf("\nParameters:\n");
        printf("  output_csv: Output file for predictions (default: output.csv)\n");
        printf("  hidden_layers: Comma-separated sizes (e.g., '64,32' for 2 hidden layers)\n");
        printf("  learning_rate: Learning rate (default: 0.01)\n");
        printf("  epochs: Number of training epochs (default: 100)\n");
        printf("  batch_size: Mini-batch size (default: 32)\n");
        printf("\nExample:\n");
        printf("  %s train.csv test.csv output.csv 64,32 0.01 100 32\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    const char *train_file = argv[1];
    const char *test_file = argv[2];
    const char *output_file = "output.csv";
    int arg_offset = 0;
    
    // Check if 3rd argument is output file or hidden layers
    if (argc > 3) {
        const char *arg3 = argv[3];
        // If it contains .csv or .txt, it's an output file
        if (strstr(arg3, ".csv") != NULL || strstr(arg3, ".txt") != NULL || strstr(arg3, ".out") != NULL) {
            output_file = arg3;
            arg_offset = 1;
        }
    }
    
    // Parse hidden layer sizes
    int hidden_layers[MAX_LAYERS];
    int num_hidden = 0;
    
    if (argc > 3 + arg_offset && strlen(argv[3 + arg_offset]) > 0) {
        char *hidden_str = strdup(argv[3 + arg_offset]);
        char *token = strtok(hidden_str, ",");
        while (token != NULL && num_hidden < MAX_LAYERS) {
            hidden_layers[num_hidden++] = atoi(token);
            token = strtok(NULL, ",");
        }
        free(hidden_str);
    } else {
        // Default: one hidden layer with 64 neurons
        hidden_layers[0] = 64;
        num_hidden = 1;
    }
    
    double learning_rate = (argc > 4 + arg_offset) ? atof(argv[4 + arg_offset]) : 0.01;
    int max_epochs = (argc > 5 + arg_offset) ? atoi(argv[5 + arg_offset]) : 100;
    int batch_size = (argc > 6 + arg_offset) ? atoi(argv[6 + arg_offset]) : 32;

    printf("=== Neural Network Classifier ===\n\n");
    printf("Configuration:\n");
    printf("  Training file: %s\n", train_file);
    printf("  Test file: %s\n", test_file);
    printf("  Output file: %s\n", output_file);
    printf("  Hidden layers: ");
    for (int i = 0; i < num_hidden; i++) {
        printf("%d%s", hidden_layers[i], i < num_hidden - 1 ? ", " : "\n");
    }
    printf("  Learning rate: %.4f\n", learning_rate);
    printf("  Epochs: %d\n", max_epochs);
    printf("  Batch size: %d\n\n", batch_size);

    // Load training data
    printf("Loading training data...\n");
    Dataset *train_data = load_csv(train_file, 1);
    if (!train_data) {
        fprintf(stderr, "Error: Failed to load training data\n");
        return 1;
    }
    printf("  Samples: %d\n", train_data->rows);
    printf("  Features: %d\n", train_data->features);
    printf("  Classes: %d\n", train_data->num_classes);
    printf("  Feature names: ");
    for (int i = 0; i < train_data->features; i++) {
        printf("%s%s", train_data->feature_names[i], i < train_data->features - 1 ? ", " : "\n");
    }
    printf("\n");

    // Build layer sizes array
    int num_layers = num_hidden + 2; // input + hidden + output
    int layer_sizes = (int)malloc(num_layers * sizeof(int));
    layer_sizes[0] = train_data->features;
    for (int i = 0; i < num_hidden; i++) {
        layer_sizes[i + 1] = hidden_layers[i];
    }
    layer_sizes[num_layers - 1] = train_data->num_classes;

    printf("Network Architecture:\n");
    printf("  Input layer: %d neurons\n", layer_sizes[0]);
    for (int i = 1; i < num_layers - 1; i++) {
        printf("  Hidden layer %d: %d neurons\n", i, layer_sizes[i]);
    }
    printf("  Output layer: %d neurons\n\n", layer_sizes[num_layers - 1]);

    // Create and train network
    printf("Training network...\n");
    NeuralNetwork *nn = create_network(layer_sizes, num_layers, learning_rate, max_epochs, batch_size);
    train_network(nn, train_data);
    printf("Training complete!\n\n");

    // Load test data
    printf("Loading test data...\n");
    Dataset *test_data = load_csv(test_file, 1);
    if (!test_data) {
        fprintf(stderr, "Error: Failed to load test data\n");
        free_dataset(train_data);
        free_network(nn);
        free(layer_sizes);
        return 1;
    }
    printf("  Test samples: %d\n\n", test_data->rows);

    // Make predictions
    printf("Making predictions...\n");
    int *predictions = predict_network(nn, test_data);
    
    // Calculate and display accuracy
    double accuracy = compute_accuracy(predictions, test_data->labels, test_data->rows);
    printf("\nTest Accuracy: %.2f%%\n\n", accuracy * 100);

    // Show confusion matrix
    printf("Confusion Matrix:\n");
    print_confusion_matrix(predictions, test_data->labels, test_data->rows, test_data->num_classes);

    // Show sample predictions
    printf("\nSample Predictions (first 10):\n");
    printf("%-6s ", "Index");
    for (int j = 0; j < test_data->features; j++) {
        printf("%-12s ", test_data->feature_names[j]);
    }
    printf("%-10s %-10s\n", "Predicted", "Actual");
    for (int j = 0; j < 6 + test_data->features * 13 + 21; j++) {
        printf("-");
    }
    printf("\n");
    
    int display_count = (test_data->rows < 10) ? test_data->rows : 10;
    for (int i = 0; i < display_count; i++) {
        printf("%-6d ", i);
        for (int j = 0; j < test_data->features; j++) {
            printf("%-12.4f ", test_data->data[i][j]);
        }
        printf("%-10d %-10d %s\n", predictions[i], test_data->labels[i],
               predictions[i] == test_data->labels[i] ? "✓" : "✗");
    }
    
    // Write predictions to file
    FILE *out_fp = fopen(output_file, "w");
    if (out_fp) {
        // Write header
        fprintf(out_fp, "index");
        for (int j = 0; j < test_data->features; j++) {
            fprintf(out_fp, ",%s", test_data->feature_names[j]);
        }
        fprintf(out_fp, ",predicted\n");
        
        // Write data
        for (int i = 0; i < test_data->rows; i++) {
            fprintf(out_fp, "%d", i);
            for (int j = 0; j < test_data->features; j++) {
                fprintf(out_fp, ",%.6f", test_data->data[i][j]);
            }
            fprintf(out_fp, ",%d\n", predictions[i]);
        }
        fclose(out_fp);
        printf("\nPredictions written to: %s\n", output_file);
    } else {
        fprintf(stderr, "Warning: Could not write to output file %s\n", output_file);
    }

    // Cleanup
    free(predictions);
    free_dataset(train_data);
    free_dataset(test_data);
    free_network(nn);
    free(layer_sizes);

    printf("\nDone!\n");
    return 0;
}

// Load CSV file and parse into Dataset
Dataset* load_csv(const char *filename, int is_training) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    Dataset dataset = (Dataset)malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(fp);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    
    // Read header line
    if (!fgets(line, MAX_LINE_LENGTH, fp)) {
        fprintf(stderr, "Error: Empty file\n");
        free(dataset);
        fclose(fp);
        return NULL;
    }

    line[strcspn(line, "\n")] = 0;
    int total_columns = count_features(line);
    dataset->features = total_columns - 1;
    
    // Parse feature names
    dataset->feature_names = (char*)malloc(dataset->features * sizeof(char));
    char *token = strtok(line, ",");
    for (int i = 0; i < dataset->features && token != NULL; i++) {
        dataset->feature_names[i] = strdup(token);
        token = strtok(NULL, ",");
    }
    
    // Count data rows
    long header_pos = ftell(fp);
    int num_rows = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp)) {
        if (strlen(line) > 1) num_rows++;
    }
    dataset->rows = num_rows;
    
    // Allocate memory
    dataset->data = (double*)malloc(num_rows * sizeof(double));
    dataset->labels = (int*)malloc(num_rows * sizeof(int));
    
    for (int i = 0; i < num_rows; i++) {
        dataset->data[i] = (double*)malloc(dataset->features * sizeof(double));
    }
    
    // Read data
    fseek(fp, header_pos, SEEK_SET);
    int row = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) && row < num_rows) {
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) > 1) {
            parse_line(line, dataset->data[row], &dataset->labels[row], dataset->features, is_training);
            row++;
        }
    }
    
    fclose(fp);
    dataset->num_classes = get_num_classes(dataset->labels, dataset->rows);
    
    return dataset;
}

void parse_line(const char *line, double *features, int *label, int num_features, int is_training) {
    char *line_copy = strdup(line);
    char *token = strtok(line_copy, ",");
    
    for (int i = 0; i < num_features && token != NULL; i++) {
        features[i] = atof(token);
        token = strtok(NULL, ",");
    }
    
    if (is_training && token != NULL) {
        *label = atoi(token);
    }
    
    free(line_copy);
}

int count_features(const char *line) {
    int count = 1;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') count++;
    }
    return count;
}

int get_num_classes(int *labels, int n) {
    int max_label = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    return max_label + 1;
}

// Create neural network
NeuralNetwork* create_network(int *layer_sizes, int num_layers, double learning_rate, int max_epochs, int batch_size) {
    NeuralNetwork nn = (NeuralNetwork)malloc(sizeof(NeuralNetwork));
    
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));
    nn->learning_rate = learning_rate;
    nn->max_epochs = max_epochs;
    nn->batch_size = batch_size;
    
    // Allocate weights and biases
    nn->weights = (double**)malloc((num_layers - 1) * sizeof(double*));
    nn->biases = (double*)malloc((num_layers - 1) * sizeof(double));
    nn->activations = (double*)malloc(num_layers * sizeof(double));
    nn->z_values = (double*)malloc(num_layers * sizeof(double));
    
    for (int l = 0; l < num_layers; l++) {
        nn->activations[l] = (double*)malloc(layer_sizes[l] * sizeof(double));
        nn->z_values[l] = (double*)malloc(layer_sizes[l] * sizeof(double));
    }
    
    // Initialize weights and biases with Xavier/He initialization
    for (int l = 0; l < num_layers - 1; l++) {
        int curr_size = layer_sizes[l + 1];
        int prev_size = layer_sizes[l];
        
        nn->weights[l] = (double*)malloc(curr_size * sizeof(double));
        nn->biases[l] = (double*)calloc(curr_size, sizeof(double));
        
        double scale = sqrt(2.0 / prev_size); // He initialization for ReLU
        
        for (int i = 0; i < curr_size; i++) {
            nn->weights[l][i] = (double*)malloc(prev_size * sizeof(double));
            for (int j = 0; j < prev_size; j++) {
                nn->weights[l][i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
            }
        }
    }
    
    return nn;
}

// ReLU activation
double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

// Sigmoid activation
double sigmoid(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Softmax activation
void softmax(double *input, int length, double *output) {
    double max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Forward propagation
void forward_propagation(NeuralNetwork *nn, double *input, int input_size) {
    // Copy input to first layer activations
    memcpy(nn->activations[0], input, input_size * sizeof(double));
    
    // Forward through each layer
    for (int l = 0; l < nn->num_layers - 1; l++) {
        int curr_size = nn->layer_sizes[l + 1];
        int prev_size = nn->layer_sizes[l];
        
        for (int i = 0; i < curr_size; i++) {
            nn->z_values[l + 1][i] = nn->biases[l][i];
            for (int j = 0; j < prev_size; j++) {
                nn->z_values[l + 1][i] += nn->weights[l][i][j] * nn->activations[l][j];
            }
            
            // Apply activation function
            if (l < nn->num_layers - 2) {
                // Hidden layers: ReLU
                nn->activations[l + 1][i] = relu(nn->z_values[l + 1][i]);
            } else {
                // Output layer: store z values, apply softmax later
                nn->activations[l + 1][i] = nn->z_values[l + 1][i];
            }
        }
    }
    
    // Apply softmax to output layer
    int output_layer = nn->num_layers - 1;
    softmax(nn->activations[output_layer], nn->layer_sizes[output_layer], nn->activations[output_layer]);
}

// Backward propagation
void backward_propagation(NeuralNetwork *nn, double *input, int true_label, int num_classes) {
    double *deltas = (double)malloc(nn->num_layers * sizeof(double));
    for (int l = 0; l < nn->num_layers; l++) {
        deltas[l] = (double*)calloc(nn->layer_sizes[l], sizeof(double));
    }
    
    // Compute output layer delta
    int output_layer = nn->num_layers - 1;
    for (int i = 0; i < nn->layer_sizes[output_layer]; i++) {
        deltas[output_layer][i] = nn->activations[output_layer][i] - (i == true_label ? 1.0 : 0.0);
    }
    
    // Backpropagate through hidden layers
    for (int l = nn->num_layers - 2; l > 0; l--) {
        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            double sum = 0.0;
            for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
                sum += nn->weights[l][j][i] * deltas[l + 1][j];
            }
            deltas[l][i] = sum * relu_derivative(nn->z_values[l][i]);
        }
    }
    
    // Update weights and biases
    for (int l = 0; l < nn->num_layers - 1; l++) {
        for (int i = 0; i < nn->layer_sizes[l + 1]; i++) {
            nn->biases[l][i] -= nn->learning_rate * deltas[l + 1][i];
            for (int j = 0; j < nn->layer_sizes[l]; j++) {
                nn->weights[l][i][j] -= nn->learning_rate * deltas[l + 1][i] * nn->activations[l][j];
            }
        }
    }
    
    // Cleanup
    for (int l = 0; l < nn->num_layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

// Shuffle training data
void shuffle_data(Dataset *dataset, int *indices) {
    for (int i = dataset->rows - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Train the network
void train_network(NeuralNetwork *nn, Dataset *train_data) {
    int n = train_data->rows;
    int indices = (int)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;
    
    for (int epoch = 0; epoch < nn->max_epochs; epoch++) {
        shuffle_data(train_data, indices);
        
        double total_loss = 0.0;
        int correct = 0;
        
        // Process mini-batches
        for (int batch_start = 0; batch_start < n; batch_start += nn->batch_size) {
            int batch_end = (batch_start + nn->batch_size < n) ? batch_start + nn->batch_size : n;
            
            for (int i = batch_start; i < batch_end; i++) {
                int idx = indices[i];
                
                // Forward propagation
                forward_propagation(nn, train_data->data[idx], train_data->features);
                
                // Compute loss
                int output_layer = nn->num_layers - 1;
                total_loss -= log(nn->activations[output_layer][train_data->labels[idx]] + 1e-10);
                
                // Check prediction
                int predicted = 0;
                double max_prob = nn->activations[output_layer][0];
                for (int c = 1; c < train_data->num_classes; c++) {
                    if (nn->activations[output_layer][c] > max_prob) {
                        max_prob = nn->activations[output_layer][c];
                        predicted = c;
                    }
                }
                if (predicted == train_data->labels[idx]) correct++;
                
                // Backward propagation
                backward_propagation(nn, train_data->data[idx], train_data->labels[idx], train_data->num_classes);
            }
        }
        
        // Print progress
        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            printf("  Epoch %3d/%d - Loss: %.4f - Accuracy: %.2f%%\n", 
                   epoch + 1, nn->max_epochs, total_loss / n, (double)correct / n * 100);
        }
    }
    
    free(indices);
}

// Predict classes for test data
int* predict_network(NeuralNetwork *nn, Dataset *test_data) {
    int n = test_data->rows;
    int predictions = (int)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        forward_propagation(nn, test_data->data[i], test_data->features);
        
        int output_layer = nn->num_layers - 1;
        int predicted = 0;
        double max_prob = nn->activations[output_layer][0];
        
        for (int c = 1; c < test_data->num_classes; c++) {
            if (nn->activations[output_layer][c] > max_prob) {
                max_prob = nn->activations[output_layer][c];
                predicted = c;
            }
        }
        
        predictions[i] = predicted;
    }
    
    return predictions;
}

// Compute accuracy
double compute_accuracy(int *predictions, int *actual, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predictions[i] == actual[i]) correct++;
    }
    return (double)correct / n;
}

// Print confusion matrix
void print_confusion_matrix(int *predictions, int *actual, int n, int num_classes) {
    int *matrix = (int)calloc(num_classes, sizeof(int));
    for (int i = 0; i < num_classes; i++) {
        matrix[i] = (int*)calloc(num_classes, sizeof(int));
    }
    
    for (int i = 0; i < n; i++) {
        matrix[actual[i]][predictions[i]]++;
    }
    
    printf("       ");
    for (int i = 0; i < num_classes; i++) {
        printf("Pred%-3d ", i);
    }
    printf("\n");
    
    for (int i = 0; i < num_classes; i++) {
        printf("Act%-3d ", i);
        for (int j = 0; j < num_classes; j++) {
            printf("%-7d ", matrix[i][j]);
        }
        printf("\n");
    }
    
    for (int i = 0; i < num_classes; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Free dataset memory
void free_dataset(Dataset *dataset) {
    if (!dataset) return;
    
    for (int i = 0; i < dataset->rows; i++) {
        free(dataset->data[i]);
    }
    free(dataset->data);
    free(dataset->labels);
    
    for (int i = 0; i < dataset->features; i++) {
        free(dataset->feature_names[i]);
    }
    free(dataset->feature_names);
    free(dataset);
}

// Free network memory
void free_network(NeuralNetwork *nn) {
    if (!nn) return;
    
    for (int l = 0; l < nn->num_layers - 1; l++) {
        for (int i = 0; i < nn->layer_sizes[l + 1]; i++) {
            free(nn->weights[l][i]);
        }
        free(nn->weights[l]);
        free(nn->biases[l]);
    }
    
    for (int l = 0; l < nn->num_layers; l++) {
        free(nn->activations[l]);
        free(nn->z_values[l]);
    }
    
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
    free(nn->layer_sizes);
    free(nn);
}
