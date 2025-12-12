/*
 * Logistic Regression Implementation in C with Text Label Support
 * Supports CSV input with text or numeric class labels
 * 
 * Features:
 * - Reads CSV files with header row containing feature names
 * - Supports text class labels (automatically mapped to 0,1,2,...)
 * - Supports multiple features (unlimited)
 * - Binary and multiclass classification
 * - Training with gradient descent
 * - Prediction on test data with label mapping
 * - Softmax for multiclass classification
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <float.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 10000
#define MAX_FEATURES 1000
#define MAX_CLASSES 100
#define MAX_LABEL_LENGTH 100

// Label mapping structure
typedef struct {
    char **text_labels;     // Array of unique text labels
    int num_labels;         // Number of unique labels
} LabelMap;

typedef struct {
    double **data;          // Feature matrix (rows x features)
    int *labels;            // Class labels (numeric 0,1,2,...)
    int rows;               // Number of samples
    int features;           // Number of features
    int num_classes;        // Number of unique classes
    char **feature_names;   // Feature names from CSV header
    int has_labels;         // Whether dataset contains labels
    LabelMap *label_map;    // Mapping from text to numeric labels
} Dataset;

typedef struct {
    double **weights;       // Weights matrix (num_classes x features)
    double *bias;           // Bias terms for each class
    int features;           // Number of features
    int num_classes;        // Number of classes
    double learning_rate;   // Learning rate
    int max_iterations;     // Maximum iterations for training
} LogisticModel;

// Function prototypes
Dataset* load_csv(const char *filename, int is_training, LabelMap *existing_map);
void free_dataset(Dataset *dataset);
LogisticModel* create_model(int features, int num_classes, double learning_rate, int max_iterations);
void free_model(LogisticModel *model);
void train_model(LogisticModel *model, Dataset *train_data);
int* predict(LogisticModel *model, Dataset *test_data);
double sigmoid(double x);
void softmax(double *input, int length, double *output);
double compute_accuracy(int *predictions, int *actual, int n);
void print_confusion_matrix(int *predictions, int *actual, int n, int num_classes, LabelMap *label_map);
int* load_actual_labels(const char *filename, int *num_labels);

// Label mapping functions
LabelMap* create_label_map();
void free_label_map(LabelMap *map);
int get_or_add_label(LabelMap *map, const char *text_label);
int get_label_index(LabelMap *map, const char *text_label);
const char* get_label_text(LabelMap *map, int index);
int is_numeric_label(const char *label);

// Utility functions
int count_lines(FILE *fp);
int count_features(const char *line);
void parse_line(const char *line, double *features, char *label_text, int num_features, int is_training);
int get_num_classes(int *labels, int n);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <train_csv> <test_csv> [output_file] [learning_rate] [iterations]\n", argv[0]);
        printf("\nCSV Format:\n");
        printf("  - First row: feature1,feature2,...,featureN,class\n");
        printf("  - Data rows: value1,value2,...,valueN,class_label\n");
        printf("  - Class labels can be text (e.g., 'cat', 'dog') or numbers (0, 1, 2, ...)\n");
        printf("  - Text labels are automatically mapped to numbers: first unique → 0, second → 1, etc.\n");
        printf("\nArguments:\n");
        printf("  output_file: Optional file to write predictions (default: output.csv)\n");
        printf("  learning_rate: Learning rate (default: 0.01)\n");
        printf("  iterations: Max iterations (default: 1000)\n");
        printf("\nExample:\n");
        printf("  %s train.csv test.csv predictions.csv 0.01 1000\n", argv[0]);
        return 1;
    }

    const char *train_file = argv[1];
    const char *test_file = argv[2];
    const char *output_file = (argc > 3) ? argv[3] : "output.csv";
    double learning_rate = (argc > 4) ? atof(argv[4]) : 0.01;
    int max_iterations = (argc > 5) ? atoi(argv[5]) : 1000;

    printf("=== Logistic Regression Classifier (Text Label Support) ===\n\n");
    printf("Configuration:\n");
    printf("  Training file: %s\n", train_file);
    printf("  Test file: %s\n", test_file);
    printf("  Output file: %s\n", output_file);
    printf("  Learning rate: %.4f\n", learning_rate);
    printf("  Max iterations: %d\n\n", max_iterations);

    // Load training data
    printf("Loading training data...\n");
    Dataset *train_data = load_csv(train_file, 1, NULL);
    if (!train_data) {
        fprintf(stderr, "Error: Failed to load training data\n");
        return 1;
    }
    printf("  Samples: %d\n", train_data->rows);
    printf("  Features: %d\n", train_data->features);
    printf("  Classes: %d\n", train_data->num_classes);
    
    // Display label mapping if text labels were used
    if (train_data->label_map && train_data->label_map->num_labels > 0) {
        printf("  Label mapping:\n");
        for (int i = 0; i < train_data->label_map->num_labels; i++) {
            printf("    '%s' → %d\n", train_data->label_map->text_labels[i], i);
        }
    }
    
    printf("  Feature names: ");
    for (int i = 0; i < train_data->features; i++) {
        printf("%s%s", train_data->feature_names[i], i < train_data->features - 1 ? ", " : "\n");
    }
    printf("\n");

    // Create and train model
    printf("Training model...\n");
    LogisticModel *model = create_model(train_data->features, train_data->num_classes, 
                                       learning_rate, max_iterations);
    train_model(model, train_data);
    printf("Training complete!\n\n");

    // Load test data (pass the label map from training data)
    printf("Loading test data...\n");
    Dataset *test_data = load_csv(test_file, 0, train_data->label_map);
    if (!test_data) {
        fprintf(stderr, "Error: Failed to load test data\n");
        free_dataset(train_data);
        free_model(model);
        return 1;
    }
    printf("  Test samples: %d\n", test_data->rows);
    printf("  Has labels: %s\n\n", test_data->has_labels ? "Yes" : "No");

    // Make predictions
    printf("Making predictions...\n");
    int *predictions = predict(model, test_data);
    
    // Calculate and display accuracy if labels exist
    if (test_data->has_labels) {
        double accuracy = compute_accuracy(predictions, test_data->labels, test_data->rows);
        printf("\nTest Accuracy: %.2f%%\n\n", accuracy * 100);

        // Show confusion matrix
        printf("Confusion Matrix:\n");
        print_confusion_matrix(predictions, test_data->labels, test_data->rows, 
                             test_data->num_classes, train_data->label_map);
    }

    // Show sample predictions
    printf("\nSample Predictions (first 10):\n");
    printf("%-6s ", "Index");
    for (int j = 0; j < test_data->features; j++) {
        printf("%-12s ", test_data->feature_names[j]);
    }
    printf("%-15s", "Predicted");
    if (test_data->has_labels) {
        printf(" %-15s", "Actual");
    }
    printf("\n");
    
    int display_count = (test_data->rows < 10) ? test_data->rows : 10;
    for (int i = 0; i < display_count; i++) {
        printf("%-6d ", i);
        for (int j = 0; j < test_data->features; j++) {
            printf("%-12.4f ", test_data->data[i][j]);
        }
        
        // Show label text if available
        const char *pred_text = get_label_text(train_data->label_map, predictions[i]);
        if (pred_text) {
            printf("%-15s", pred_text);
        } else {
            printf("%-15d", predictions[i]);
        }
        
        if (test_data->has_labels) {
            const char *actual_text = get_label_text(train_data->label_map, test_data->labels[i]);
            if (actual_text) {
                printf(" %-15s %s", actual_text,
                       predictions[i] == test_data->labels[i] ? "✓" : "✗");
            } else {
                printf(" %-15d %s", test_data->labels[i],
                       predictions[i] == test_data->labels[i] ? "✓" : "✗");
            }
        }
        printf("\n");
    }
    
    // Write predictions to file
    FILE *out_fp = fopen(output_file, "w");
    if (out_fp) {
        // Write header
        fprintf(out_fp, "index");
        for (int j = 0; j < test_data->features; j++) {
            fprintf(out_fp, ",%s", test_data->feature_names[j]);
        }
        fprintf(out_fp, ",predicted");
        if (test_data->has_labels) {
            fprintf(out_fp, ",actual,correct");
        }
        fprintf(out_fp, "\n");
        
        // Write data
        for (int i = 0; i < test_data->rows; i++) {
            fprintf(out_fp, "%d", i);
            for (int j = 0; j < test_data->features; j++) {
                fprintf(out_fp, ",%.6f", test_data->data[i][j]);
            }
            
            // Write predicted label (use text if available)
            const char *pred_text = get_label_text(train_data->label_map, predictions[i]);
            if (pred_text) {
                fprintf(out_fp, ",%s", pred_text);
            } else {
                fprintf(out_fp, ",%d", predictions[i]);
            }
            
            if (test_data->has_labels) {
                const char *actual_text = get_label_text(train_data->label_map, test_data->labels[i]);
                if (actual_text) {
                    fprintf(out_fp, ",%s", actual_text);
                } else {
                    fprintf(out_fp, ",%d", test_data->labels[i]);
                }
                fprintf(out_fp, ",%s", predictions[i] == test_data->labels[i] ? "yes" : "no");
            }
            fprintf(out_fp, "\n");
        }
        
        fclose(out_fp);
        printf("\nPredictions written to: %s\n", output_file);
    } else {
        fprintf(stderr, "Warning: Could not open output file %s\n", output_file);
    }

    // Cleanup
    free(predictions);
    
    // Free test data (but not its label_map, it belongs to train_data)
    if (test_data) {
        if (test_data->data) {
            for (int i = 0; i < test_data->rows; i++) {
                if (test_data->data[i]) free(test_data->data[i]);
            }
            free(test_data->data);
        }
        if (test_data->labels) free(test_data->labels);
        if (test_data->feature_names) {
            for (int i = 0; i < test_data->features; i++) {
                if (test_data->feature_names[i]) free(test_data->feature_names[i]);
            }
            free(test_data->feature_names);
        }
        free(test_data);
    }
    
    // Free training data (including label_map)
    if (train_data) {
        if (train_data->data) {
            for (int i = 0; i < train_data->rows; i++) {
                if (train_data->data[i]) free(train_data->data[i]);
            }
            free(train_data->data);
        }
        if (train_data->labels) free(train_data->labels);
        if (train_data->feature_names) {
            for (int i = 0; i < train_data->features; i++) {
                if (train_data->feature_names[i]) free(train_data->feature_names[i]);
            }
            free(train_data->feature_names);
        }
        if (train_data->label_map) free_label_map(train_data->label_map);
        free(train_data);
    }
    
    free_model(model);

    return 0;
}

// Label mapping functions

LabelMap* create_label_map() {
    LabelMap map = (LabelMap)malloc(sizeof(LabelMap));
    if (!map) return NULL;
    
    map->text_labels = (char*)malloc(MAX_CLASSES * sizeof(char));
    if (!map->text_labels) {
        free(map);
        return NULL;
    }
    
    for (int i = 0; i < MAX_CLASSES; i++) {
        map->text_labels[i] = NULL;
    }
    
    map->num_labels = 0;
    return map;
}

void free_label_map(LabelMap *map) {
    if (!map) return;
    
    if (map->text_labels) {
        for (int i = 0; i < map->num_labels; i++) {
            if (map->text_labels[i]) {
                free(map->text_labels[i]);
            }
        }
        free(map->text_labels);
    }
    free(map);
}

// Check if a label string is purely numeric
int is_numeric_label(const char *label) {
    if (!label || strlen(label) == 0) return 0;
    
    int i = 0;
    // Skip leading whitespace
    while (isspace(label[i])) i++;
    
    // Check for sign
    if (label[i] == '-' || label[i] == '+') i++;
    
    // Must have at least one digit
    if (!isdigit(label[i])) return 0;
    
    // Check remaining characters
    while (label[i] != '\0' && !isspace(label[i])) {
        if (!isdigit(label[i]) && label[i] != '.') return 0;
        i++;
    }
    
    return 1;
}

// Get or add a text label to the mapping (returns numeric index)
int get_or_add_label(LabelMap *map, const char *text_label) {
    if (!map || !text_label) return 0;
    
    // Trim the label
    char trimmed[MAX_LABEL_LENGTH];
    int start = 0, end = strlen(text_label) - 1;
    while (start <= end && isspace(text_label[start])) start++;
    while (end >= start && isspace(text_label[end])) end--;
    
    int len = end - start + 1;
    if (len >= MAX_LABEL_LENGTH) len = MAX_LABEL_LENGTH - 1;
    strncpy(trimmed, text_label + start, len);
    trimmed[len] = '\0';
    
    // Check if label already exists
    for (int i = 0; i < map->num_labels; i++) {
        if (map->text_labels[i] && strcmp(map->text_labels[i], trimmed) == 0) {
            return i;
        }
    }
    
    // Add new label
    if (map->num_labels < MAX_CLASSES) {
        map->text_labels[map->num_labels] = strdup(trimmed);
        return map->num_labels++;
    }
    
    return 0;  // Fallback
}

// Get label index (returns -1 if not found)
int get_label_index(LabelMap *map, const char *text_label) {
    if (!map || !text_label) return -1;
    
    char trimmed[MAX_LABEL_LENGTH];
    int start = 0, end = strlen(text_label) - 1;
    while (start <= end && isspace(text_label[start])) start++;
    while (end >= start && isspace(text_label[end])) end--;
    
    int len = end - start + 1;
    if (len >= MAX_LABEL_LENGTH) len = MAX_LABEL_LENGTH - 1;
    strncpy(trimmed, text_label + start, len);
    trimmed[len] = '\0';
    
    for (int i = 0; i < map->num_labels; i++) {
        if (map->text_labels[i] && strcmp(map->text_labels[i], trimmed) == 0) {
            return i;
        }
    }
    
    return -1;
}

// Get text label from numeric index
const char* get_label_text(LabelMap *map, int index) {
    if (!map || index < 0 || index >= map->num_labels) {
        return NULL;
    }
    return map->text_labels[index];
}

// Load CSV with label mapping support
Dataset* load_csv(const char *filename, int is_training, LabelMap *existing_map) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    
    // Read header
    if (!fgets(line, MAX_LINE_LENGTH, fp)) {
        fprintf(stderr, "Error: Empty CSV file\n");
        fclose(fp);
        return NULL;
    }

    Dataset dataset = (Dataset)malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(fp);
        return NULL;
    }
    
    dataset->has_labels = is_training;  // Training data should have labels
    dataset->label_map = NULL;
    
    // Initialize or reuse label map
    if (is_training) {
        dataset->label_map = create_label_map();
    } else if (existing_map) {
        dataset->label_map = existing_map;  // Use training data's label map
    }
    
    line[strcspn(line, "\n")] = 0;
    line[strcspn(line, "\r")] = 0;
    
    int num_cols = count_features(line);
    
    // Determine if last column is a label
    if (is_training) {
        dataset->features = num_cols - 1;  // Assume last column is label
        dataset->has_labels = 1;
    } else {
        // For test data, check if it looks like it has labels
        dataset->features = num_cols - 1;  // Try assuming it has labels
        dataset->has_labels = 1;  // We'll validate this when reading data
    }
    
    if (dataset->features <= 0) {
        fprintf(stderr, "Error: Invalid number of features\n");
        free(dataset);
        fclose(fp);
        return NULL;
    }
    
    // Parse feature names
    dataset->feature_names = (char*)malloc(dataset->features * sizeof(char));
    if (!dataset->feature_names) {
        free(dataset);
        fclose(fp);
        return NULL;
    }
    
    char *line_copy = strdup(line);
    if (!line_copy) {
        free(dataset->feature_names);
        free(dataset);
        fclose(fp);
        return NULL;
    }
    
    char *token = strtok(line_copy, ",");
    for (int i = 0; i < dataset->features && token != NULL; i++) {
        dataset->feature_names[i] = strdup(token);
        token = strtok(NULL, ",");
    }
    free(line_copy);
    
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
        line[strcspn(line, "\r")] = 0;
        if (strlen(line) > 1) {
            char label_text[MAX_LABEL_LENGTH];
            parse_line(line, dataset->data[row], label_text, dataset->features, dataset->has_labels);
            
            if (dataset->has_labels && strlen(label_text) > 0) {
                // Check if numeric or text
                if (is_numeric_label(label_text)) {
                    // Numeric label - use directly
                    dataset->labels[row] = atoi(label_text);
                } else {
                    // Text label - map to numeric
                    if (is_training) {
                        dataset->labels[row] = get_or_add_label(dataset->label_map, label_text);
                    } else {
                        // For test data, look up in existing map
                        int idx = get_label_index(dataset->label_map, label_text);
                        dataset->labels[row] = (idx >= 0) ? idx : 0;
                    }
                }
            } else {
                dataset->labels[row] = 0;
            }
            
            row++;
        }
    }
    
    fclose(fp);
    
    // Determine number of classes
    if (dataset->has_labels) {
        if (dataset->label_map && dataset->label_map->num_labels > 0) {
            dataset->num_classes = dataset->label_map->num_labels;
        } else {
            dataset->num_classes = get_num_classes(dataset->labels, dataset->rows);
        }
    } else {
        dataset->num_classes = 0;
    }
    
    return dataset;
}

// Parse a CSV line into features and label text
void parse_line(const char *line, double *features, char *label_text, int num_features, int is_training) {
    if (!line || !features || strlen(line) == 0) {
        if (label_text) label_text[0] = '\0';
        return;
    }
    
    char *line_copy = strdup(line);
    if (!line_copy) {
        if (label_text) label_text[0] = '\0';
        return;
    }
    
    char *token = strtok(line_copy, ",");
    int feature_count = 0;
    
    while (token != NULL && feature_count < num_features) {
        while (*token == ' ' || *token == '\t') token++;
        features[feature_count] = atof(token);
        feature_count++;
        token = strtok(NULL, ",");
    }
    
    // Get label text if training/testing and token exists
    if (is_training && label_text != NULL) {
        if (token != NULL) {
            while (*token == ' ' || *token == '\t') token++;
            strncpy(label_text, token, MAX_LABEL_LENGTH - 1);
            label_text[MAX_LABEL_LENGTH - 1] = '\0';
        } else {
            label_text[0] = '\0';
        }
    }
    
    free(line_copy);
}

// Free dataset
void free_dataset(Dataset *dataset) {
    if (!dataset) return;
    
    if (dataset->data) {
        for (int i = 0; i < dataset->rows; i++) {
            if (dataset->data[i]) {
                free(dataset->data[i]);
            }
        }
        free(dataset->data);
    }
    
    if (dataset->labels) {
        free(dataset->labels);
    }
    
    if (dataset->feature_names) {
        for (int i = 0; i < dataset->features; i++) {
            if (dataset->feature_names[i]) {
                free(dataset->feature_names[i]);
            }
        }
        free(dataset->feature_names);
    }
    
    // Only free label_map if we own it (was created for this dataset)
    // Don't free if it was passed from another dataset
    free(dataset);
}

// Count features
int count_features(const char *line) {
    int count = 1;
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') count++;
    }
    return count;
}

// Get number of unique classes
int get_num_classes(int *labels, int n) {
    int max_label = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] > max_label) {
            max_label = labels[i];
        }
    }
    return max_label + 1;
}

// Create model
LogisticModel* create_model(int features, int num_classes, double learning_rate, int max_iterations) {
    LogisticModel model = (LogisticModel)malloc(sizeof(LogisticModel));
    if (!model) return NULL;
    
    model->features = features;
    model->num_classes = num_classes;
    model->learning_rate = learning_rate;
    model->max_iterations = max_iterations;
    
    model->weights = (double*)malloc(num_classes * sizeof(double));
    model->bias = (double*)calloc(num_classes, sizeof(double));
    
    for (int i = 0; i < num_classes; i++) {
        model->weights[i] = (double*)calloc(features, sizeof(double));
    }
    
    return model;
}

// Free model
void free_model(LogisticModel *model) {
    if (!model) return;
    
    if (model->weights) {
        for (int i = 0; i < model->num_classes; i++) {
            if (model->weights[i]) {
                free(model->weights[i]);
            }
        }
        free(model->weights);
    }
    
    if (model->bias) {
        free(model->bias);
    }
    
    free(model);
}

// Sigmoid function
double sigmoid(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

// Softmax function
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

// Train model
void train_model(LogisticModel *model, Dataset *train_data) {
    int n = train_data->rows;
    int features = model->features;
    int num_classes = model->num_classes;
    double lr = model->learning_rate;
    
    double scores = (double)malloc(num_classes * sizeof(double));
    double probs = (double)malloc(num_classes * sizeof(double));
    
    for (int iter = 0; iter < model->max_iterations; iter++) {
        double total_loss = 0.0;
        
        for (int i = 0; i < n; i++) {
            // Compute scores for each class
            for (int c = 0; c < num_classes; c++) {
                scores[c] = model->bias[c];
                for (int f = 0; f < features; f++) {
                    scores[c] += model->weights[c][f] * train_data->data[i][f];
                }
            }
            
            // Apply softmax
            softmax(scores, num_classes, probs);
            
            // Compute loss
            int true_class = train_data->labels[i];
            total_loss -= log(probs[true_class] + 1e-10);
            
            // Update weights and bias
            for (int c = 0; c < num_classes; c++) {
                double error = (c == true_class) ? (probs[c] - 1.0) : probs[c];
                model->bias[c] -= lr * error;
                for (int f = 0; f < features; f++) {
                    model->weights[c][f] -= lr * error * train_data->data[i][f];
                }
            }
        }
        
        if (iter % 100 == 0) {
            printf("  Iteration %d, Loss: %.4f\n", iter, total_loss / n);
        }
    }
    
    free(scores);
    free(probs);
}

// Predict
int* predict(LogisticModel *model, Dataset *test_data) {
    int predictions = (int)malloc(test_data->rows * sizeof(int));
    double scores = (double)malloc(model->num_classes * sizeof(double));
    
    for (int i = 0; i < test_data->rows; i++) {
        for (int c = 0; c < model->num_classes; c++) {
            scores[c] = model->bias[c];
            for (int f = 0; f < model->features; f++) {
                scores[c] += model->weights[c][f] * test_data->data[i][f];
            }
        }
        
        // Find class with max score
        int max_class = 0;
        double max_score = scores[0];
        for (int c = 1; c < model->num_classes; c++) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                max_class = c;
            }
        }
        
        predictions[i] = max_class;
    }
    
    free(scores);
    return predictions;
}

// Compute accuracy
double compute_accuracy(int *predictions, int *actual, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (predictions[i] == actual[i]) {
            correct++;
        }
    }
    return (double)correct / n;
}

// Print confusion matrix with label names
void print_confusion_matrix(int *predictions, int *actual, int n, int num_classes, LabelMap *label_map) {
    int *matrix = (int)calloc(num_classes, sizeof(int));
    for (int i = 0; i < num_classes; i++) {
        matrix[i] = (int*)calloc(num_classes, sizeof(int));
    }
    
    for (int i = 0; i < n; i++) {
        matrix[actual[i]][predictions[i]]++;
    }
    
    printf("       ");
    for (int i = 0; i < num_classes; i++) {
        const char *label = get_label_text(label_map, i);
        if (label) {
            printf("%-8s ", label);
        } else {
            printf("Class%-3d ", i);
        }
    }
    printf("\n");
    
    for (int i = 0; i < num_classes; i++) {
        const char *label = get_label_text(label_map, i);
        if (label) {
            printf("%-7s", label);
        } else {
            printf("Class%-2d", i);
        }
        for (int j = 0; j < num_classes; j++) {
            printf("%-9d", matrix[i][j]);
        }
        printf("\n");
    }
    
    for (int i = 0; i < num_classes; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
