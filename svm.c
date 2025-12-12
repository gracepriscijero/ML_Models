/*
 * Support Vector Machine (SVM) Implementation in C
 * Supports CSV input with multiple features and binary/multiclass classification
 * 
 * Features:
 * - Reads CSV files with header row containing feature names
 * - Supports multiple features (unlimited)
 * - Binary and multiclass classification (One-vs-All approach)
 * - Training with Sequential Minimal Optimization (SMO) algorithm
 * - Multiple kernel functions: Linear, Polynomial, RBF (Gaussian)
 * - Prediction on test data
 * - Cross-validation support
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_LINE_LENGTH 10000
#define MAX_FEATURES 1000
#define MAX_CLASSES 100
#define EPSILON 1e-3
#define TOL 1e-3

// Kernel types
typedef enum {
    LINEAR,
    POLYNOMIAL,
    RBF
} KernelType;

typedef struct {
    double **data;          // Feature matrix (rows x features)
    int *labels;            // Class labels
    int rows;               // Number of samples
    int features;           // Number of features
    int num_classes;        // Number of unique classes
    char **feature_names;   // Feature names from CSV header
} Dataset;

typedef struct {
    double *alpha;          // Lagrange multipliers
    double *w;              // Weight vector (for linear kernel)
    double b;               // Bias term
    double **support_vectors; // Support vectors
    int *sv_labels;         // Support vector labels
    int num_sv;             // Number of support vectors
    int features;           // Number of features
    double *errors;         // Error cache
    KernelType kernel_type; // Kernel function type
    double C;               // Regularization parameter
    double gamma;           // RBF kernel parameter
    int degree;             // Polynomial kernel degree
    double coef0;           // Polynomial kernel coefficient
} SVMBinary;

typedef struct {
    SVMBinary **binary_svms; // One binary SVM per class
    int num_classes;        // Number of classes
    int features;           // Number of features
    KernelType kernel_type; // Kernel function type
    double C;               // Regularization parameter
    double gamma;           // RBF kernel parameter
    int degree;             // Polynomial kernel degree
    double coef0;           // Polynomial kernel coefficient
    int max_iterations;     // Maximum iterations for training
} SVMModel;

// Function prototypes
Dataset* load_csv(const char *filename, int is_training);
void free_dataset(Dataset *dataset);
SVMModel* create_svm_model(int features, int num_classes, KernelType kernel, 
                          double C, double gamma, int degree, double coef0, int max_iter);
void free_svm_model(SVMModel *model);
void train_svm_model(SVMModel *model, Dataset *train_data);
int* predict_svm(SVMModel *model, Dataset *test_data);
double compute_accuracy(int *predictions, int *actual, int n);
void print_confusion_matrix(int *predictions, int *actual, int n, int num_classes);

// Binary SVM functions
SVMBinary* create_binary_svm(int features, KernelType kernel, double C, 
                             double gamma, int degree, double coef0);
void free_binary_svm(SVMBinary *svm);
void train_binary_svm(SVMBinary *svm, double **data, int *labels, int n, int max_iter);
double predict_binary_svm(SVMBinary *svm, double *x);
double kernel_function(SVMBinary *svm, double *x1, double *x2, int features);
int take_step(SVMBinary *svm, int i1, int i2, double **data, int *labels, int n);
int examine_example(SVMBinary *svm, int i2, double **data, int *labels, int n);

// Utility functions
int count_features(const char *line);
void parse_line(const char *line, double *features, int *label, int num_features, int is_training);
int get_num_classes(int *labels, int n);
double dot_product(double *v1, double *v2, int n);
const char* kernel_name(KernelType type);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <train_csv> <test_csv> [output_csv] [C] [kernel] [gamma] [degree] [coef0] [max_iter]\n", argv[0]);
        printf("\nCSV Format:\n");
        printf("  - First row: feature1,feature2,...,featureN,class\n");
        printf("  - Data rows: value1,value2,...,valueN,class_label\n");
        printf("  - Class labels should be integers (0, 1, 2, ...)\n");
        printf("\nParameters:\n");
        printf("  output_csv: Output file for predictions (default: output.csv)\n");
        printf("  C         : Regularization parameter (default: 1.0)\n");
        printf("  kernel    : Kernel type: linear, poly, rbf (default: rbf)\n");
        printf("  gamma     : RBF/Poly kernel parameter (default: auto = 1/n_features)\n");
        printf("  degree    : Polynomial kernel degree (default: 3)\n");
        printf("  coef0     : Polynomial kernel coefficient (default: 0.0)\n");
        printf("  max_iter  : Maximum iterations (default: 1000)\n");
        printf("\nExamples:\n");
        printf("  %s train.csv test.csv\n", argv[0]);
        printf("  %s train.csv test.csv output.csv 1.0 rbf 0.1\n", argv[0]);
        printf("  %s train.csv test.csv predictions.csv 1.0 linear\n", argv[0]);
        printf("  %s train.csv test.csv results.csv 1.0 poly 0.1 3 1.0\n", argv[0]);
        return 1;
    }

    const char *train_file = argv[1];
    const char *test_file = argv[2];
    const char *output_file = "output.csv";
    int arg_offset = 0;
    
    // Check if 3rd argument is output file or a parameter
    if (argc > 3) {
        const char *arg3 = argv[3];
        // If it contains .csv, .txt, or .out, it's an output file
        if (strstr(arg3, ".csv") != NULL || strstr(arg3, ".txt") != NULL || strstr(arg3, ".out") != NULL) {
            output_file = arg3;
            arg_offset = 1;
        }
    }
    
    double C = (argc > 3 + arg_offset) ? atof(argv[3 + arg_offset]) : 1.0;
    
    KernelType kernel = RBF;
    if (argc > 4 + arg_offset) {
        if (strcmp(argv[4 + arg_offset], "linear") == 0) kernel = LINEAR;
        else if (strcmp(argv[4 + arg_offset], "poly") == 0) kernel = POLYNOMIAL;
        else if (strcmp(argv[4 + arg_offset], "rbf") == 0) kernel = RBF;
    }
    
    double gamma = -1.0; // Auto
    if (argc > 5 + arg_offset) gamma = atof(argv[5 + arg_offset]);
    
    int degree = (argc > 6 + arg_offset) ? atoi(argv[6 + arg_offset]) : 3;
    double coef0 = (argc > 7 + arg_offset) ? atof(argv[7 + arg_offset]) : 0.0;
    int max_iterations = (argc > 8 + arg_offset) ? atoi(argv[8 + arg_offset]) : 1000;

    srand(time(NULL));

    printf("=== Support Vector Machine Classifier ===\n\n");
    printf("Configuration:\n");
    printf("  Training file: %s\n", train_file);
    printf("  Test file: %s\n", test_file);
    printf("  Output file: %s\n", output_file);
    printf("  Kernel: %s\n", kernel_name(kernel));
    printf("  C (regularization): %.4f\n", C);
    if (kernel == RBF || kernel == POLYNOMIAL) {
        if (gamma < 0) printf("  Gamma: auto\n");
        else printf("  Gamma: %.4f\n", gamma);
    }
    if (kernel == POLYNOMIAL) {
        printf("  Degree: %d\n", degree);
        printf("  Coef0: %.4f\n", coef0);
    }
    printf("  Max iterations: %d\n\n", max_iterations);

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

    // Auto-calculate gamma if needed
    if (gamma < 0) {
        gamma = 1.0 / train_data->features;
        printf("Auto gamma: %.4f\n\n", gamma);
    }

    // Create and train model
    printf("Training SVM model...\n");
    SVMModel *model = create_svm_model(train_data->features, train_data->num_classes,
                                       kernel, C, gamma, degree, coef0, max_iterations);
    train_svm_model(model, train_data);
    printf("Training complete!\n\n");

    // Load test data
    printf("Loading test data...\n");
    Dataset *test_data = load_csv(test_file, 1);
    if (!test_data) {
        fprintf(stderr, "Error: Failed to load test data\n");
        free_dataset(train_data);
        free_svm_model(model);
        return 1;
    }
    printf("  Test samples: %d\n\n", test_data->rows);

    // Make predictions
    printf("Making predictions...\n");
    int *predictions = predict_svm(model, test_data);
    
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
    free_svm_model(model);

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
    
    // Count features
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

// Parse a CSV line
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

// Count features in a line
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
        if (labels[i] > max_label) max_label = labels[i];
    }
    return max_label + 1;
}

// Create SVM model
SVMModel* create_svm_model(int features, int num_classes, KernelType kernel,
                          double C, double gamma, int degree, double coef0, int max_iter) {
    SVMModel model = (SVMModel)malloc(sizeof(SVMModel));
    
    model->features = features;
    model->num_classes = num_classes;
    model->kernel_type = kernel;
    model->C = C;
    model->gamma = gamma;
    model->degree = degree;
    model->coef0 = coef0;
    model->max_iterations = max_iter;
    
    // For multiclass, use one-vs-all approach
    model->binary_svms = (SVMBinary*)malloc(num_classes * sizeof(SVMBinary));
    for (int i = 0; i < num_classes; i++) {
        model->binary_svms[i] = create_binary_svm(features, kernel, C, gamma, degree, coef0);
    }
    
    return model;
}

// Create binary SVM
SVMBinary* create_binary_svm(int features, KernelType kernel, double C,
                            double gamma, int degree, double coef0) {
    SVMBinary svm = (SVMBinary)malloc(sizeof(SVMBinary));
    
    svm->features = features;
    svm->kernel_type = kernel;
    svm->C = C;
    svm->gamma = gamma;
    svm->degree = degree;
    svm->coef0 = coef0;
    svm->b = 0.0;
    svm->alpha = NULL;
    svm->w = NULL;
    svm->support_vectors = NULL;
    svm->sv_labels = NULL;
    svm->num_sv = 0;
    svm->errors = NULL;
    
    return svm;
}

// Kernel function
double kernel_function(SVMBinary *svm, double *x1, double *x2, int features) {
    switch (svm->kernel_type) {
        case LINEAR:
            return dot_product(x1, x2, features);
        
        case POLYNOMIAL: {
            double dot = dot_product(x1, x2, features);
            return pow(svm->gamma * dot + svm->coef0, svm->degree);
        }
        
        case RBF: {
            double sum = 0.0;
            for (int i = 0; i < features; i++) {
                double diff = x1[i] - x2[i];
                sum += diff * diff;
            }
            return exp(-svm->gamma * sum);
        }
        
        default:
            return 0.0;
    }
}

// Compute SVM output for a sample
double compute_svm_output(SVMBinary *svm, double *x, double **data, int *labels, int n) {
    double result = 0.0;
    
    for (int i = 0; i < n; i++) {
        if (svm->alpha[i] > 0) {
            result += svm->alpha[i] * labels[i] * kernel_function(svm, data[i], x, svm->features);
        }
    }
    
    return result - svm->b;
}

// SMO algorithm - take step
int take_step(SVMBinary *svm, int i1, int i2, double **data, int *labels, int n) {
    if (i1 == i2) return 0;
    
    double alpha1 = svm->alpha[i1];
    double alpha2 = svm->alpha[i2];
    int y1 = labels[i1];
    int y2 = labels[i2];
    double E1 = svm->errors[i1];
    double E2 = svm->errors[i2];
    int s = y1 * y2;
    
    // Compute bounds
    double L, H;
    if (y1 != y2) {
        L = fmax(0, alpha2 - alpha1);
        H = fmin(svm->C, svm->C + alpha2 - alpha1);
    } else {
        L = fmax(0, alpha1 + alpha2 - svm->C);
        H = fmin(svm->C, alpha1 + alpha2);
    }
    
    if (L >= H) return 0;
    
    // Compute kernel values
    double k11 = kernel_function(svm, data[i1], data[i1], svm->features);
    double k12 = kernel_function(svm, data[i1], data[i2], svm->features);
    double k22 = kernel_function(svm, data[i2], data[i2], svm->features);
    
    double eta = k11 + k22 - 2 * k12;
    
    double a2_new;
    if (eta > 0) {
        a2_new = alpha2 + y2 * (E1 - E2) / eta;
        if (a2_new < L) a2_new = L;
        else if (a2_new > H) a2_new = H;
    } else {
        a2_new = L;
    }
    
    if (fabs(a2_new - alpha2) < EPSILON * (a2_new + alpha2 + EPSILON)) {
        return 0;
    }
    
    double a1_new = alpha1 + s * (alpha2 - a2_new);
    
    // Update threshold b
    double b1 = svm->b + E1 + y1 * (a1_new - alpha1) * k11 + y2 * (a2_new - alpha2) * k12;
    double b2 = svm->b + E2 + y1 * (a1_new - alpha1) * k12 + y2 * (a2_new - alpha2) * k22;
    
    double b_new;
    if (0 < a1_new && a1_new < svm->C) b_new = b1;
    else if (0 < a2_new && a2_new < svm->C) b_new = b2;
    else b_new = (b1 + b2) / 2;
    
    // Update error cache
    double db = b_new - svm->b;
    for (int i = 0; i < n; i++) {
        if (svm->alpha[i] > 0 && svm->alpha[i] < svm->C) {
            double ki1 = kernel_function(svm, data[i], data[i1], svm->features);
            double ki2 = kernel_function(svm, data[i], data[i2], svm->features);
            svm->errors[i] += y1 * (a1_new - alpha1) * ki1 + y2 * (a2_new - alpha2) * ki2 - db;
        }
    }
    
    svm->errors[i1] = 0.0;
    svm->errors[i2] = 0.0;
    
    svm->alpha[i1] = a1_new;
    svm->alpha[i2] = a2_new;
    svm->b = b_new;
    
    return 1;
}

// SMO algorithm - examine example
int examine_example(SVMBinary *svm, int i2, double **data, int *labels, int n) {
    double alpha2 = svm->alpha[i2];
    double E2 = svm->errors[i2];
    int y2 = labels[i2];
    double r2 = E2 * y2;
    
    if ((r2 < -TOL && alpha2 < svm->C) || (r2 > TOL && alpha2 > 0)) {
        // Find i1 with maximum |E1 - E2|
        int i1 = -1;
        double max_diff = 0;
        
        for (int i = 0; i < n; i++) {
            if (svm->alpha[i] > 0 && svm->alpha[i] < svm->C) {
                double diff = fabs(svm->errors[i] - E2);
                if (diff > max_diff) {
                    max_diff = diff;
                    i1 = i;
                }
            }
        }
        
        if (i1 >= 0) {
            if (take_step(svm, i1, i2, data, labels, n)) return 1;
        }
        
        // Try all non-bound examples
        int start = rand() % n;
        for (int i = 0; i < n; i++) {
            i1 = (start + i) % n;
            if (svm->alpha[i1] > 0 && svm->alpha[i1] < svm->C) {
                if (take_step(svm, i1, i2, data, labels, n)) return 1;
            }
        }
        
        // Try all examples
        start = rand() % n;
        for (int i = 0; i < n; i++) {
            i1 = (start + i) % n;
            if (take_step(svm, i1, i2, data, labels, n)) return 1;
        }
    }
    
    return 0;
}

// Train binary SVM using SMO algorithm
void train_binary_svm(SVMBinary *svm, double **data, int *labels, int n, int max_iter) {
    // Initialize alpha and errors
    svm->alpha = (double*)calloc(n, sizeof(double));
    svm->errors = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        svm->errors[i] = -labels[i];
    }
    
    int num_changed = 0;
    int examine_all = 1;
    int iter = 0;
    
    while ((num_changed > 0 || examine_all) && iter < max_iter) {
        num_changed = 0;
        
        if (examine_all) {
            for (int i = 0; i < n; i++) {
                num_changed += examine_example(svm, i, data, labels, n);
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (svm->alpha[i] > 0 && svm->alpha[i] < svm->C) {
                    num_changed += examine_example(svm, i, data, labels, n);
                }
            }
        }
        
        if (examine_all) {
            examine_all = 0;
        } else if (num_changed == 0) {
            examine_all = 1;
        }
        
        iter++;
    }
    
    // Extract support vectors
    int count_sv = 0;
    for (int i = 0; i < n; i++) {
        if (svm->alpha[i] > 0) count_sv++;
    }
    
    svm->num_sv = count_sv;
    svm->support_vectors = (double*)malloc(count_sv * sizeof(double));
    svm->sv_labels = (int*)malloc(count_sv * sizeof(int));
    
    int sv_idx = 0;
    for (int i = 0; i < n; i++) {
        if (svm->alpha[i] > 0) {
            svm->support_vectors[sv_idx] = (double*)malloc(svm->features * sizeof(double));
            memcpy(svm->support_vectors[sv_idx], data[i], svm->features * sizeof(double));
            svm->sv_labels[sv_idx] = labels[i];
            svm->alpha[sv_idx] = svm->alpha[i];
            sv_idx++;
        }
    }
}

// Train SVM model (multiclass using one-vs-all)
void train_svm_model(SVMModel *model, Dataset *train_data) {
    int n = train_data->rows;
    
    for (int c = 0; c < model->num_classes; c++) {
        printf("  Training classifier for class %d...\n", c);
        
        // Create binary labels (1 for class c, -1 for others)
        int binary_labels = (int)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            binary_labels[i] = (train_data->labels[i] == c) ? 1 : -1;
        }
        
        train_binary_svm(model->binary_svms[c], train_data->data, binary_labels, n, model->max_iterations);
        
        printf("    Support vectors: %d\n", model->binary_svms[c]->num_sv);
        
        free(binary_labels);
    }
}

// Predict using binary SVM
double predict_binary_svm(SVMBinary *svm, double *x) {
    double result = 0.0;
    
    for (int i = 0; i < svm->num_sv; i++) {
        result += svm->alpha[i] * svm->sv_labels[i] * 
                 kernel_function(svm, svm->support_vectors[i], x, svm->features);
    }
    
    return result - svm->b;
}

// Predict using SVM model (multiclass)
int* predict_svm(SVMModel *model, Dataset *test_data) {
    int n = test_data->rows;
    int predictions = (int)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) {
        double max_score = -DBL_MAX;
        int predicted_class = 0;
        
        for (int c = 0; c < model->num_classes; c++) {
            double score = predict_binary_svm(model->binary_svms[c], test_data->data[i]);
            if (score > max_score) {
                max_score = score;
                predicted_class = c;
            }
        }
        
        predictions[i] = predicted_class;
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

// Utility functions
double dot_product(double *v1, double *v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

const char* kernel_name(KernelType type) {
    switch (type) {
        case LINEAR: return "Linear";
        case POLYNOMIAL: return "Polynomial";
        case RBF: return "RBF (Gaussian)";
        default: return "Unknown";
    }
}

// Free dataset
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

// Free binary SVM
void free_binary_svm(SVMBinary *svm) {
    if (!svm) return;
    
    if (svm->alpha) free(svm->alpha);
    if (svm->w) free(svm->w);
    if (svm->errors) free(svm->errors);
    if (svm->sv_labels) free(svm->sv_labels);
    
    if (svm->support_vectors) {
        for (int i = 0; i < svm->num_sv; i++) {
            free(svm->support_vectors[i]);
        }
        free(svm->support_vectors);
    }
    
    free(svm);
}

// Free SVM model
void free_svm_model(SVMModel *model) {
    if (!model) return;
    
    for (int i = 0; i < model->num_classes; i++) {
        free_binary_svm(model->binary_svms[i]);
    }
    free(model->binary_svms);
    
    free(model);
}
