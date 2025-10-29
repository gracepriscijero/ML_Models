#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_CLASSES 7  
#define MAX_DATA 20000
#define MAX_FEATURES 16
#define MAX_ITER 10
#define MAX_LINE_LEN 1024
#define MAX_FEATURE_NAME 32

typedef struct {
    int id;  // track ID
    double features[MAX_FEATURES];
    int label;  // 0 = Normal_Weight, 1 = Obesity_Type_I, 2 = Obesity_Type_II, 3 = Obesity_Type_III
                // 4 = Overweight_Level_I, 5 = Overweight_Level_II, 6 = Insufficient_Weight
    double pred[NUM_CLASSES]; // running prediction
} DataPoint;

typedef struct {
    int feature_index;
    double threshold;
    double left_weight[NUM_CLASSES];   // one weight per class
    double right_weight[NUM_CLASSES];  // one weight per class
    double gain;
} TreeNode;


char feature_names[MAX_FEATURES][MAX_FEATURE_NAME];


// Helper Functions 


double gradient(double y, double p) {
    return p - y;
}

double hessian(double p) {
    return p * (1 - p);
}

double compute_gain(double G, double H, double lambda) {
    if (H + lambda == 0) return 0.0;
    return (G * G) / (H + lambda);
}

int map_label(const char *label) {
    if (strcmp(label, "Normal_Weight") == 0) return 0;
    if (strcmp(label, "Insufficient_Weight") == 0) return 1;
    if (strcmp(label, "Overweight_Level_I") == 0) return 2;
    if (strcmp(label, "Overweight_Level_II") == 0) return 3;
    if (strcmp(label, "Obesity_Type_I") == 0) return 4;
    if (strcmp(label, "Obesity_Type_II") == 0) return 5;
    if (strcmp(label, "Obesity_Type_III") == 0) return 6;
    return -1; // unknown label
}


// Category Mapping 

double map_category(const char *value) {
    if (strcmp(value, "yes") == 0) return 1.0;
    if (strcmp(value, "no") == 0) return 0.0;
    if (strcmp(value, "Sometimes") == 0) return 0.33;
    if (strcmp(value, "Frequently") == 0) return 0.66;
    if (strcmp(value, "Always") == 0) return 1.0;
    if (strcmp(value, "Public_Transportation") == 0) return 0.0;
    if (strcmp(value, "Automobile") == 0) return 0.25;
    if (strcmp(value, "Walking") == 0) return 0.5;
    if (strcmp(value, "Bike") == 0) return 0.75;
    if (strcmp(value, "Motorbike") == 0) return 1.0;
    if (strcmp(value, "male") == 0) return 1.0;
    if (strcmp(value, "female") == 0) return 0.0;
    return atof(value);
}


// ====== Load CSV ======

int load_csv(const char *filename, DataPoint *data, int max_features, int *num_features) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("File open failed");
        return -1;
    }

    char line[MAX_LINE_LEN];
    int count = 0;

    // Read header
    if (fgets(line, sizeof(line), fp)) {
        char *token = strtok(line, ",");
        int i = 0;
        while (token && i < max_features + 2) {
            if (i > 0 && i <= max_features) {  // skip id (i=0)
                strncpy(feature_names[i - 1], token, MAX_FEATURE_NAME - 1);
                feature_names[i - 1][strcspn(feature_names[i - 1], "\r\n")] = '\0';
            }
            token = strtok(NULL, ",");
            i++;
        }
    }

    // Read data rows
    while (fgets(line, sizeof(line), fp)) {
        if (count >= MAX_DATA)
            break;

        char *token = strtok(line, ",");
        int col = 0, f = 0;

        while (token) {
            token[strcspn(token, "\r\n")] = 0; // clean newline

            if (col == 0) {
                data[count].id = atoi(token); // first column = ID
            } else if (col <= max_features) {
                data[count].features[f++] = map_category(token);
            } else {
                // last column = label
                data[count].label = map_label(token);

            }

            token = strtok(NULL, ",");
            col++;
        }

        if (f != max_features) {
            fprintf(stderr, "⚠️ Incomplete row at line %d (%d features found)\n", count + 1, f);
            continue;
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            data[count].pred[c] = 0.0; // initialize predictions

        count++;
        *num_features = f;
    }

    fclose(fp);
    return count;
}


// ====== Tree Training ======

TreeNode train_tree(DataPoint *data, int n, int num_features, double lambda) {
    TreeNode best;
    best.gain = -1e9; // initialize to very low value

    for (int f = 0; f < num_features; f++) {
        double best_gain_for_feature = -1e9;
        double best_threshold_for_feature = 0.0;
        double best_GL[NUM_CLASSES], best_HL[NUM_CLASSES];
        double best_GR[NUM_CLASSES], best_HR[NUM_CLASSES];

        for (int i = 0; i < n; i++) {
            double threshold = data[i].features[f];

            double GL[NUM_CLASSES] = {0}, HL[NUM_CLASSES] = {0};
            double GR[NUM_CLASSES] = {0}, HR[NUM_CLASSES] = {0};

            for (int j = 0; j < n; j++) {
                // --- Compute the probabilities ---
                double sum_exp = 0.0;
                double p[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    sum_exp += exp(data[j].pred[c]);
                    p[c] = exp(data[j].pred[c]) / sum_exp;
                }

                // --- Compute gradient & hessian per class ---
                for (int c = 0; c < NUM_CLASSES; c++) {
                    double g = p[c] - (data[j].label == c ? 1.0 : 0.0);
                    double h = p[c] * (1 - p[c]);

                    if (data[j].features[f] <= threshold) {
                        GL[c] += g;
                        HL[c] += h;
                    } else {
                        GR[c] += g;
                        HR[c] += h;
                    }
                }
            }

            // --- Compute gain for this split ---
            double gain = 0.0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                gain += compute_gain(GL[c], HL[c], lambda) +
                        compute_gain(GR[c], HR[c], lambda) -
                        compute_gain(GL[c] + GR[c], HL[c] + HR[c], lambda);
            }

            // Save best split for this feature
            if (gain > best_gain_for_feature) {
                best_gain_for_feature = gain;
                best_threshold_for_feature = threshold;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    best_GL[c] = GL[c];
                    best_HL[c] = HL[c];
                    best_GR[c] = GR[c];
                    best_HR[c] = HR[c];
                }
            }
        }

        // --- Update global best if this feature is better ---
        if (best_gain_for_feature > best.gain) {
            best.gain = best_gain_for_feature;
            best.feature_index = f;
            best.threshold = best_threshold_for_feature;
            for (int c = 0; c < NUM_CLASSES; c++) {
                best.left_weight[c]  = -best_GL[c] / (best_HL[c] + lambda);
                best.right_weight[c] = -best_GR[c] / (best_HR[c] + lambda);
            }
        }

        printf("Feature %s: Best Gain = %.4f | Threshold = %.4f\n",
               feature_names[f], best_gain_for_feature, best_threshold_for_feature);
    }

    return best;
}



// ====== Apply Tree ======

void apply_tree(DataPoint *data, int n, TreeNode tree) {
    for (int i = 0; i < n; i++) {
        if (data[i].features[tree.feature_index] <= tree.threshold) {
            for (int c = 0; c < NUM_CLASSES; c++)
                data[i].pred[c] += tree.left_weight[c];
        } else {
            for (int c = 0; c < NUM_CLASSES; c++)
                data[i].pred[c] += tree.right_weight[c];
        }
    }
}



// ====== Evaluate ======



void evaluate(DataPoint *data, int n) {
    int correct = 0;
    for (int i = 0; i < n; i++) {
        double max_pred = data[i].pred[0];
        int pred_class = 0;
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (data[i].pred[c] > max_pred) {
                max_pred = data[i].pred[c];
                pred_class = c;
            }
        }
        if (pred_class == data[i].label)
            correct++;
    }
    printf("Accuracy: %.2f%%\n", 100.0 * correct / n);
}


void print_sorted_feature_importance(TreeNode trees[], int T, char feature_names[MAX_FEATURES][MAX_FEATURE_NAME]) {
    double feature_importance[MAX_FEATURES] = {0};

    // Aggregate gains per feature
    for (int t = 0; t < T; t++) {
        int f = trees[t].feature_index;
        feature_importance[f] += trees[t].gain;
    }

    // Sort features by importance (descending)
    for (int i = 0; i < MAX_FEATURES - 1; i++) {
        for (int j = i + 1; j < MAX_FEATURES; j++) {
            if (feature_importance[j] > feature_importance[i]) {
                // Swap gains
                double temp_gain = feature_importance[i];
                feature_importance[i] = feature_importance[j];
                feature_importance[j] = temp_gain;

                // Swap feature names
                char temp_name[MAX_FEATURE_NAME];
                strcpy(temp_name, feature_names[i]);
                strcpy(feature_names[i], feature_names[j]);
                strcpy(feature_names[j], temp_name);
            }
        }
    }

    // Print results
    printf("\nFeature Importance (sorted):\n");
    for (int i = 0; i < MAX_FEATURES; i++) {
        if (feature_importance[i] > 0)
            printf("%s: %.4f\n", feature_names[i], feature_importance[i]);
    }

    FILE *fp = fopen("feature_importance_xg.csv", "w");
    fprintf(fp, "Feature,Importance\n");
    for (int i = 0; i < MAX_FEATURES; i++) {
        if (feature_importance[i] > 0)
            fprintf(fp, "%s,%.4f\n", feature_names[i], feature_importance[i]);
    }
    fclose(fp);

}



// ====== Main ======

int main() {
    DataPoint train[MAX_DATA], test[MAX_DATA];
    int num_train_features = 0, num_test_features = 0;

    int n_train = load_csv("train.csv", train, MAX_FEATURES, &num_train_features);
    int n_test = load_csv("test.csv", test, MAX_FEATURES, &num_test_features);

    if (n_train <= 0 || n_test <= 0) {
        printf("Failed to load data.\n");
        return 1;
    }

    int T = MAX_ITER;
    double lambda = 1.0;
    TreeNode trees[MAX_ITER];

    printf("\n--- Training XGBoost-like Model ---\n");
    for (int t = 0; t < T; t++) {
        trees[t] = train_tree(train, n_train, num_train_features, lambda);
        apply_tree(train, n_train, trees[t]);
        printf("Iteration %d: Gain = %.4f | Feature = %s | Threshold = %.2f\n",
               t + 1, trees[t].gain,
               feature_names[trees[t].feature_index],
               trees[t].threshold);
    }

    printf("\n--- Testing ---\n");
    for (int i = 0; i < n_test; i++)
    for (int c = 0; c < NUM_CLASSES; c++)
        test[i].pred[c] = 0.0;


    for (int t = 0; t < T; t++)
        apply_tree(test, n_test, trees[t]);

    evaluate(test, n_test);

    FILE *out_fp = fopen("outputxg.csv", "w");
    if (!out_fp) {
        perror("Failed to open output file");
        return 1;
    }
    const char *label_names[] = {"Normal_Weight", "Insufficient_Weight", "Overweight_Level_I",
                                 "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
                                 "Obesity_Type_III"};

    fprintf(out_fp, "id,Prediction\n");
    for (int i = 0; i < n_test; i++) {
        int final_pred = 0;
        double max_pred = test[i].pred[0]; // start with first class

        // Find class with highest prediction score
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (test[i].pred[c] > max_pred) {
                max_pred = test[i].pred[c];
                final_pred = c;
            }
        }

        // Write label name instead of 0/1
        fprintf(out_fp, "%d,%s\n", test[i].id, label_names[final_pred]);
    }

    fclose(out_fp);

    printf("Predictions written to outputxg.csv\n");

    print_sorted_feature_importance(trees, T, feature_names);

    return 0;
}
