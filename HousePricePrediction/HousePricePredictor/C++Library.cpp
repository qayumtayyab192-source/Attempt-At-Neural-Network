#include <iostream>
#include <fstream>   
#include <sstream>   
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <random>

void InitializeWeightFile(const std::string& filename, int rows, int cols, double scale) {
    if (std::filesystem::exists(filename)) return; 
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-scale, scale); 

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << dis(gen) << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
}

void InitializeBiasFile(const std::string& filename, int rows) {
    if (std::filesystem::exists(filename)) return; 
    std::ofstream file(filename);
    for (int i = 0; i < rows; ++i) {
        file << 0.0 << "\n";
    }
}

std::vector<std::vector<double>> MatrixMultiplier(const std::vector<std::vector<double>>& MatrixA, const std::vector<std::vector<double>>& MatrixB) {
    int RowsA = MatrixA.size();
    int ColumnsA = MatrixA[0].size();
    int ColumnsB = MatrixB[0].size();
    std::vector<std::vector<double>> ResultantMatrix(RowsA, std::vector<double>(ColumnsB, 0.0));

    for (int i = 0; i < ColumnsB; i++) {
        for (int j = 0; j < RowsA; j++) {
            for (int k = 0; k < ColumnsA; k++) {
                ResultantMatrix[j][i] += MatrixA[j][k] * MatrixB[k][i];
            }
        }
    }
    return ResultantMatrix;
}

std::vector<std::vector<double>> MatrixAddition(const std::vector<std::vector<double>>& MatrixA, const std::vector<std::vector<double>>& MatrixB) {
    size_t rowsA = MatrixA.size();
    size_t colsA = MatrixA[0].size();
    std::vector<std::vector<double>> ResultantMatrix(rowsA, std::vector<double>(colsA, 0.0));
    
    // Check if MatrixB is a column vector (rows = colsA, cols = 1)
    bool isColumnVector = (MatrixB.size() == colsA && MatrixB[0].size() == 1);

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsA; j++) {
            if (isColumnVector) {
                ResultantMatrix[i][j] = MatrixA[i][j] + MatrixB[j][0];
            } else {
                // Fallback for a standard row vector layout (1 x colsA)
                ResultantMatrix[i][j] = MatrixA[i][j] + MatrixB[0][j];
            }
        }
    }
    return ResultantMatrix;
}

std::vector<std::vector<double>> ErrorValues(const std::vector<std::vector<double>>& CalculatedValues, const std::vector<std::vector<double>>& ActualValues) {
    std::vector<std::vector<double>> Err(ActualValues.size(), std::vector<double>(1, 0.0));
    for (size_t i = 0; i < ActualValues.size(); i++) {
        for (size_t j = 0; j < ActualValues[0].size(); j++) {
            Err[i][j] = CalculatedValues[i][j] - ActualValues[i][j]; 
        }
    }
    return Err;
}

std::vector<std::vector<double>> MatrixTransposer(const std::vector<std::vector<double>>& MatrixStarted) {
    std::vector<std::vector<double>> TransposedMatrix(MatrixStarted[0].size(), std::vector<double>(MatrixStarted.size(), 0.0));
    for (size_t i = 0; i < MatrixStarted.size(); i++) {
        for (size_t j = 0; j < MatrixStarted[0].size(); j++) {
            TransposedMatrix[j][i] = MatrixStarted[i][j];
        }
    }
    return TransposedMatrix;
}

std::vector<std::vector<double>> LeakyReLUActivation(const std::vector<std::vector<double>>& InitialMatrix) {
    std::vector<std::vector<double>> Result = InitialMatrix;
    for (size_t i = 0; i < InitialMatrix.size(); i++) {
        for (size_t j = 0; j < InitialMatrix[0].size(); j++) {
            double val = InitialMatrix[i][j];
            Result[i][j] = (val > 0.0) ? val : 0.01 * val;
        }
    }
    return Result;
}

std::vector<std::vector<double>> LeakyReLUDerivative(const std::vector<std::vector<double>>& ActivatedMatrix) {
    std::vector<std::vector<double>> Result = ActivatedMatrix;
    for (size_t i = 0; i < ActivatedMatrix.size(); i++) {
        for (size_t j = 0; j < ActivatedMatrix[0].size(); j++) {
            double val = ActivatedMatrix[i][j];
            Result[i][j] = (val > 0.0) ? 1.0 : 0.01;
        }
    }
    return Result;
}

std::vector<std::vector<double>> WeightUpdater(const std::vector<std::vector<double>>& Gradient, const std::vector<std::vector<double>>& InitialWeights, double LearningRate) {
    std::vector<std::vector<double>> FinalMatrix = InitialWeights;
    for (size_t i = 0; i < InitialWeights.size(); ++i) {
        for (size_t j = 0; j < InitialWeights[0].size(); ++j) {
            FinalMatrix[i][j] -= Gradient[i][j] * LearningRate;
        }
    }
    return FinalMatrix;
}

std::vector<std::vector<double>> BiasUpdater(const std::vector<std::vector<double>>& ErrorMatrix, const std::vector<std::vector<double>>& InitialBiases, double LearningRate) {
    std::vector<std::vector<double>> FinalMatrix = InitialBiases;
    size_t rows = ErrorMatrix.size();        
    size_t num_biases = InitialBiases.size(); 

    for (size_t j = 0; j < num_biases; ++j) {
        double errorSum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            errorSum += ErrorMatrix[i][j];
        }
        
        // Calculate the raw update step
        double update = (errorSum / rows) * LearningRate;

        // SAFEGUARD: Prevent bias explosion by clamping the maximum step size
        const double max_bias_step = 0.5;
        if (update > max_bias_step) update = max_bias_step;
        if (update < -max_bias_step) update = -max_bias_step;

        FinalMatrix[j][0] -= update; 
    }
    return FinalMatrix;
}

std::vector<std::vector<double>> ElementwiseMultiply(const std::vector<std::vector<double>>& MatrixA, const std::vector<std::vector<double>>& MatrixB) {
    std::vector<std::vector<double>> Result = MatrixA;
    for (size_t i = 0; i < MatrixA.size(); i++) {
        for (size_t j = 0; j < MatrixA[0].size(); j++) {
            Result[i][j] *= MatrixB[i][j];
        }
    }
    return Result;
}

std::vector<std::vector<double>> Normalisation(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return matrix;
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> normalizedMatrix = matrix;

    for (size_t j = 0; j < cols; ++j) {
        double min_val = matrix[0][j];
        double max_val = matrix[0][j];
        for (size_t i = 1; i < rows; ++i) {
            if (matrix[i][j] < min_val) min_val = matrix[i][j];
            if (matrix[i][j] > max_val) max_val = matrix[i][j];
        }
        double diff = max_val - min_val;
        for (size_t i = 0; i < rows; ++i) {
            normalizedMatrix[i][j] = (diff == 0.0) ? 0.0 : (matrix[i][j] - min_val) / diff;
        }
    }
    return normalizedMatrix;
}

double UnnormalisePrice(double normalizedPrice, const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty() || matrix.front().empty()) {
        return normalizedPrice;
    }

    double min_val = matrix.front()[0];
    double max_val = matrix.front()[0];
    for (const auto& row : matrix) {
        if (row.empty()) continue;
        min_val = std::min(min_val, row[0]);
        max_val = std::max(max_val, row[0]);
    }

    const double diff = max_val - min_val;
    return diff == 0.0 ? min_val : normalizedPrice * diff + min_val;
}

std::vector<std::vector<double>> FileImporter(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;
            while (iss >> value) row.push_back(value);
            if (!row.empty()) matrix.push_back(row);
        }
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
    return matrix;
}

void FileExporter(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& row : matrix) {
            for (size_t j = 0; j < row.size(); ++j) {
                file << row[j] << (j < row.size() - 1 ? " " : "");
            }
            file << "\n";
        }
    }
}

double MeanSquaredError(const std::vector<std::vector<double>>& predicted, const std::vector<std::vector<double>>& actual) {
    double sum = 0.0;
    size_t n = predicted.size();
    for (size_t i = 0; i < n; ++i) {
        double err = predicted[i][0] - actual[i][0];
        sum += err * err;
    }
    return sum / n;
}

void ClipMatrix(std::vector<std::vector<double>>& matrix, double threshold) {
    for (auto& row : matrix) {
        for (auto& val : row) {
            if (val > threshold) val = threshold;
            else if (val < -threshold) val = -threshold;
        }
    }
}

int main() {
    InitializeWeightFile("LayerOneWeights.txt", 8, 32, 0.05);
    InitializeBiasFile("LayerOneBiases.txt", 32);
    InitializeWeightFile("LayerTwoWeights.txt", 32, 16, 0.05);
    InitializeBiasFile("LayerTwoBiases.txt", 16);
    InitializeWeightFile("LayerThreeWeights.txt", 16, 1, 0.05);
    InitializeBiasFile("LayerThreeBiases.txt", 1);

    std::cout << "Loading dataset..." << std::endl;
    std::vector<std::vector<double>> InputMatrix = FileImporter("HouseInformation.txt");
    std::vector<std::vector<double>> ActualPrices = FileImporter("ActualPrices.txt");

    std::vector<std::vector<double>> Weights1Matrix = FileImporter("LayerOneWeights.txt");
    std::vector<std::vector<double>> Biases1Matrix = FileImporter("LayerOneBiases.txt");
    std::vector<std::vector<double>> Weights2Matrix = FileImporter("LayerTwoWeights.txt");
    std::vector<std::vector<double>> Biases2Matrix = FileImporter("LayerTwoBiases.txt");
    std::vector<std::vector<double>> Weights3Matrix = FileImporter("LayerThreeWeights.txt");
    std::vector<std::vector<double>> Biases3Matrix = FileImporter("LayerThreeBiases.txt");

    if (InputMatrix.empty() || ActualPrices.empty()) {
        std::cerr << "Dataset loading failed. Ensure HouseInformation.txt and ActualPrices.txt exist." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> NormalisedInput = Normalisation(InputMatrix);
    std::vector<std::vector<double>> NormalisedPrices = Normalisation(ActualPrices);

    int NumberOfEpochs = 10;
    double LearningRate = 0.001;
    double MSE = 1.0;
    int BatchSize = 64;
    double m = static_cast<double>(NormalisedInput.size());

    std::cout << "Starting training loop..." << std::endl;

    for (int epoch = 0; epoch < NumberOfEpochs; ++epoch) {
        for (size_t i = 0; i < NormalisedInput.size(); i += BatchSize) {
            size_t currentBatchSize = std::min(static_cast<size_t>(BatchSize), NormalisedInput.size() - i);

            std::vector<std::vector<double>> BatchInput(
                NormalisedInput.begin() + i, 
                NormalisedInput.begin() + i + currentBatchSize
            );
            std::vector<std::vector<double>> BatchPrices(
                NormalisedPrices.begin() + i, 
                NormalisedPrices.begin() + i + currentBatchSize
            );

            double m_batch = static_cast<double>(currentBatchSize);

            // 2. Forward Pass on the mini-batch
            std::vector<std::vector<double>> OutputMatrix1 = MatrixAddition(MatrixMultiplier(BatchInput, Weights1Matrix), Biases1Matrix);
            std::vector<std::vector<double>> ActivatedOutput1 = LeakyReLUActivation(OutputMatrix1);

            std::vector<std::vector<double>> OutputMatrix2 = MatrixAddition(MatrixMultiplier(ActivatedOutput1, Weights2Matrix), Biases2Matrix);
            std::vector<std::vector<double>> ActivatedOutput2 = LeakyReLUActivation(OutputMatrix2);

            std::vector<std::vector<double>> OutputMatrix3 = MatrixAddition(MatrixMultiplier(ActivatedOutput2, Weights3Matrix), Biases3Matrix);

            // 3. Backward Pass on the mini-batch
            std::vector<std::vector<double>> ErrorMatrix3 = ErrorValues(OutputMatrix3, BatchPrices);

            std::vector<std::vector<double>> GradientDescent3 = MatrixMultiplier(MatrixTransposer(ActivatedOutput2), ErrorMatrix3);
            for (auto& row : GradientDescent3) for (auto& val : row) val /= m_batch;
            ClipMatrix(GradientDescent3, 5.0);
            Weights3Matrix = WeightUpdater(GradientDescent3, Weights3Matrix, LearningRate);
            Biases3Matrix = BiasUpdater(ErrorMatrix3, Biases3Matrix, LearningRate);

            std::vector<std::vector<double>> Derivative2 = LeakyReLUDerivative(ActivatedOutput2);
            std::vector<std::vector<double>> ErrorMatrix2 = ElementwiseMultiply(MatrixMultiplier(ErrorMatrix3, MatrixTransposer(Weights3Matrix)), Derivative2);
            std::vector<std::vector<double>> GradientDescent2 = MatrixMultiplier(MatrixTransposer(ActivatedOutput1), ErrorMatrix2);
            
            for (auto& row : GradientDescent2) for (auto& val : row) val /= m_batch;
            ClipMatrix(GradientDescent2, 5.0);
            Weights2Matrix = WeightUpdater(GradientDescent2, Weights2Matrix, LearningRate);
            Biases2Matrix = BiasUpdater(ErrorMatrix2, Biases2Matrix, LearningRate);

            std::vector<std::vector<double>> Derivative1 = LeakyReLUDerivative(ActivatedOutput1);
            std::vector<std::vector<double>> ErrorMatrix1 = ElementwiseMultiply(MatrixMultiplier(ErrorMatrix2, MatrixTransposer(Weights2Matrix)), Derivative1);
            std::vector<std::vector<double>> GradientDescent1 = MatrixMultiplier(MatrixTransposer(BatchInput), ErrorMatrix1);
            
            for (auto& row : GradientDescent1) for (auto& val : row) val /= m_batch;
            ClipMatrix(GradientDescent1, 5.0);
            Weights1Matrix = WeightUpdater(GradientDescent1, Weights1Matrix, LearningRate);
            Biases1Matrix = BiasUpdater(ErrorMatrix1, Biases1Matrix, LearningRate);
        }

        std::cout << "Completed Epoch " << epoch + 1 << std::endl;
    }
     

    FileExporter("LayerOneWeights.txt", Weights1Matrix);
    FileExporter("LayerOneBiases.txt", Biases1Matrix);
    FileExporter("LayerTwoWeights.txt", Weights2Matrix);
    FileExporter("LayerTwoBiases.txt", Biases2Matrix);
    FileExporter("LayerThreeWeights.txt", Weights3Matrix);
    FileExporter("LayerThreeBiases.txt", Biases3Matrix);

    std::vector<std::vector<double>> FinalOut1 = LeakyReLUActivation(MatrixAddition(MatrixMultiplier(NormalisedInput, Weights1Matrix), Biases1Matrix));
    std::vector<std::vector<double>> FinalOut2 = LeakyReLUActivation(MatrixAddition(MatrixMultiplier(FinalOut1, Weights2Matrix), Biases2Matrix));
    std::vector<std::vector<double>> FinalOut3 = MatrixAddition(MatrixMultiplier(FinalOut2, Weights3Matrix), Biases3Matrix);
    MSE = MeanSquaredError(FinalOut3, NormalisedPrices);

    std::cout << "Training complete! Final MSE: " << MSE << std::endl;
    return 0;

    /*
  
    std::vector<std::vector<double>> Weights1Matrix = FileImporter("LayerOneWeights.txt");
    std::vector<std::vector<double>> Weights2Matrix = FileImporter("LayerTwoWeights.txt");
    std::vector<std::vector<double>> Weights3Matrix = FileImporter("LayerThreeWeights.txt");
    std::vector<std::vector<double>> Biases1Matrix = FileImporter("LayerOneBiases.txt");
    std::vector<std::vector<double>> Biases2Matrix = FileImporter("LayerTwoBiases.txt");
    std::vector<std::vector<double>> Biases3Matrix = FileImporter("LayerThreeBiases.txt");
    std::vector<std::vector<double>> ActualPrices = FileImporter("ActualPrices.txt");
    std::vector<std::vector<double>> InputData = FileImporter("HouseInformation.txt");

    double MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude;

    std::cout << "Enter Median Income: "; std::cin >> MedInc;
    std::cout << "Enter House Age: "; std::cin >> HouseAge;
    std::cout << "Enter Average Rooms: "; std::cin >> AveRooms;
    std::cout << "Enter Average Bedrooms: "; std::cin >> AveBedrms;
    std::cout << "Enter Population: "; std::cin >> Population;
    std::cout << "Enter Average Occupancy: "; std::cin >> AveOccup;
    std::cout << "Enter Latitude: "; std::cin >> Latitude;
    std::cout << "Enter Longitude: "; std::cin >> Longitude;

    std::vector<double> InputMatrix = {MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude};
    InputData.push_back(InputMatrix);

    std::vector<std::vector<double>> NormalisedMatrix = Normalisation(InputData);
    std::vector<std::vector<double>> NormalisedInput = {NormalisedMatrix.back()};

    std::vector<std::vector<double>> OutputMatrix1 = MatrixAddition(MatrixMultiplier(NormalisedInput, Weights1Matrix), Biases1Matrix);
    std::vector<std::vector<double>> ActivatedOutput1 = LeakyReLUActivation(OutputMatrix1);
    
    std::vector<std::vector<double>> OutputMatrix2 = MatrixAddition(MatrixMultiplier(ActivatedOutput1, Weights2Matrix), Biases2Matrix);
    std::vector<std::vector<double>> ActivatedOutput2 = LeakyReLUActivation(OutputMatrix2);
    
    std::vector<std::vector<double>> OutputMatrix3 = MatrixAddition(MatrixMultiplier(ActivatedOutput2, Weights3Matrix), Biases3Matrix);
    double PredictedPrice = UnnormalisePrice(OutputMatrix3[0][0], ActualPrices) * 100000.0;

    if (PredictedPrice < 0) {
        PredictedPrice *= -1;
    } 
    
    std::cout << "Predicted House Price: " << PredictedPrice << std::endl;
    */
}
//8.2816 29.0 6.715625 1.05416666666666672 184.0 2.275 34.12 -118.39