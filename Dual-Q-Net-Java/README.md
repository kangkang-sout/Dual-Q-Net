# Dual Q-Net Java Implementation

A Java implementation of the Dual Q-matrix-Embedded Neural Networks for Cognitive Diagnosis using Deeplearning4j (DL4J).

## Overview

This project is a Java port of the original Python implementation of Dual Q-Net, a neural network method for cognitive diagnosis based on Q-matrix constraints. The implementation introduces Q-matrix constraints from traditional cognitive diagnosis to ensure the reliability and interpretability of the network structure.

## Features

- **Three Neural Network Models**:
  - MLP: Multi-Layer Perceptron
  - ANN: Artificial Neural Network
  - QNet: Dual Q-matrix-Embedded Neural Network (main contribution)

- **Q-matrix Constraints**: Incorporates Q-matrix and Q-star matrix constraints for cognitive diagnosis
- **Comprehensive Metrics**: AAR (Attribute Accuracy Rate) and PAR (Pattern Accuracy Rate)
- **Synthetic Data Generation**: Built-in DINA model-based data generation for testing

## Requirements

- Java 11 or higher
- Maven 3.6+

## Dependencies

- Deeplearning4j 1.0.0-M2.1
- ND4J (N-Dimensional Arrays for Java)
- Apache Commons Math3
- SLF4J for logging

## Project Structure

```
src/main/java/com/cognitivediagnosis/
├── Main.java                    # Main entry point
├── models/
│   ├── BaseNet.java            # Abstract base class for neural networks
│   ├── MLP.java                # Multi-Layer Perceptron
│   ├── ANN.java                # Artificial Neural Network
│   ├── QNet.java               # Dual Q-matrix-Embedded Neural Network
│   └── TrainingHistory.java    # Training metrics storage
└── data/
    └── DataProcessor.java      # Data processing utilities
```

## Quick Start

1. **Clone and build the project**:
   ```bash
   git clone <repository-url>
   cd dual-q-net-java
   mvn clean compile
   ```

2. **Run the example**:
   ```bash
   mvn exec:java -Dexec.mainClass="com.cognitivediagnosis.Main"
   ```

3. **Or build and run the JAR**:
   ```bash
   mvn clean package
   java -jar target/dual-q-net-java-1.0.0.jar
   ```

## Usage Example

```java
// Load Q-matrix
INDArray qMatrix = DataProcessor.getSampleQMatrix1();

// Generate Q-star matrix
INDArray qApos = DataProcessor.generateQApos(4, qMatrix.shape()[1]);
INDArray qStar = DataProcessor.generateQStar(qMatrix, qApos);

// Create and initialize Dual Q-Net
QNet qnet = new QNet(qMatrix, qStar);
qnet.initializeNetwork(inputSize, outputSize);

// Generate synthetic data
DataProcessor.DataSplit data = DataProcessor.generateSyntheticData(qMatrix, 500, 123L);

// Create data iterators
DataSetIterator trainIterator = DataProcessor.createDataSetIterator(
    data.getTrainFeatures(), data.getTrainLabels(), 64, true);
DataSetIterator testIterator = DataProcessor.createDataSetIterator(
    data.getTestFeatures(), data.getTestLabels(), 64, false);

// Train the model
TrainingHistory history = qnet.trainNetwork(trainIterator, testIterator, 100, true);

// Make predictions
INDArray predictions = qnet.predict(data.getTestFeatures(), true);
```

## Architecture

### Dual Q-Net Architecture
The Dual Q-Net consists of three main layers:

1. **Main Constraint Layer (mc)**: Items → Attributes (ReLU activation)
2. **Latent Constraint Layer (lc)**: Items → Q-star Attributes (Tanh activation)
3. **Combination Layer (cc)**: (Attributes + Q-star) → Attributes (Sigmoid activation)

The network applies Q-matrix constraints during training to ensure interpretability.

## Metrics

- **AAR (Attribute Accuracy Rate)**: Average accuracy across all attributes
- **PAR (Pattern Accuracy Rate)**: Accuracy of complete attribute patterns

## Differences from Python Implementation

- Uses DL4J instead of PyTorch for neural network operations
- Simplified constraint application (applied per epoch instead of per batch)
- No R integration (Python version used rpy2 for traditional CDM methods)
- Built-in synthetic data generation using DINA model

## Performance

The Java implementation provides comparable performance to the Python version while offering:
- Better integration with enterprise Java environments
- Improved type safety
- Enhanced performance through JVM optimizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project maintains the same license as the original Python implementation.

## References

Original Python implementation and research paper references should be cited when using this implementation.

## Contact

For questions or issues, please open an issue on the project repository.
