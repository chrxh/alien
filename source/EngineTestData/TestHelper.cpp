#include "TestHelper.h"

namespace
{
    // Compare NeuralNetworkDesc with tolerance for weight serialization precision loss
    // Weights are stored as int8_t with scale factor of 32, giving ~0.03 quantization step
    bool compareNeuralNetwork(NeuralNetworkDesc const& left, NeuralNetworkDesc const& right)
    {
        constexpr float weightSerializationTolerance = 0.04f;  // int8_t quantization: 1/32 = 0.03125, plus margin
        if (left._weights.size() != right._weights.size()) {
            return false;
        }
        for (size_t i = 0; i < left._weights.size(); ++i) {
            if (!TestHelper::approxCompare(left._weights[i], right._weights[i], weightSerializationTolerance)) {
                return false;
            }
        }
        if (left._biases != right._biases) {
            return false;
        }
        if (left._activationFunctions != right._activationFunctions) {
            return false;
        }
        if (left._connectionWeights != right._connectionWeights) {
            return false;
        }
        return true;
    }

    // Compare CellDesc with special handling for NeuralNetwork (weight serialization tolerance)
    bool compareCellDesc(CellDesc const& left, CellDesc const& right)
    {
        if (!compareNeuralNetwork(left._neuralNetwork, right._neuralNetwork)) {
            return false;
        }
        // Compare all other fields using a modified CellDesc with zeroed weights
        CellDesc leftCopy = left;
        CellDesc rightCopy = right;
        leftCopy._neuralNetwork._weights.clear();
        rightCopy._neuralNetwork._weights.clear();
        return leftCopy == rightCopy;
    }
}

bool TestHelper::approxCompare(double expected, double actual, float precision /* = 0.001f*/)
{
    return approxCompare(toFloat(expected), toFloat(actual), precision);
}

bool TestHelper::approxCompare(float expected, float actual, float precision /* = 0.001f*/)
{
    auto absNorm = std::abs(expected) + std::abs(actual);
    if (absNorm < precision) {
        return true;
    }
    return std::abs(expected - actual) / absNorm < precision;
}

bool TestHelper::approxCompare(RealVector2D const& expected, RealVector2D const& actual, float precision /* = 0.001f*/)
{
    return approxCompare(expected.x, actual.x, precision) && approxCompare(expected.y, actual.y, precision);
}

bool TestHelper::approxCompare(std::vector<float> const& expected, std::vector<float> const& actual, float precision /* = 0.001f*/)
{
    if (expected.size() != actual.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approxCompare(expected[i], actual[i], precision)) {
            return false;
        }
    }
    return true;
}

bool TestHelper::approxCompareAngles(float expected, float actual, float precision /* = 0.001f*/)
{
    return approxCompare(Math::getNormalizedAngle(expected - actual, -180.0f), 0.0f, precision);
}

bool TestHelper::compare(Desc left, Desc right)
{
    std::sort(left._objects.begin(), left._objects.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._objects.begin(), right._objects.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._energies.begin(), left._energies.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._energies.begin(), right._energies.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._creatures.begin(), left._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._creatures.begin(), right._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(left._genomes.begin(), left._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    std::sort(right._genomes.begin(), right._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
    
    // Compare objects with special handling for FP16 weights
    if (left._objects.size() != right._objects.size()) {
        return false;
    }
    for (size_t i = 0; i < left._objects.size(); ++i) {
        if (!compare(left._objects[i], right._objects[i])) {
            return false;
        }
    }
    
    // Compare genomes with special handling for weight serialization tolerance
    if (left._genomes.size() != right._genomes.size()) {
        return false;
    }
    for (size_t i = 0; i < left._genomes.size(); ++i) {
        if (!compare(left._genomes[i], right._genomes[i])) {
            return false;
        }
    }
    
    // Clear objects and genomes to avoid double comparison
    left._objects.clear();
    right._objects.clear();
    left._genomes.clear();
    right._genomes.clear();
    
    return left == right;
}

bool TestHelper::compare(ObjectDesc left, ObjectDesc right)
{
    left._id = 0;
    right._id = 0;
    
    // For cells, use special comparison that handles FP16 precision loss in weights
    if (left.getObjectType() == ObjectType_Cell && right.getObjectType() == ObjectType_Cell) {
        if (!compareCellDesc(left.getCellRef(), right.getCellRef())) {
            return false;
        }
        // Clear neural network weights to avoid double comparison in operator==
        left.getCellRef()._neuralNetwork._weights.clear();
        right.getCellRef()._neuralNetwork._weights.clear();
    }
    return left == right;
}

bool TestHelper::compare(EnergyDesc left, EnergyDesc right)
{
    left._id = 0;
    right._id = 0;
    return left == right;
}

namespace
{
    // Compare NeuralNetworkGenomeDesc with tolerance for weight serialization precision loss
    bool compareNeuralNetworkGenome(NeuralNetworkGenomeDesc const& left, NeuralNetworkGenomeDesc const& right)
    {
        constexpr float weightSerializationTolerance = 0.04f;  // int8_t quantization: 1/32 = 0.03125, plus margin
        if (left._weights.size() != right._weights.size()) {
            return false;
        }
        for (size_t i = 0; i < left._weights.size(); ++i) {
            if (!TestHelper::approxCompare(left._weights[i], right._weights[i], weightSerializationTolerance)) {
                return false;
            }
        }
        if (left._biases != right._biases) {
            return false;
        }
        if (left._activationFunctions != right._activationFunctions) {
            return false;
        }
        if (left._connectionWeights != right._connectionWeights) {
            return false;
        }
        return true;
    }

    bool compareNodeDesc(NodeDesc left, NodeDesc right)
    {
        if (!compareNeuralNetworkGenome(left._neuralNetwork, right._neuralNetwork)) {
            return false;
        }
        // Clear weights to use default comparison for remaining fields
        left._neuralNetwork._weights.clear();
        right._neuralNetwork._weights.clear();
        return left == right;
    }

    bool compareGeneDesc(GeneDesc const& left, GeneDesc const& right)
    {
        if (left._nodes.size() != right._nodes.size()) {
            return false;
        }
        for (size_t i = 0; i < left._nodes.size(); ++i) {
            if (!compareNodeDesc(left._nodes[i], right._nodes[i])) {
                return false;
            }
        }
        // Compare remaining fields - need to copy and clear nodes for default comparison
        GeneDesc leftCopy = left;
        GeneDesc rightCopy = right;
        leftCopy._nodes.clear();
        rightCopy._nodes.clear();
        return leftCopy == rightCopy;
    }
}

bool TestHelper::compare(GenomeDesc const& left, GenomeDesc const& right)
{
    if (left._genes.size() != right._genes.size()) {
        return false;
    }
    for (size_t i = 0; i < left._genes.size(); ++i) {
        if (!compareGeneDesc(left._genes[i], right._genes[i])) {
            return false;
        }
    }
    // Compare remaining fields - copy and clear genes
    GenomeDesc leftCopy = left;
    GenomeDesc rightCopy = right;
    leftCopy._genes.clear();
    rightCopy._genes.clear();
    // Also clear IDs for comparison
    leftCopy._id = 0;
    rightCopy._id = 0;
    return leftCopy == rightCopy;
}
