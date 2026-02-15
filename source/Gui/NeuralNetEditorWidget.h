#pragma once

#include <EngineInterface/GenomeDescription.h>

#include "Definitions.h"

class _NeuralNetEditorWidget
{
public:
    static NeuralNetEditorWidget create();

    void process(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<float>& connectionWeights);

private:
    _NeuralNetEditorWidget();

    struct SelectionData
    {
        int neuronIndex = 0;
    };

    void processActionButtons(std::vector<NeuralNetWeight>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

    template <typename T>
    SelectionData& getValueRef(std::unordered_map<unsigned int, T>& idToValueMap);

    std::unordered_map<unsigned int, SelectionData> _dataById;

    struct NetData
    {
        std::vector<NeuralNetWeight> weights;
        std::vector<float> biases;
        std::vector<ActivationFunction> activationFunctions;
    };
    std::optional<NetData> _copiedNet;
};
