#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"

class _NeuralNetWidget
{
public:
    static NeuralNetWidget create();

    void process(std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

private:
    _NeuralNetWidget();

    struct SelectionData
    {
        int inputNeuronIndex = 0;
        int outputNeuronIndex = 0;
    };

    void processNetwork(SelectionData& selectionData, std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);
    void processEditWidgets(
        SelectionData& selectionData,
        std::vector<float>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions);
    void processActionButtons(std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

    template <typename T>
    SelectionData& getValueRef(std::unordered_map<unsigned int, T>& idToValueMap);

    std::unordered_map<unsigned int, SelectionData> _dataById;

    struct NetData
    {
        std::vector<float> weights;
        std::vector<float> biases;
        std::vector<ActivationFunction> activationFunctions;
    };
    std::optional<NetData> _copiedNet;
};
