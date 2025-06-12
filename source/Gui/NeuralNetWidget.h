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

    std::unordered_map<unsigned int, int> _neuronSelectedInput;
    std::unordered_map<unsigned int, int> _neuronSelectedOutput;
};
