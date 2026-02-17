#pragma once

#include <imgui.h>

#include <EngineInterface/GenomeDesc.h>

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

    struct LayoutData
    {
        float width = 0;
        float connectionButtonWidth = 0;
        float channelButtonWidth = 0;
        ImVec2 connectionButtonBottomLeft[MAX_OBJECT_CONNECTIONS];
        ImVec2 connectionButtonBottomRight[MAX_OBJECT_CONNECTIONS];
        ImVec2 channelsButtonTopCenter;
        ImVec2 inputChannelBottomCenter[MAX_CHANNELS];
        ImVec2 inputButtonTopLeft[MAX_CHANNELS];
        ImVec2 neuronTopCenter[MAX_CHANNELS];
    };

    void processConnectionWeightSliders(std::vector<float>& connectionWeights, LayoutData& layout);
    void processChannelWeightSliders(std::vector<NeuralNetWeight>& weights, SelectionData& selectionData, LayoutData& layout);
    void processBiasVisualization(std::vector<float>& biases, ImDrawList* drawList, LayoutData const& layout);
    void processNeuronSelectionButtons(SelectionData& selectionData, LayoutData& layout);
    void processBiasSliders(std::vector<float>& biases, LayoutData const& layout);
    void processActivationFunctions(std::vector<ActivationFunction>& activationFunctions, ImDrawList* drawList, LayoutData const& layout);
    void drawConnectionWeightLines(std::vector<float>& connectionWeights, ImDrawList* drawList, LayoutData const& layout);
    void drawChannelWeightLines(std::vector<NeuralNetWeight>& weights, ImDrawList* drawList, LayoutData const& layout);
    void processActionButtons(std::vector<NeuralNetWeight>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

    static void pushDefaultColors();
    static void pushHighlightingColors();
    static void popColors();
    static ImColor calcWeightColor(float value);
    static ImColor calcBiasColor(float value);

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
