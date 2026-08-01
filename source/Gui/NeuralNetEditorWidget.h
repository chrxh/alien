#pragma once

#include <optional>
#include <unordered_map>

#include <imgui.h>

#include <EngineInterface/GenomeDesc.h>

#include "CellFunctionChannels.h"
#include "Definitions.h"

class _NeuralNetEditorWidget
{
public:
    static NeuralNetEditorWidget create();

    // Live activity values shown in the graph when a simulated cell (instead of a genome node) is edited
    struct LiveData
    {
        std::vector<float> memoryActivities;
        float energy = 0;
        bool attacked = false;
        int age = 0;
        float speed = 0;
    };

    void process(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<float>& connectionWeights,
        std::vector<CellFunctionModule> const& cellFunctionModules = {},
        std::optional<LiveData> const& liveData = std::nullopt);

private:
    _NeuralNetEditorWidget();

    struct SelectionData
    {
        int inputIndex = 0;
        int outputIndex = 0;
    };

    // Horizontal extent of the block of one cell function, located outside of the outgoing block
    struct CellFunctionLane
    {
        float minX = 0;
        float maxX = 0;
    };

    struct LayoutData
    {
        ImVec2 inputNodePos[NEURAL_NET_INPUTS];
        ImVec2 outputNodePos[NEURAL_NET_OUTPUTS];
        float leftBlockMinX = 0;
        float leftBlockMaxX = 0;
        float rightBlockMinX = 0;
        float rightBlockMaxX = 0;
        std::vector<CellFunctionLane> cellFunctionLanes;
    };

    struct GraphGeometry
    {
        ImVec2 origin;
        float width = 0;
        float groupBlockGapMinX = 0;
        float groupBlockGapWidth = 0;
    };

    void processConnectionWeightSliders(std::vector<float>& connectionWeights);
    GraphGeometry processGraph(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        SelectionData& selectionData,
        std::optional<LiveData> const& liveData);
    void drawGroupBlock(ImDrawList* drawList, std::string const& name, ImColor const& color, ImVec2 const& min, ImVec2 const& max, bool leftSide);
    void drawGroupBlocks(ImDrawList* drawList, LayoutData const& layout);
    void drawCellFunctionBlocks(
        ImDrawList* drawList,
        LayoutData const& layout,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        SelectionData const& selectionData);
    void drawWeightCurves(std::vector<NeuralNetWeight>& weights, SelectionData const& selectionData, ImDrawList* drawList, LayoutData const& layout);
    void drawInputNodes(SelectionData& selectionData, ImDrawList* drawList, LayoutData const& layout, std::optional<LiveData> const& liveData);
    void drawOutputNodes(
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        SelectionData& selectionData,
        ImDrawList* drawList,
        LayoutData const& layout);
    void processInspectorCard(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        SelectionData& selectionData,
        GraphGeometry const& graphGeometry);
    void processActionButtons(std::vector<NeuralNetWeight>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

    static ImColor calcWeightColor(float value, float alpha);
    static float calcInputNodeMargin();
    static float calcOutputNodeMargin();
    static float calcCellFunctionLaneWidth(CellFunctionModule const& cellFunctionModule);
    static float calcCellFunctionAreaWidth(std::vector<CellFunctionModule> const& cellFunctionModules);
    static std::string getInputLabel(int inputIndex);
    static std::string getOutputLabel(int outputIndex);

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
