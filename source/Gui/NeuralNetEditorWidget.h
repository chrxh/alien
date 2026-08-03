#pragma once

#include <optional>
#include <unordered_map>

#include <imgui.h>

#include <EngineInterface/GenomeDesc.h>

#include "CellFunctionChannels.h"
#include "Definitions.h"
#include "ModalWindow.h"

class _NeuralNetEditorWidget
{
public:
    static NeuralNetEditorWidget create();

    // Live activity values shown in the graph when a simulated cell (instead of a genome node) is edited
    struct LiveData
    {
        std::vector<float> memory;
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

    // Shows the editor additionally in a modal dialog which edits a copy until it is adopted there
    void openDialog();

private:
    _NeuralNetEditorWidget();

    enum class EditorMode
    {
        Embedded,
        Dialog
    };

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

    void processEditor(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<float>& connectionWeights,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        std::optional<LiveData> const& liveData,
        SelectionData& selectionData,
        EditorMode mode);
    void processDialog(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<float>& connectionWeights,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        std::optional<LiveData> const& liveData,
        SelectionData& selectionData);
    void processConnectionWeightSliders(std::vector<float>& connectionWeights);
    GraphGeometry processGraph(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        SelectionData& selectionData,
        std::optional<LiveData> const& liveData,
        bool narrowLayout,
        float rowSpacing);
    void drawGroupBlock(ImDrawList* drawList, std::string const& name, ImColor const& color, ImVec2 const& min, ImVec2 const& max, bool leftSide);
    void drawGroupBlocks(ImDrawList* drawList, LayoutData const& layout, float rowSpacing);
    void drawCellFunctionBlocks(
        ImDrawList* drawList,
        LayoutData const& layout,
        std::vector<CellFunctionModule> const& cellFunctionModules,
        SelectionData const& selectionData,
        float rowSpacing);
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
        GraphGeometry const& graphGeometry,
        bool narrowLayout);
    void processInspectorCardContent(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        SelectionData const& selectionData,
        float cardWidth,
        float cardHeight);
    void processActionButtons(
        std::vector<NeuralNetWeight>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions,
        EditorMode mode);
    void processNetToolButtons(std::vector<NeuralNetWeight>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions);

    static ImColor calcWeightColor(float value, float alpha);
    // Stretches the rows until the graph fills the given height, but never below its natural spacing
    static float calcGraphRowSpacing(float availableHeight);
    static float calcNetToolButtonsWidth();
    static float calcInspectorCardHeight();

    // Depend on the label font and are therefore no static helpers
    bool isNarrowLayout(float availableWidth, std::vector<CellFunctionModule> const& cellFunctionModules);
    float calcGraphMinWidth(std::vector<CellFunctionModule> const& cellFunctionModules);
    float calcInputNodeMargin();
    float calcOutputNodeMargin();
    float calcCellFunctionLaneWidth(CellFunctionModule const& cellFunctionModule);
    float calcCellFunctionAreaWidth(std::vector<CellFunctionModule> const& cellFunctionModules);
    static std::string getInputLabel(int inputIndex);
    static std::string getOutputLabel(int outputIndex);

    template <typename T>
    SelectionData& getValueRef(std::unordered_map<unsigned int, T>& idToValueMap);

    std::unordered_map<unsigned int, SelectionData> _dataById;

    // Chosen at the beginning of an editor pass, since the dialog labels the graph larger than the embedded editor
    ImFont* _labelFont = nullptr;

    // Upper border of the action button row, taken during an editor pass to keep the floating card above it
    float _actionButtonsMinY = 0;

    struct NetData
    {
        std::vector<NeuralNetWeight> weights;
        std::vector<float> biases;
        std::vector<ActivationFunction> activationFunctions;
    };
    std::optional<NetData> _copiedNet;

    ModalWindow _dialog;

    // The dialog edits a copy of the net so that its changes can be discarded
    NetData _dialogNet;
    std::vector<float> _dialogConnectionWeights;

    // The net can only be copied during an editor pass, therefore the opening is deferred until then
    bool _openDialogRequested = false;
    bool _adopted = false;
};
