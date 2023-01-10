#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

struct MemoryEditor;
struct CompilationResult;

class _InspectorWindow
{
public:
    _InspectorWindow(
        SimulationController const& simController,
        Viewport const& viewport,
        EditorModel const& editorModel,
        GenomeEditorWindow const& genomeEditorWindow,
        uint64_t entityId,
        RealVector2D const& initialPos,
        bool selectGenomeTab);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isCell() const;
    std::string generateTitle() const;

    void processCell(CellDescription cell);
    void showCellBaseTab(CellDescription& cell);
    void showCellFunctionTab(CellDescription& cell);
    void showCellFunctionPropertiesTab(CellDescription& cell);
    void showCellGenomeTab(CellDescription& cell);
    void showCellMetadataTab(CellDescription& cell);

    void showNerveContent(NerveDescription& nerve);
    void showNeuronContent(NeuronDescription& neuron);
    void showConstructorContent(ConstructorDescription& constructor);
    void showAttackerContent(AttackerDescription& attacker);
    void showTransmitterContent(TransmitterDescription& transmitter);
    void showMuscleContent(MuscleDescription& muscle);
    void showSensorContent(SensorDescription& sensor);

    void showActivityContent(CellDescription& cell);

    void processParticle(ParticleDescription particle);

    float calcWindowWidth() const;

    void validationAndCorrection(CellDescription& cell) const;

private:
    SimulationController _simController;
    Viewport _viewport; 
    EditorModel _editorModel;
    GenomeEditorWindow _genomeEditorWindow;

    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    char _cellCode[1024 * 16];
    char _tokenMemory[256];
    int _selectedInput = 0;
    int _selectedOutput = 0;
    float _genomeZoom = 20.0f;
    bool _selectGenomeTab = false;
};
