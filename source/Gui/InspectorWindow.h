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
        RealVector2D const& initialPos);
    ~_InspectorWindow();

    void process();

    bool isClosed() const;
    uint64_t getId() const;

private:
    bool isCell() const;
    std::string generateTitle() const;

    void processCell(CellDescription cell);
    void showCellGeneralTab(CellDescription& cell);
    void showCellFunctionTab(CellDescription& cell);

    void showConstructorContent(ConstructorDescription& constructor);
    void showAttackerContent(AttackerDescription& attacker);
    void showTransmitterContent(TransmitterDescription& transmitter);
    void showMuscleContent(MuscleDescription& muscle);

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
};
