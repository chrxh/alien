#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineImpl/Definitions.h"
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
    void showCellCodeTab(CellDescription& cell);
    void showCellMemoryTab(CellDescription& cell);
    void showTokenTab(TokenDescription& cell, int index);
    void showCompilationResult(CompilationResult const& compilationResult);

    void processParticle(ParticleDescription particle);

    float calcWindowWidth() const;

private:
    boost::shared_ptr<MemoryEditor> _cellDataMemoryEdit;
    boost::shared_ptr<MemoryEditor> _cellInstructionMemoryEdit;
    std::vector<boost::shared_ptr<MemoryEditor>> _tokenMemoryEdits;

    boost::shared_ptr<CompilationResult> _lastCompilationResult;
    SimulationController _simController;
    Viewport _viewport; 
    EditorModel _editorModel;
    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    char _cellCode[1024 * 16];
    char _cellMemory[256];

    char _cellName[1024];
    char _cellDescription[1024*16];

    char _tokenMemory[256];
};
