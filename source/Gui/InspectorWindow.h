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
    void showCellInOutChannelTab(CellDescription& cell);
    void showTokenTab(CellDescription& cell, int tokenIndex);
    void showTokenMemorySection(
        int address,
        int numBytes,
        int& currentMemoryEditIndex,
        std::vector<std::string> const& symbols = {});
    void showCompilationResult(CompilationResult const& compilationResult);

    void processParticle(ParticleDescription particle);

    void showScannerTableContent();
    void showCommunicationTableContent();
    void showDigestionTableContent();
    void showConstructionTableContent();
    void showMuscleTableContent();
    void showSensorTableContent();

    float calcWindowWidth() const;
    void addToken(CellDescription& cell);
    void delToken(CellDescription& cell, int index);

private:
    std::shared_ptr<MemoryEditor> _cellDataMemoryEdit;
    std::shared_ptr<MemoryEditor> _cellInstructionMemoryEdit;
    std::vector<std::shared_ptr<MemoryEditor>> _tokenMemoryEdits;

    std::shared_ptr<CompilationResult> _lastCompilationResult;
    SimulationController _simController;
    Viewport _viewport; 
    EditorModel _editorModel;
    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    char _cellCode[1024 * 16];
    char _cellMemory[256];

    char _cellDescription[1024*16];

    char _tokenMemory[256];
};
