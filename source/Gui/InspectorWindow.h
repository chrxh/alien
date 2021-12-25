#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

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
    void processCellPropertyTab(CellDescription& cell);
    void processCodeTab(CellDescription& cell);

    void processParticle(ParticleDescription particle);

private:
    SimulationController _simController;
    Viewport _viewport; 
    EditorModel _editorModel;
    RealVector2D _initialPos;

    bool _on = true;
    uint64_t _entityId = 0;
    char _sourcecode[1024 * 16];
};
