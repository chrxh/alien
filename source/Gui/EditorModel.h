#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"
#include "InspectorWindow.h"

class _EditorModel
{
public:
    _EditorModel(SimulationController const& simController, Viewport const& viewport);

    SelectionShallowData const& getSelectionShallowData() const;
    void update();

    bool isSelectionEmpty() const;
    void clear();

    std::vector<InspectorWindow> const& getInspectorWindows() const;
    void inspectEntities(std::vector<CellOrParticleDescription> const& entities);

private:
    SimulationController _simController;
    Viewport _viewport;
    SelectionShallowData _selectionShallowData;

    std::vector<InspectorWindow> _inspectorWindows;
};