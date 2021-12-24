#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"
#include "InspectorWindow.h"

class _EditorModel
{
public:
    _EditorModel(SimulationController const& simController);

    SelectionShallowData const& getSelectionShallowData() const;
    void update();

    bool isSelectionEmpty() const;
    void clear();

    std::vector<CellOrParticleDescription> fetchEntitiesToInspect();
    void inspectEntities(std::vector<CellOrParticleDescription> const& entities);

private:
    SimulationController _simController;
    SelectionShallowData _selectionShallowData;

    std::vector<CellOrParticleDescription> _entitiesToInspect;
};