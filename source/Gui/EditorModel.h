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

    CellOrParticleDescription getInspectedEntity(uint64_t id) const;
    void addInspectedEntity(CellOrParticleDescription const& entity);
    void setInspectedEntityById(std::unordered_map<uint64_t, CellOrParticleDescription> const& inspectedEntityById);

private:
    SimulationController _simController;
    SelectionShallowData _selectionShallowData;

    std::vector<CellOrParticleDescription> _entitiesToInspect;
    std::unordered_map<uint64_t, CellOrParticleDescription> _inspectedEntityById;
};