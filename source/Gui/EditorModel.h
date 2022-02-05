#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "Definitions.h"
#include "InspectorWindow.h"

class _EditorModel
{
public:
    _EditorModel(SimulationController const& simController);

    SelectionShallowData const& getSelectionShallowData() const;
    void update();

    bool isSelectionEmpty() const;
    bool isCellSelectionEmpty() const;
    void clear();

    std::vector<CellOrParticleDescription> fetchEntitiesToInspect();
    void inspectEntities(std::vector<CellOrParticleDescription> const& entities);

    bool existsInspectedEntity(uint64_t id) const;
    CellOrParticleDescription getInspectedEntity(uint64_t id) const;
    void addInspectedEntity(CellOrParticleDescription const& entity);
    void setInspectedEntities(std::vector<CellOrParticleDescription> const& inspectedEntities);

    void setDrawMode(bool value);
    bool isDrawMode() const;

    void setDefaultColorCode(int value);
    int getDefaultColorCode() const;

    void setRolloutToClusters(bool value);
    bool isRolloutToClusters() const;

private:
    SimulationController _simController;
    SelectionShallowData _selectionShallowData;

    std::vector<CellOrParticleDescription> _entitiesToInspect;
    std::unordered_map<uint64_t, CellOrParticleDescription> _inspectedEntityById;
    bool _drawMode = false;
    int _defaultColorCode = 0;
    bool _rolloutToClusters = true;
};