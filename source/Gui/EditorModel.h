#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SelectionShallowData.h>

#include "Definitions.h"
#include "InspectorWindow.h"

class EditorModel
{
    MAKE_SINGLETON(EditorModel);

public:
    void setup();

    SelectionShallowData const& getSelectionShallowData() const;
    void update();

    bool isSelectionEmpty() const;
    bool isCellSelectionEmpty() const;
    void clear();

    bool existsInspectedEntity(uint64_t id) const;
    ExtendedCellOrEnergyDescription getInspectedEntity(uint64_t id) const;
    void addInspectedEntity(ExtendedCellOrEnergyDescription const& entity);
    void setInspectedEntities(std::vector<ExtendedCellOrEnergyDescription> const& inspectedEntities);
    bool areEntitiesInspected() const;

    void setPencilWidth(float value);
    float getPencilWidth() const;

    void setDefaultColorCode(int value);
    int getDefaultColorCode() const;

    void setForceNoRollout(bool value);

    void setRolloutToClusters(bool value);
    bool isRolloutToClusters() const;

private:
    SelectionShallowData _selectionShallowData;

    std::unordered_map<uint64_t, ExtendedCellOrEnergyDescription> _inspectedEntityById;

    float _pencilWidth = 3.0f;
    int _defaultColorCode = 0;
    bool _rolloutToClusters = true;
    bool _forceNoRollout = false;
};
