#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SelectionShallowData.h>

#include "Definitions.h"
#include "InspectionWindow.h"

using EditTool = int;
enum EditTool_
{
    EditTool_Select,
    EditTool_Scissors,
    EditTool_Object,
    EditTool_Rectangle,
    EditTool_Hexagon,
    EditTool_Disc,
    EditTool_Line,
    EditTool_Curve,
    EditTool_Polygon,
    EditTool_Freehand,
    EditTool_Force
};

struct SelectionBounds
{
    RealVector2D center;
    RealVector2D velocity;
    RealVector2D topLeft;
    RealVector2D bottomRight;
};

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
    ExtendedObjectOrEnergyDesc getInspectedEntity(uint64_t id) const;
    void addInspectedEntity(ExtendedObjectOrEnergyDesc const& entity);
    void setInspectedEntities(std::vector<ExtendedObjectOrEnergyDesc> const& inspectedEntities);
    bool areEntitiesInspected() const;

    void setDefaultColorCode(int value);
    int getDefaultColorCode() const;

    EditTool getTool() const;
    void setTool(EditTool value);

    // Holding SHIFT inverts the scope temporarily
    bool isApplyToNetworks() const;
    bool isApplyToNetworksPersistent() const;
    void setApplyToNetworks(bool value);
    void setScopeInvertedTemporarily(bool value);

    bool isGlueOnContact() const;
    void setGlueOnContact(bool value);

    SelectionBounds getSelectionBounds(bool includeClusters) const;

private:
    SelectionShallowData _selectionShallowData;

    std::unordered_map<uint64_t, ExtendedObjectOrEnergyDesc> _inspectedEntityById;

    int _defaultColorCode = 0;

    EditTool _tool = EditTool_Select;
    bool _applyToNetworks = true;
    bool _scopeInvertedTemporarily = false;
    bool _glueOnContact = false;
};
