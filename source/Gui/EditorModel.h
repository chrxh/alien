#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "Definitions.h"

class _EditorModel
{
public:
    _EditorModel();

    SelectionShallowData const& getSelectionShallowData() const;
    void setSelectionShallowData(SelectionShallowData const& value);
    void setOrigSelectionShallowData(SelectionShallowData const& value);
    void clear();
    bool isSelectionEmpty() const;

    RealVector2D getClusterCenterPosDelta() const;
    RealVector2D getClusterCenterVelDelta() const;
    RealVector2D getCenterPosDelta() const;
    RealVector2D getCenterVelDelta() const;

private:
    SelectionShallowData _origSelectionShallowData;
    SelectionShallowData _selectionShallowData;
};