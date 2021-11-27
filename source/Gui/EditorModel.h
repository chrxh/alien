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

    RealVector2D getDeltaExtCenterPos() const;
    RealVector2D getDeltaExtCenterVel() const;

private:
    SelectionShallowData _origSelectionShallowData;
    SelectionShallowData _selectionShallowData;
};