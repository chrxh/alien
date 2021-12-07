#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _EditorModel
{
public:
    _EditorModel(SimulationController const& simController);

    SelectionShallowData const& getSelectionShallowData() const;
    void update();

    bool isSelectionEmpty() const;
    void clear();
private:
    SimulationController _simController;
    SelectionShallowData _selectionShallowData;
};