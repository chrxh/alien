#pragma once

#include <Base/Interface/Singleton.h>

#include <Network/Interface/Definitions.h>
#include <Network/Interface/NetworkResourceTreeTO.h>

#include "AlienDialog.h"
#include "Definitions.h"
#include "ResourcePreviewWidget.h"

class ReplaceSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ReplaceSimulationDialog);

public:
    void open(NetworkResourceType resourceType, BrowserLeaf const& leaf);

private:
    ReplaceSimulationDialog();

    void processIntern() override;

    void onReplace();

    NetworkResourceType _resourceType = NetworkResourceType_Simulation;
    BrowserLeaf _leaf;
    ResourcePreviewWidget _preview;
};
