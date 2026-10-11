#pragma once

#include <Base/Singleton.h>

#include <Network/Definitions.h>
#include <Network/NetworkResourceTreeTO.h>

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
