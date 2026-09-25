#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "EditorModel.h"
#include "MainLoopEntity.h"

class EditToolbar : public MainLoopEntity
{
    MAKE_SINGLETON(EditToolbar);

private:
    void init() override {}
    void process() override;
    void shutdown() override {}

    void processDock();
    void processToolOptions();
    void processShortcuts();

    void selectTool(EditTool tool);

    float _dockTop = 0;
};
