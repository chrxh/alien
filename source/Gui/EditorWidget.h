#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "EditorModel.h"
#include "MainLoopEntity.h"

class EditorWidget : public MainLoopEntity
{
    MAKE_SINGLETON(EditorWidget);

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
