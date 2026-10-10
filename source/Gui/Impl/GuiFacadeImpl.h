#pragma once

#include <Gui/Interface/GuiFacade.h>

#include "Definitions.h"

// User interface with Dear ImGui
class _GuiFacadeImpl : public _GuiFacade
{
public:
    static void set(GuiFacade const& instance);

    void setup() override;
    void runMainLoop() override;
    void shutdown() override;

private:
    MainWindow _mainWindow;
};
