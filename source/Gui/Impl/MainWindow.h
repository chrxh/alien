#pragma once

#include <Network/Interface/Definitions.h>

#include <Engine/Interface/Definitions.h>

#include <Persister/Interface/Definitions.h>

#include "Definitions.h"

class _MainWindow
{
public:
    _MainWindow();
    void mainLoop();
    void shutdown();

private:
    void initGlfwAndRendering();
    void initFileDialogs();
};
