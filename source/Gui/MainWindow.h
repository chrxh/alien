#pragma once

#include <Network/Definitions.h>

#include <EngineInterface/Definitions.h>

#include <PersisterInterface/Definitions.h>

#include "Definitions.h"

class _MainWindow
{
public:
    _MainWindow();
    void mainLoop();
    void shutdown();

private:
    void initGlfwAndOpenGL();
    void initGlad();
    void initFileDialogs();
};
