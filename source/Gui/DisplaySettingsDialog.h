#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _DisplaySettingsDialog
{
public:
    _DisplaySettingsDialog(GLFWwindow* window);

    void process();

    void show();

private:
    GLFWwindow* _window;
    bool _show = false;
    int _currentDisplaySize = 0;
};