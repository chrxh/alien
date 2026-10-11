#pragma once

#include "Definitions.h"

// Manages the main window with the user interface. The implementation encapsulates the GUI library.
class _GuiFacade
{
public:
    virtual ~_GuiFacade() = default;

    static GuiFacade get();

    // Requires the facades of the other components
    virtual void setup() = 0;

    // Returns when the main window is closed
    virtual void runMainLoop() = 0;

    virtual void shutdown() = 0;

protected:
    static GuiFacade _instance;
};
