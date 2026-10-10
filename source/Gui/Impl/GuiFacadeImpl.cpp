#include "GuiFacadeImpl.h"

#include "MainWindow.h"

void _GuiFacadeImpl::set(GuiFacade const& instance)
{
    _instance = instance;
}

void _GuiFacadeImpl::setup()
{
    _mainWindow = std::make_shared<_MainWindow>();
}

void _GuiFacadeImpl::runMainLoop()
{
    _mainWindow->mainLoop();
}

void _GuiFacadeImpl::shutdown()
{
    _mainWindow->shutdown();
    _mainWindow.reset();
}
