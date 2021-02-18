#include "GuiLogger.h"

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "LoggingView.h"

GuiLogger::GuiLogger(LoggingView* view)
    : QObject(view)
    , _view(view)
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->registerCallBack(this);
}

void GuiLogger::newLogMessage(std::string const& message)
{
    _view->setNewLogMessage(message);
}
