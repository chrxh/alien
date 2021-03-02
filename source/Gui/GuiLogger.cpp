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

GuiLogger::~GuiLogger()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->unregisterCallBack(this);
}

void GuiLogger::newLogMessage(Priority priority, std::string const& message)
{
    if (Priority::Important == priority) {
        _view->setNewLogMessage(message);
    }
}
