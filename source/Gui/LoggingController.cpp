#include "LoggingController.h"

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"
#include "LoggingView.h"

LoggingController::LoggingController(QWidget* parent /*= nullptr*/)
    : QObject(parent)
{
    _view = new LoggingView(parent);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->registerCallBack(this);
}

QWidget* LoggingController::getWidget() const
{
    return _view;
}

void LoggingController::newLogMessage(std::string const& message)
{
    _view->setNewLogMessage(message);
}
