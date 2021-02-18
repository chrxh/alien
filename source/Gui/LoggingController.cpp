#include "LoggingController.h"

#include "LoggingView.h"
#include "GuiLogger.h"

LoggingController::LoggingController(QWidget* parent /*= nullptr*/)
    : QObject(parent)
{
    _view = new LoggingView(parent);
    _guiLogger = new GuiLogger(_view);
}

QWidget* LoggingController::getWidget() const
{
    return _view;
}
