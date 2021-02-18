#pragma once
#include <QObject>

#include "Base/LoggingService.h"

#include "Definitions.h"

class LoggingController
    : public QObject
{
    Q_OBJECT

public:
    LoggingController(QWidget* parent = nullptr);
    virtual ~LoggingController() = default;

    QWidget* getWidget() const;

private:
    GuiLogger* _guiLogger = nullptr;
    LoggingView* _view = nullptr;
};
