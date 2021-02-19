#pragma once
#include <QObject>

#include "Base/LoggingService.h"
#include "Definitions.h"

class GuiLogger
    : public QObject
    , public LoggingCallBack
{
    Q_OBJECT

public:
    GuiLogger(LoggingView* view);
    virtual ~GuiLogger() = default;

    void newLogMessage(Priority priority, std::string const& message) override;

private:
    LoggingView* _view = nullptr;
};
