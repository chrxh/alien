#pragma once
#include <QObject>

#include "Base/LoggingService.h"

#include "Definitions.h"

class LoggingController
    : public QObject
    , public LoggingCallBack
{
    Q_OBJECT

public:
    LoggingController(QWidget* parent = nullptr);
    virtual ~LoggingController() = default;

    QWidget* getWidget() const;

    void newLogMessage(std::string const& message) override;

private:
    LoggingView* _view = nullptr;
};
