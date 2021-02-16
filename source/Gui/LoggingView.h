#pragma once

#include <QWidget>

#include "Definitions.h"

namespace Ui
{
    class LoggingView;
}

class LoggingView : public QWidget
{
    Q_OBJECT

public:
    LoggingView(QWidget* parent = nullptr);
    virtual ~LoggingView();

    void setNewLogMessage(std::string const& message);

private:
    Ui::LoggingView* ui;
};
