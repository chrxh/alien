#pragma once

#include <QTime>
#include <QCoreApplication>

class QtHelper
{
public:
    static void processEventsForMilliSec(int millisec)
    {
        auto const timeout = QTime::currentTime().addMSecs(millisec);
        while (QTime::currentTime() < timeout) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
        }
    }
};