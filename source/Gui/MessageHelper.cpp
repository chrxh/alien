#include <QProgressDialog>
#include <QCoreApplication>
#include <QTime>

#include "MessageHelper.h"

QWidget * MessageHelper::createProgressDialog(std::string message, QWidget * parent)
{
    auto const progress = new QProgressDialog(QString::fromStdString(message), QString(), 0, 0, parent);
    progress->setModal(false);
    progress->show();

    QCoreApplication::processEvents(QEventLoop::AllEvents);

    return progress;
}
