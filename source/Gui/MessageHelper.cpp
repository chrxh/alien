#include <QProgressDialog>
#include <QCoreApplication>
#include <QTime>

#include "MessageHelper.h"

QWidget * MessageHelper::getProgress(std::string message, QWidget * parent)
{
    auto const progress = new QProgressDialog(QString::fromStdString(message), QString(), 0, 0, parent);
    progress->setWindowModality(Qt::WindowModal);
    progress->show();

    QCoreApplication::processEvents(QEventLoop::AllEvents);

    return progress;
}
