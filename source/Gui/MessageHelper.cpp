#include <QProgressDialog>
#include <QCoreApplication>
#include <QTime>

#include "MessageHelper.h"

QWidget * MessageHelper::createProgressDialog(std::string message, QWidget * parent)
{
    auto const progress = new QProgressDialog(QString::fromStdString(message), QString(), 0, 0, parent);
    progress->setWindowFlags(progress->windowFlags() & ~Qt::WindowCloseButtonHint);
    progress->setModal(true);
    progress->show();

    QCoreApplication::processEvents(QEventLoop::AllEvents);

    return progress;
}
