#include <QProgressDialog>

#include "MessageHelper.h"

QWidget * MessageHelper::getProgress(std::string message, QWidget * parent)
{
    auto const progress = new QProgressDialog(QString::fromStdString(message), QString(), 0, 0, parent);
    progress->setModal(false);
    progress->show();
    return progress;
}
