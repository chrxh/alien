#include "StartWindow.h"
#include "ui_StartWindow.h"


StartWindow::StartWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StartWindow)
{
    ui->setupUi(this);
    ui->textBrowser->setSource(QUrl("qrc:///Tutorial/StartWindow.html"));
}

StartWindow::~StartWindow()
{
    delete ui;
}

bool StartWindow::event(QEvent* event)
{
    if (event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}


