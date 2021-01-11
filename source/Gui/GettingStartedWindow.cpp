#include "GettingStartedWindow.h"
#include "ui_GettingStartedWindow.h"


GettingStartedWindow::GettingStartedWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GettingStartedWindow)
{
    ui->setupUi(this);
    ui->textBrowser->setSource(QUrl("qrc:///GettingStartedWindow.html"));
}

GettingStartedWindow::~GettingStartedWindow()
{
    delete ui;
}

bool GettingStartedWindow::event(QEvent* event)
{
    if (event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}


