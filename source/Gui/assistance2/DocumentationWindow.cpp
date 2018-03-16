#include "DocumentationWindow.h"
#include "ui_DocumentationWindow.h"


DocumentationWindow::DocumentationWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DocumentationWindow)
{
    ui->setupUi(this);
    ui->textBrowser->setSource(QUrl("qrc:///tutorial/tutorial.html"));

}

DocumentationWindow::~DocumentationWindow()
{
    delete ui;
}

bool DocumentationWindow::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}
