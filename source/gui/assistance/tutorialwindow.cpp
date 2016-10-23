#include "tutorialwindow.h"
#include "ui_tutorialwindow.h"

#include "global/globalfunctions.h"


TutorialWindow::TutorialWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::TutorialWindow)
{
    ui->setupUi(this);
    ui->textBrowser->setSource(QUrl("qrc:///tutorial/tutorial.html"));
//    ui->webView->load(QUrl("qrc:///tutorial/tutorial.html"));
//    ui->webView->show();

}

TutorialWindow::~TutorialWindow()
{
    delete ui;
}

bool TutorialWindow::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        emit closed();
    }
    QMainWindow::event(event);
    return false;
}
