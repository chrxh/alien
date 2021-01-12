#include "GettingStartedWindow.h"
#include "ui_GettingStartedWindow.h"
#include "Settings.h"

GettingStartedWindow::GettingStartedWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GettingStartedWindow)
{
    ui->setupUi(this);
    ui->textBrowser->setSource(QUrl("qrc:///GettingStartedWindow.html"));

    ui->checkBoxShowingAfterStartup->setChecked(
        GuiSettings::getSettingsValue(Const::GettingStartedWindowKey, Const::GettingStartedWindowKeyDefault));
}

GettingStartedWindow::~GettingStartedWindow()
{
    delete ui;
}

bool GettingStartedWindow::event(QEvent* event)
{
    if (event->type() == QEvent::Close) {
        GuiSettings::setSettingsValue(Const::GettingStartedWindowKey, ui->checkBoxShowingAfterStartup->isChecked());
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}


