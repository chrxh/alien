#pragma once

#include <QMainWindow>

namespace Ui {
    class GettingStartedWindow;
}

class GettingStartedWindow : public QMainWindow
{
    Q_OBJECT

public:
    GettingStartedWindow(QWidget *parent = 0);
    virtual ~GettingStartedWindow();

    Q_SIGNAL void closed();

protected:
    bool event(QEvent* event);

private:
    Ui::GettingStartedWindow *ui;
};

