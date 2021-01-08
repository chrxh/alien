#pragma once

#include <QMainWindow>

namespace Ui {
    class StartWindow;
}

class StartWindow : public QMainWindow
{
    Q_OBJECT

public:
    StartWindow(QWidget *parent = 0);
    virtual ~StartWindow();

    Q_SIGNAL void closed();

protected:
    bool event(QEvent* event);

private:
    Ui::StartWindow *ui;
};

