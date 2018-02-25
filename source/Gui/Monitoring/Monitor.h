#pragma once

#include <QMainWindow>

namespace Ui {
class Monitor;
}

class Monitor : public QMainWindow
{
    Q_OBJECT

public:
    Monitor(QWidget *parent = nullptr);
    virtual ~Monitor();

    void update (QMap< QString, qreal > data);

Q_SIGNALS:
    void closed ();

protected:
    bool event(QEvent* event);

private:
    Ui::Monitor *ui;
};

