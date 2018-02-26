#pragma once

#include <QMainWindow>

namespace Ui {
class MonitorView;
}

class MonitorView
	: public QMainWindow
{
    Q_OBJECT

public:
    MonitorView(QWidget *parent = nullptr);
    virtual ~MonitorView();

    void update (QMap< QString, qreal > data);

	Q_SIGNAL void closed ();

protected:
    bool event(QEvent* event);

private:
    Ui::MonitorView *ui;
};

