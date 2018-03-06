#pragma once

#include <QMainWindow>

#include "Gui/Definitions.h"

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

	void init(MonitorModel const& model);

	void update();

	Q_SIGNAL void closed ();

protected:
    bool event(QEvent* event);

private:
	QString generateString() const;

	MonitorModel _model;
    Ui::MonitorView *ui;
};

