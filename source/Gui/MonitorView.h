#pragma once

#include <QWidget>

#include "Definitions.h"

namespace Ui {
class MonitorView;
}

class MonitorView
	: public QWidget
{
    Q_OBJECT

public:
    MonitorView(QWidget *parent = nullptr);
    virtual ~MonitorView();

	void init(MonitorDataSP const& model);
		
	void update();

	Q_SIGNAL void closed ();

protected:
    bool event(QEvent* event);

private:
	QString generateString() const;

	MonitorDataSP _model;
    Ui::MonitorView *ui;
};

