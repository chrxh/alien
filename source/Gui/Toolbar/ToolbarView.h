#pragma once

#include <QWidget>

#include "ui_ToolbarView.h"

class ToolbarView
	: public QWidget
{
	Q_OBJECT
public:
	ToolbarView(QWidget * parent = nullptr);
	~ToolbarView();

private:
	Ui::Toolbar ui;
};
