#pragma once

#include <QWidget>

#include "Gui/Definitions.h"

#include "ui_ToolbarView.h"

class ToolbarView
	: public QWidget
{
	Q_OBJECT
public:
	ToolbarView(IntVector2D const& upperLeftPosition, QWidget * parent = nullptr);
	~ToolbarView();

private:
	Ui::Toolbar ui;
};
