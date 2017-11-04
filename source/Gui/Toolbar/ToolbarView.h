#pragma once

#include <QWidget>

#include "Gui/Definitions.h"

#include "ui_ToolbarView.h"

class ToolbarView
	: public QWidget
{
	Q_OBJECT
public:
	ToolbarView(QWidget * parent = nullptr);
	virtual ~ToolbarView() = default;

	void init(IntVector2D const& upperLeftPosition, ToolbarController* _controller);

	void setEnableDeleteSelections(bool enable);
	void setEnableAddToken(bool enable);
	void setEnableDeleteToken(bool enable);

private:
	Ui::Toolbar ui;

	ToolbarController* _controller = nullptr;
};
