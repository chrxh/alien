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

	void init(IntVector2D const& upperLeftPosition, ActionHolder* actions, ToolbarController* _controller);

private:
	Ui::Toolbar ui;

	list<QMetaObject::Connection> _connections;

	ToolbarController* _controller = nullptr;
	ActionHolder* _actions = nullptr;
};
