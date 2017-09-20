#pragma once
#include <QWidget>

#include "Gui/Definitions.h"

class ToolbarController
	: public QObject
{
	Q_OBJECT
public:
	ToolbarController(QWidget * parent = nullptr);
	~ToolbarController();

	ToolbarContext* getContext() const;

private:
	ToolbarContext* _context = nullptr;
};
