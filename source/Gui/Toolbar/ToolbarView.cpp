#include "ToolbarView.h"

#include "Gui/Settings.h"

ToolbarView::ToolbarView(IntVector2D const& upperLeftPosition, QWidget * parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(BUTTON_STYLESHEET);
	setGeometry(upperLeftPosition.x, upperLeftPosition.y, width(), height());
}

ToolbarView::~ToolbarView()
{
	
}
