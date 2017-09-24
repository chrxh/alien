#include "ToolbarView.h"

#include "Gui/Settings.h"

ToolbarView::ToolbarView(QWidget * parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(BUTTON_STYLESHEET);
}

void ToolbarView::init(IntVector2D const & upperLeftPosition)
{
	setGeometry(upperLeftPosition.x, upperLeftPosition.y, width(), height());
}
