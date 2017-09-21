#include "ToolbarView.h"

#include "Gui/Settings.h"

ToolbarView::ToolbarView(QWidget * parent) : QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(BUTTON_STYLESHEET);

}

ToolbarView::~ToolbarView()
{
	
}
