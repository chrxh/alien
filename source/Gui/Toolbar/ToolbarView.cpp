#include "ToolbarView.h"

#include "Gui/SettingsT.h"

ToolbarView::ToolbarView(QWidget * parent) : QWidget(parent)
{
	ui.setupUi(this);
	setStyleSheet(BUTTON_STYLESHEET);

}

ToolbarView::~ToolbarView()
{
	
}
